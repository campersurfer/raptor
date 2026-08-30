"""Fail-closed discovery for local Foundation qualification evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_hex40(value: object) -> bool:
    return isinstance(value, str) and _HEX40.fullmatch(value) is not None


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _has_successful_semgrep(evidence: object) -> bool:
    if not isinstance(evidence, dict):
        return False
    return (
        evidence.get("scanner_exit_code") == 0
        and evidence.get("packs_dispatched") == 16
        and evidence.get("packs_succeeded") == 16
        and evidence.get("packs_failed") == 0
        and evidence.get("combined_sarif_valid") is True
        and evidence.get("sandbox_engagement") == "engaged"
    )


def _has_bound_execution(evidence: object, candidate: dict[str, Any]) -> bool:
    return (
        isinstance(evidence, dict)
        and evidence.get("execution_commit") == candidate["commit"]
        and evidence.get("execution_tree") == candidate["tree"]
        and _is_hex64(evidence.get("artifact_sha256"))
    )


def is_active_foundation_qualification(record: object) -> bool:
    """Return whether one record is complete, current-shaped Foundation evidence.

    Callers must still require exactly one candidate.  This predicate never
    promotes historical, revoked, direct-only, failed, incomplete, or
    execution-misbound evidence.
    """
    if not isinstance(record, dict):
        return False
    if (
        record.get("schema_version") != 1
        or record.get("record_kind") != "foundation_raptor_qualification"
        or record.get("immutable") is not True
        or record.get("status") != "qualified"
        or record.get("promotable") is not True
        or record.get("superseded_by") is not None
    ):
        return False

    candidate = record.get("qualified_candidate")
    if not isinstance(candidate, dict):
        return False
    if not (
        isinstance(candidate.get("branch"), str)
        and _is_hex40(candidate.get("commit"))
        and _is_hex40(candidate.get("tree"))
    ):
        return False

    execution = record.get("execution")
    if not isinstance(execution, dict) or (
        execution.get("commit") != candidate["commit"]
        or execution.get("tree") != candidate["tree"]
    ):
        return False

    direct = record.get("direct_semgrep_qualification")
    if not _has_bound_execution(direct, candidate) or not _has_successful_semgrep(direct):
        return False

    integrated = record.get("integrated_agentic_qualification")
    if not _has_bound_execution(integrated, candidate):
        return False
    if not isinstance(integrated, dict) or (
        integrated.get("agentic_exit_code") != 0
        or integrated.get("provider") != "gemini"
        or integrated.get("model") != "gemini-2.5-flash"
        or not isinstance(integrated.get("provider_turn_count"), int)
        or integrated["provider_turn_count"] < 1
        or integrated.get("scanner_started") is not True
        or not _has_successful_semgrep(integrated.get("semgrep"))
    ):
        return False
    prepass = integrated.get("understand_prepass")
    if not isinstance(prepass, dict) or (
        prepass.get("ran") is not True
        or prepass.get("terminal_call_count") != 1
        or prepass.get("context_map_valid") is not True
        or prepass.get("semantic_complete") is not True
    ):
        return False

    artifact_hashes = record.get("consumed_artifact_hashes")
    return (
        isinstance(artifact_hashes, dict)
        and bool(artifact_hashes)
        and all(_is_hex64(value) for value in artifact_hashes.values())
    )


def discover_active_foundation_qualifications(qualification_dir: Path) -> tuple[Path, ...]:
    """Return each complete promotable Foundation record in a directory."""
    matches: list[Path] = []
    for path in sorted(qualification_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if is_active_foundation_qualification(record):
            matches.append(path)
    return tuple(matches)


def select_active_foundation_qualification(qualification_dir: Path) -> Path | None:
    """Select exactly one active Foundation record or fail closed."""
    matches = discover_active_foundation_qualifications(qualification_dir)
    return matches[0] if len(matches) == 1 else None
