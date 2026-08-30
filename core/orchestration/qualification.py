"""Fail-closed discovery for local Foundation qualification evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SARIF_SCHEMA = "engine/schemas/sarif-2.1.0.json"


def _is_hex40(value: object) -> bool:
    return isinstance(value, str) and _HEX40.fullmatch(value) is not None


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _is_exact_int(value: object) -> bool:
    return type(value) is int


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
        and _is_repository_relative_path(evidence.get("artifact"))
        and _is_hex64(evidence.get("artifact_sha256"))
    )


def _is_repository_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _has_schema_binding(evidence: object) -> bool:
    if not isinstance(evidence, dict):
        return False
    schema = evidence.get("sarif_schema")
    return (
        isinstance(schema, dict)
        and schema.get("path") == _CANONICAL_SARIF_SCHEMA
        and _is_hex64(schema.get("sha256"))
    )


def _has_fresh_canary_binding(evidence: object, candidate: dict[str, Any]) -> bool:
    return (
        _has_bound_execution(evidence, candidate)
        and isinstance(evidence, dict)
        and evidence.get("mode") == "fresh_standalone"
        and evidence.get("provider") == "gemini"
        and evidence.get("model") == "gemini-2.5-flash"
        and _is_exact_int(evidence.get("provider_turn_count"))
        and evidence["provider_turn_count"] >= 1
        and _is_exact_int(evidence.get("terminal_call_count"))
        and evidence["terminal_call_count"] == 1
    )


def _has_hash_bound_reused_canary_binding(
    evidence: object, candidate: dict[str, Any]
) -> bool:
    if not isinstance(evidence, dict) or evidence.get("mode") != "hash_bound_reuse":
        return False
    if not (
        _is_repository_relative_path(evidence.get("artifact"))
        and _is_hex64(evidence.get("artifact_sha256"))
    ):
        return False
    prior = evidence.get("prior_attestation")
    candidate_execution = evidence.get("candidate_execution")
    return (
        isinstance(prior, dict)
        and prior.get("artifact") == evidence["artifact"]
        and prior.get("artifact_sha256") == evidence["artifact_sha256"]
        and _is_hex40(prior.get("execution_commit"))
        and _is_hex40(prior.get("execution_tree"))
        and prior.get("provider") == "gemini"
        and prior.get("model") == "gemini-2.5-flash"
        and _is_exact_int(prior.get("provider_turn_count"))
        and prior["provider_turn_count"] >= 1
        and _is_exact_int(prior.get("terminal_call_count"))
        and prior["terminal_call_count"] == 1
        and isinstance(candidate_execution, dict)
        and candidate_execution.get("commit") == candidate["commit"]
        and candidate_execution.get("tree") == candidate["tree"]
        and candidate_execution.get("candidate_code_cannot_affect_canary_outcome") is True
    )


def _has_canary_binding(evidence: object, candidate: dict[str, Any]) -> bool:
    return _has_fresh_canary_binding(evidence, candidate) or _has_hash_bound_reused_canary_binding(
        evidence, candidate
    )


def _has_evidence_paths(record: dict[str, Any]) -> bool:
    paths = record.get("evidence_paths")
    return (
        isinstance(paths, dict)
        and bool(paths)
        and all(
            _is_repository_relative_path(path) and _is_hex64(digest)
            for path, digest in paths.items()
        )
    )


def _has_verified_artifact(evidence_paths: dict[str, Any], evidence: object) -> bool:
    return (
        isinstance(evidence, dict)
        and _is_repository_relative_path(evidence.get("artifact"))
        and _is_hex64(evidence.get("artifact_sha256"))
        and evidence_paths.get(evidence["artifact"]) == evidence["artifact_sha256"]
    )


def _has_verified_schema(evidence_paths: dict[str, Any], evidence: object) -> bool:
    if not _has_schema_binding(evidence):
        return False
    schema = evidence["sarif_schema"]
    return evidence_paths.get(schema["path"]) == schema["sha256"]


def _has_complete_consumed_hashes(record: dict[str, Any], evidence_paths: dict[str, Any]) -> bool:
    consumed = record.get("consumed_artifact_hashes")
    return isinstance(consumed, dict) and consumed == evidence_paths


def _evidence_paths_match_repository(record: dict[str, Any], repo_root: Path) -> bool:
    paths = record.get("evidence_paths")
    if not isinstance(paths, dict):
        return False
    for relative, expected_digest in paths.items():
        if not _is_repository_relative_path(relative) or not _is_hex64(expected_digest):
            return False
        path = repo_root / str(relative)
        try:
            if not path.is_file():
                return False
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if observed != expected_digest:
            return False
    return True


def is_active_foundation_qualification(record: object) -> bool:
    """Return whether one record is complete, current-shaped Foundation evidence.

    Callers must still require exactly one candidate. This predicate never
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
    if (
        not _has_bound_execution(direct, candidate)
        or not _has_successful_semgrep(direct)
        or not _has_schema_binding(direct)
    ):
        return False

    integrated = record.get("integrated_agentic_qualification")
    if not _has_bound_execution(integrated, candidate):
        return False
    if not isinstance(integrated, dict) or (
        integrated.get("agentic_exit_code") != 0
        or integrated.get("provider") != "gemini"
        or integrated.get("model") != "gemini-2.5-flash"
        or not _is_exact_int(integrated.get("provider_turn_count"))
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

    evidence_paths = record.get("evidence_paths")
    canary = record.get("canary_binding")
    return (
        isinstance(evidence_paths, dict)
        and _has_evidence_paths(record)
        and _has_verified_artifact(evidence_paths, direct)
        and _has_verified_schema(evidence_paths, direct)
        and _has_verified_artifact(evidence_paths, integrated)
        and _has_canary_binding(canary, candidate)
        and _has_verified_artifact(evidence_paths, canary)
        and _has_complete_consumed_hashes(record, evidence_paths)
    )


def discover_active_foundation_qualifications(qualification_dir: Path) -> tuple[Path, ...]:
    """Return each complete promotable Foundation record in a directory."""
    matches: list[Path] = []
    repo_root = qualification_dir.resolve().parent
    for path in sorted(qualification_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            is_active_foundation_qualification(record)
            and _evidence_paths_match_repository(record, repo_root)
        ):
            matches.append(path)
    return tuple(matches)


def select_active_foundation_qualification(qualification_dir: Path) -> Path | None:
    """Select exactly one active Foundation record or fail closed."""
    matches = discover_active_foundation_qualifications(qualification_dir)
    return matches[0] if len(matches) == 1 else None
