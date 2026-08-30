"""Offline contracts for the explicit-model agentic qualification prepass."""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import raptor_agentic
from core.llm.config import ModelConfig
from core.orchestration.agentic_passes import PrepassResult
from core.orchestration.qualification import (
    discover_active_foundation_qualifications,
    select_active_foundation_qualification,
)


_MODEL = ModelConfig(provider="gemini", model_name="gemini-2.5-flash")


def _context_map(path: Path) -> None:
    path.write_text(json.dumps({
        "sources": [],
        "sinks": [],
        "trust_boundaries": [],
        "meta": {"language": ["C++"], "app_type": "library"},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [],
    }), encoding="utf-8")


def _main_patches(out_dir: Path):
    return [
        patch.object(raptor_agentic, "check_repo_claude_trust", return_value=False),
        patch("core.run.get_output_dir", return_value=out_dir),
        patch("core.run.start_run"),
        patch("core.run.complete_run"),
        patch("core.run.fail_run"),
        patch("core.inventory.binary_oracle_cli.apply_to_config"),
        patch("core.inventory.build_inventory"),
        patch("core.sage.hooks.recall_context_for_scan", return_value=[]),
        patch("packages.llm_analysis.detect_llm_availability", return_value={}),
    ]


def _qualified_foundation_record() -> dict:
    commit = "a" * 40
    tree = "b" * 40
    digest = "c" * 64
    semgrep = {
        "scanner_exit_code": 0,
        "packs_dispatched": 16,
        "packs_succeeded": 16,
        "packs_failed": 0,
        "combined_sarif_valid": True,
        "sandbox_engagement": "engaged",
    }
    return {
        "schema_version": 1,
        "record_kind": "foundation_raptor_qualification",
        "immutable": True,
        "status": "qualified",
        "promotable": True,
        "superseded_by": None,
        "qualified_candidate": {
            "branch": "repair/minimal-cpp-qualification-20260830",
            "commit": commit,
            "tree": tree,
        },
        "execution": {"commit": commit, "tree": tree},
        "direct_semgrep_qualification": {
            "artifact": "direct.json",
            "artifact_sha256": digest,
            "execution_commit": commit,
            "execution_tree": tree,
            **semgrep,
        },
        "integrated_agentic_qualification": {
            "artifact": "integrated.json",
            "artifact_sha256": digest,
            "execution_commit": commit,
            "execution_tree": tree,
            "agentic_exit_code": 0,
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "provider_turn_count": 1,
            "scanner_started": True,
            "understand_prepass": {
                "ran": True,
                "terminal_call_count": 1,
                "context_map_valid": True,
                "semantic_complete": True,
            },
            "semgrep": semgrep,
        },
        "consumed_artifact_hashes": {
            "direct.json": digest,
            "integrated.json": digest,
        },
    }


def _write_qualification(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_historical_qualification_records_are_not_discoverable(tmp_path):
    source_dir = Path(__file__).resolve().parents[3] / "qualification"
    names = (
        "foundation-raptor-qualification-20260830.json",
        "minimal-cpp-direct-semgrep-qualification.json",
        "minimal-cpp-agentic-qualification-failure-20260830.json",
        "minimal-cpp-semgrep-diagnostic.json",
    )
    for name in names:
        (tmp_path / name).write_text((source_dir / name).read_text(), encoding="utf-8")

    assert discover_active_foundation_qualifications(tmp_path) == ()
    assert select_active_foundation_qualification(tmp_path) is None


def test_exactly_one_active_foundation_qualification_is_discoverable(tmp_path):
    record = _qualified_foundation_record()
    path = _write_qualification(tmp_path / "active.json", record)

    assert discover_active_foundation_qualifications(tmp_path) == (path,)
    assert select_active_foundation_qualification(tmp_path) == path


def test_revoked_failed_incomplete_and_misbound_records_are_unselectable(tmp_path):
    revoked = _qualified_foundation_record()
    revoked.update({
        "status": "revoked",
        "promotable": False,
        "invalidation_reason": "stale",
        "invalidated_by_commit": "d" * 40,
    })
    failed = {
        "schema_version": 1,
        "record_kind": "minimal_cpp_agentic_qualification_failure",
        "status": "failed",
        "promotable": False,
    }
    incomplete = _qualified_foundation_record()
    incomplete["direct_semgrep_qualification"]["packs_succeeded"] = 15
    misbound = _qualified_foundation_record()
    misbound["integrated_agentic_qualification"]["execution_tree"] = "d" * 40

    for name, record in (
        ("revoked.json", revoked),
        ("failed.json", failed),
        ("incomplete.json", incomplete),
        ("misbound.json", misbound),
    ):
        _write_qualification(tmp_path / name, record)

    assert discover_active_foundation_qualifications(tmp_path) == ()
    assert select_active_foundation_qualification(tmp_path) is None


def test_multiple_active_foundation_qualifications_fail_closed(tmp_path):
    _write_qualification(tmp_path / "first.json", _qualified_foundation_record())
    _write_qualification(tmp_path / "second.json", _qualified_foundation_record())

    assert len(discover_active_foundation_qualifications(tmp_path)) == 2
    assert select_active_foundation_qualification(tmp_path) is None


def test_threat_model_requires_explicit_model(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / ".git").mkdir()

    with patch.object(sys, "argv", [
        "raptor_agentic.py", "--repo", str(repo), "--threat-model",
    ]):
        with pytest.raises(SystemExit) as exc:
            raptor_agentic.main()

    assert exc.value.code == 2


def test_zero_terminal_context_map_fails_before_scanner(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / ".git").mkdir()
    out_dir = tmp_path / "out"
    observed_models = []

    def fake_prepass(**kwargs):
        observed_models.append(kwargs["model"])
        return PrepassResult(
            ran=False,
            skipped_reason="submit_context_map must be invoked exactly once; observed 0",
        )

    with patch.object(sys, "argv", [
        "raptor_agentic.py", "--repo", str(repo), "--out", str(out_dir),
        "--threat-model", "--model", "gemini-2.5-flash", "--no-codeql",
        "--sandbox", "strict",
    ]):
        with ExitStack() as stack:
            for item in _main_patches(out_dir):
                stack.enter_context(item)
            stack.enter_context(patch(
                "packages.llm_analysis.orchestrator.build_llm_config_from_flags",
                return_value=SimpleNamespace(primary_model=_MODEL),
            ))
            stack.enter_context(patch(
                "core.orchestration.run_understand_prepass", side_effect=fake_prepass,
            ))
            popen = stack.enter_context(patch.object(raptor_agentic.subprocess, "Popen"))
            assert raptor_agentic.main() == 1

    assert observed_models == [_MODEL]
    summary = json.loads((out_dir / "understand-prepass-summary.json").read_text())
    assert summary == {
        "schema_version": 1,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "ran": False,
        "terminal_call_count": 0,
        "context_map_valid": False,
        "semantic_complete": False,
        "failure_class": "terminal_context_map_count",
        "failure_stage": "understand_prepass",
    }
    assert not popen.called


def test_model_resolution_failure_remains_distinct_from_isolation_startup():
    summary = raptor_agentic.build_understand_prepass_summary(
        provider="gemini",
        model="gemini-2.5-flash",
        prepass_result=PrepassResult(ran=False),
        model_resolved=False,
    )

    assert summary["failure_stage"] == "model_resolution"
    assert summary["failure_class"] == "model_resolution_failed"
    assert summary["failure_class"] != "credential_isolation_socket_path_invalid"


def test_successful_canonical_prepass_enters_materialization(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / ".git").mkdir()
    out_dir = tmp_path / "out"
    understand_dir = tmp_path / "understand"
    understand_dir.mkdir()
    context_map = understand_dir / "context-map.json"
    _context_map(context_map)
    observed_models = []

    def fake_prepass(**kwargs):
        observed_models.append(kwargs["model"])
        return PrepassResult(
            ran=True,
            understand_dir=understand_dir,
            context_map_path=context_map,
            checklist_enriched=True,
        )

    materialized = {
        "enabled": True,
        "completed": True,
        "semantic_complete": True,
        "generated_candidates": 0,
    }
    with patch.object(sys, "argv", [
        "raptor_agentic.py", "--repo", str(repo), "--out", str(out_dir),
        "--threat-model-only", "--model", "gemini-2.5-flash", "--no-codeql",
        "--sandbox", "strict",
    ]):
        with ExitStack() as stack:
            for item in _main_patches(out_dir):
                stack.enter_context(item)
            stack.enter_context(patch(
                "packages.llm_analysis.orchestrator.build_llm_config_from_flags",
                return_value=SimpleNamespace(primary_model=_MODEL),
            ))
            stack.enter_context(patch(
                "core.orchestration.run_understand_prepass", side_effect=fake_prepass,
            ))
            materialise = stack.enter_context(patch.object(
                raptor_agentic,
                "_materialise_threat_model_phase",
                return_value=materialized,
            ))
            assert raptor_agentic.main() == 0

    assert observed_models == [_MODEL]
    assert materialise.called
    summary = json.loads((out_dir / "understand-prepass-summary.json").read_text())
    assert summary["provider"] == "gemini"
    assert summary["model"] == "gemini-2.5-flash"
    assert summary["terminal_call_count"] == 1
    assert summary["context_map_valid"] is True
    assert summary["semantic_complete"] is True
    report = json.loads((out_dir / "raptor_agentic_report.json").read_text())
    assert report["outputs"]["understand_prepass_summary"] == "understand-prepass-summary.json"
    assert report["repository"] == "."
    for value in report["outputs"].values():
        if isinstance(value, str):
            assert not Path(value).is_absolute()


def test_threat_model_report_metadata_is_relative_and_redacted(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    phase = {
        "enabled": True,
        "completed": True,
        "semantic_complete": True,
        "threat_model_json": out_dir / "threat-model.json",
        "context_map": tmp_path / "understand" / "context-map.json",
        "stale_files": [tmp_path / "outside.cpp"],
        "skipped_reason": (
            "Bearer top-secret-value at /private/var/folders/abc/opaque-fixture"
        ),
    }

    report = raptor_agentic._report_threat_model_phase(out_dir, phase)

    assert report["threat_model_json"] == "threat-model.json"
    assert report["context_map"] is None
    assert report["stale_file_count"] == 1
    assert "stale_files" not in report
    rendered = json.dumps(report)
    assert "top-secret-value" not in rendered
    assert "/private/var/folders" not in rendered
    assert "opaque-fixture" not in rendered


def test_semgrep_failure_persists_bounded_pack_diagnostics(tmp_path):
    out_dir = tmp_path / "out"
    scan_dir = out_dir / "scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "semgrep-run-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "scanner": {},
        "packs_dispatched": 1,
        "packs_succeeded": 0,
        "packs_failed": 1,
        "all_semgrep_failed": True,
        "aggregate_exit_code": 4,
        "failure_class": None,
        "packs": [{
            "pack_name": "category_auth",
            "config_kind": "local_directory",
            "config_sha256": None,
            "raw_exit_code": -1,
            "sarif_exists": False,
            "sarif_valid": False,
            "stderr_class": "target_staging_error",
            "failure_class": "target_staging_error",
            "bounded_stderr_tail": "staging rejected control character",
            "sandbox_denial_count": 0,
            "proxy_event_count": 0,
        }],
        "combined_sarif_exists": False,
        "combined_sarif_valid": False,
        "sandbox_engagement": {"state": "engaged", "denial_count": 0},
    }), encoding="utf-8")

    with patch.object(
        raptor_agentic, "_semgrep_artifact_state", return_value=(False, False),
    ), patch("core.run.fail_run") as fail_run:
        payload = raptor_agentic.persist_agentic_semgrep_failure(
            out_dir=out_dir,
            return_code=4,
            stdout="request failed: Authorization: Bearer top-secret-value",
            stderr="/private/var/folders/abc/opaque-fixture",
        )

    assert payload["failure_class"] == "aggregate_all_packs_failed"
    assert payload["failed_packs"] == [{
        "pack_name": "category_auth",
        "failure_class": "target_staging_error",
        "raw_exit_code": -1,
        "stderr_tail": "staging rejected control character",
        "sandbox_denial_count": 0,
        "proxy_event_count": 0,
    }]
    rendered = (scan_dir / "semgrep-agentic-failure.json").read_text()
    assert "top-secret-value" not in rendered
    assert "/private/var/folders" not in rendered
    assert "opaque-fixture" not in rendered
    fail_run.assert_called_once()


def test_skipped_sarif_validation_is_rejected(tmp_path):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    (scan_dir / "combined.sarif").write_text(
        '{"version":"2.1.0","runs":[]}', encoding="utf-8",
    )
    (scan_dir / "scan_metrics.json").write_text("{}", encoding="utf-8")

    with patch("core.sarif.parser.validate_sarif", return_value=None):
        assert raptor_agentic._semgrep_artifact_state(scan_dir) == (False, True)


def test_agentic_semgrep_success_rejects_skipped_sarif_validation(tmp_path):
    out_dir = tmp_path / "out"
    scan_dir = out_dir / "scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "semgrep-run-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "scanner": {},
        "packs_dispatched": 1,
        "packs_succeeded": 1,
        "packs_failed": 0,
        "all_semgrep_failed": False,
        "aggregate_exit_code": 0,
        "failure_class": None,
        "packs": [{
            "pack_name": "category_auth",
            "config_kind": "local_directory",
            "config_sha256": None,
            "raw_exit_code": 0,
            "sarif_exists": True,
            "sarif_valid": True,
            "stderr_class": "none",
            "failure_class": None,
            "bounded_stderr_tail": "",
            "sandbox_denial_count": 0,
            "proxy_event_count": 0,
        }],
        "combined_sarif_exists": True,
        "combined_sarif_valid": True,
        "sandbox_engagement": {"state": "engaged", "denial_count": 0},
    }), encoding="utf-8")
    (scan_dir / "combined.sarif").write_text(
        '{"version":"2.1.0","runs":[]}', encoding="utf-8",
    )
    (scan_dir / "scan_metrics.json").write_text("{}", encoding="utf-8")

    with patch("core.sarif.parser.validate_sarif", return_value=None):
        summary, error = raptor_agentic.validate_agentic_semgrep_success(out_dir)

    assert summary is None
    assert error == "scanner_artifact_invalid"
def test_agentic_semgrep_success_rejects_nonzero_aggregate_exit(tmp_path):
    out_dir = tmp_path / "out"
    scan_dir = out_dir / "scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "semgrep-run-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "scanner": {},
        "packs_dispatched": 1,
        "packs_succeeded": 1,
        "packs_failed": 0,
        "all_semgrep_failed": False,
        "aggregate_exit_code": 1,
        "failure_class": None,
        "packs": [{
            "pack_name": "category_auth",
            "config_kind": "local_directory",
            "config_sha256": None,
            "raw_exit_code": 1,
            "sarif_exists": True,
            "sarif_valid": True,
            "stderr_class": "none",
            "failure_class": None,
            "bounded_stderr_tail": "",
            "sandbox_denial_count": 0,
            "proxy_event_count": 0,
        }],
        "combined_sarif_exists": True,
        "combined_sarif_valid": True,
        "sandbox_engagement": {"state": "engaged", "denial_count": 0},
    }), encoding="utf-8")
    (scan_dir / "combined.sarif").write_text(
        '{"version":"2.1.0","runs":[]}', encoding="utf-8",
    )
    (scan_dir / "scan_metrics.json").write_text("{}", encoding="utf-8")

    with patch("core.sarif.parser.validate_sarif", return_value=True):
        summary, error = raptor_agentic.validate_agentic_semgrep_success(out_dir)

    assert summary is None
    assert error == "aggregate_exit_nonzero"


def test_agentic_semgrep_failure_retains_scanner_fallback_class(tmp_path):
    out_dir = tmp_path / "out"
    scan_dir = out_dir / "scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "semgrep-run-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "scanner": {},
        "packs_dispatched": 1,
        "packs_succeeded": 1,
        "packs_failed": 0,
        "all_semgrep_failed": False,
        "aggregate_exit_code": 1,
        "failure_class": "internal_scanner_exception",
        "packs": [{
            "pack_name": "category_auth",
            "config_kind": "local_directory",
            "config_sha256": None,
            "raw_exit_code": 1,
            "sarif_exists": False,
            "sarif_valid": False,
            "stderr_class": "none",
            "failure_class": None,
            "bounded_stderr_tail": "",
            "sandbox_denial_count": 0,
            "proxy_event_count": 0,
        }],
        "combined_sarif_exists": False,
        "combined_sarif_valid": False,
        "sandbox_engagement": {"state": "engaged", "denial_count": 0},
    }), encoding="utf-8")

    with patch.object(raptor_agentic, "_semgrep_artifact_state", return_value=(False, False)), patch("core.run.fail_run"):
        payload = raptor_agentic.persist_agentic_semgrep_failure(
            out_dir=out_dir, return_code=1, stdout="", stderr="",
        )

    assert payload["failure_class"] == "internal_scanner_exception"
    assert payload["scanner_failure_class"] == "internal_scanner_exception"


def test_agentic_semgrep_failure_prints_bounded_diagnostics(capsys):
    raptor_agentic.print_agentic_semgrep_failure({
        "failure_class": "aggregate_all_packs_failed",
        "summary_path": "scan/semgrep-run-summary.json",
        "failed_packs": [{
            "pack_name": "category_auth",
            "failure_class": "invalid_config",
        }],
        "stderr_tail": "bounded scanner stderr",
    })

    rendered = capsys.readouterr().err
    assert "scan/semgrep-run-summary.json" in rendered
    assert "category_auth:invalid_config" in rendered
    assert "bounded scanner stderr" in rendered
