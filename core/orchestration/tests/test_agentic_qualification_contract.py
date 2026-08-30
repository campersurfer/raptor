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
            stdout="Bearer top-secret-value",
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
