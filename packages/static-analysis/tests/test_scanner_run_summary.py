"""Contract tests for durable, redacted Semgrep diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


_SCANNER_PATH = Path(__file__).parent.parent / "scanner.py"
_spec = importlib.util.spec_from_file_location(
    "static_analysis_scanner_run_summary", _SCANNER_PATH,
)
_scanner_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_spec.loader.exec_module(_scanner_mod)


_EXECUTABLE_IDENTITY = {
    "basename": "semgrep",
    "path_kind": "system",
    "exists": True,
    "executable": True,
    "version": "1.0.0",
    "sha256": "a" * 64,
}


def _write_pack(out_dir: Path, name: str, exit_code: int, stderr: str = "") -> None:
    suffix = _scanner_mod._sanitize_pack_name(name)
    (out_dir / f"semgrep_{suffix}.exit").write_text(str(exit_code), encoding="utf-8")
    (out_dir / f"semgrep_{suffix}.sarif").write_text(
        '{"version":"2.1.0","runs":[]}', encoding="utf-8",
    )
    (out_dir / f"semgrep_{suffix}.stderr.log").write_text(stderr, encoding="utf-8")


class TestSemgrepRunSummary:
    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_all_pack_failure_is_durable_and_redacted(self, _identity, _validate, tmp_path):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        _write_pack(
            tmp_path,
            "category_auth",
            -1,
            "SBPL string contains control character at /private/var/folders/abc/opaque-fixture\n"
            "request failed: Authorization: Bearer top-secret-value",
        )
        (tmp_path / ".sandbox-denials.jsonl").write_text("{}\n{}\n", encoding="utf-8")
        (tmp_path / "proxy-events.jsonl").write_text("{}\n", encoding="utf-8")

        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=4,
            sandbox_engagement_state="engaged",
        )

        assert summary["aggregate_exit_code"] == 4
        assert summary["all_semgrep_failed"] is True
        pack = summary["packs"][0]
        assert pack["failure_class"] == "target_staging_error"
        assert summary["schema_version"] == 2
        assert summary["failure_class"] == "target_staging_error"
        assert summary["failure_class_counts"] == {"target_staging_error": 1}
        assert pack["config_kind"] == "local_file"
        assert pack["config_sha256"]
        assert pack["sandbox_denial_count"] == 2
        assert pack["proxy_event_count"] == 1
        rendered = json.dumps(summary)
        assert "top-secret-value" not in rendered
        assert "/private/var/folders" not in rendered
        assert "opaque-fixture" not in rendered
        assert (tmp_path / "semgrep-run-summary.json").is_file()

        _scanner_mod.cleanup_per_pack_artifacts(tmp_path)
        assert (tmp_path / "semgrep-run-summary.json").is_file()

    @patch.object(_scanner_mod, "validate_sarif", return_value=None)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_skipped_sarif_validation_fails_the_summary(self, _identity, _validate, tmp_path):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        _write_pack(tmp_path, "category_auth", 0)
        (tmp_path / "combined.sarif").write_text(
            '{"version":"2.1.0","runs":[]}', encoding="utf-8",
        )

        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=0,
            sandbox_engagement_state="engaged",
        )

        assert summary["packs"][0]["failure_class"] == "sarif_full_validation_unavailable"
        assert summary["packs"][0]["sarif_validation_status"] == "full_validation_unavailable"
        assert summary["combined_sarif_validation_status"] == "full_validation_unavailable"
        assert summary["all_semgrep_failed"] is True
        assert summary["aggregate_exit_code"] == 4
    @patch.object(_scanner_mod, "validate_sarif", return_value=None)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    @patch.object(
        _scanner_mod,
        "run",
        return_value=(1, '{"version":"2.1.0","runs":[]}', ""),
    )
    def test_unavailable_full_validation_has_one_fail_closed_meaning(
        self, _run, _identity, _validate, tmp_path,
    ):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")

        _, scan_success = _scanner_mod.run_single_semgrep(
            "category_auth",
            str(config),
            tmp_path,
            tmp_path,
            timeout=1,
        )
        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=0,
            sandbox_engagement_state="engaged",
        )

        assert (
            scan_success,
            summary["packs"][0]["failure_class"],
            summary["aggregate_exit_code"],
        ) == (False, "sarif_full_validation_unavailable", 4)

    def test_invalid_unavailable_and_missing_sarif_are_distinct(self, tmp_path):
        sarif = tmp_path / "result.sarif"
        sarif.write_text('{"version":"2.1.0","runs":[]}', encoding="utf-8")
        with patch.object(_scanner_mod, "validate_sarif", return_value=None):
            assert _scanner_mod._sarif_validation_status(sarif) == "full_validation_unavailable"
        with patch.object(_scanner_mod, "validate_sarif", return_value=False):
            assert _scanner_mod._sarif_validation_status(sarif) == "invalid"
        assert _scanner_mod._sarif_validation_status(tmp_path / "missing.sarif") == "missing"

    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_mixed_pack_failures_retain_counts_without_single_cause(
        self, _identity, _validate, tmp_path,
    ):
        first_config = tmp_path / "first.yml"
        second_config = tmp_path / "second.yml"
        first_config.write_text("rules: []\n", encoding="utf-8")
        second_config.write_text("rules: []\n", encoding="utf-8")
        _write_pack(tmp_path, "category_auth", 127, "semgrep: command not found")
        _write_pack(tmp_path, "category_flows", 2, "invalid yaml configuration")

        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[
                ("category_auth", str(first_config)),
                ("category_flows", str(second_config)),
            ],
            aggregate_exit_code=0,
            sandbox_engagement_state="engaged",
        )

        assert summary["failure_class"] == "mixed_pack_failures"
        assert summary["failure_class_source"] == "mixed_pack_failure_classes"
        assert summary["failure_class_counts"] == {
            "invalid_config": 1,
            "semgrep_missing": 1,
        }

    def test_summary_parser_accepts_historical_one_and_current_two(self):
        from raptor_agentic import _validate_semgrep_run_summary

        common = {
            "packs_dispatched": 1,
            "packs_succeeded": 1,
            "packs_failed": 0,
            "all_semgrep_failed": False,
            "aggregate_exit_code": 0,
            "combined_sarif_exists": True,
            "sandbox_engagement": {"state": "engaged", "denial_count": 0},
        }
        version_one = {
            **common,
            "schema_version": 1,
            "combined_sarif_valid": True,
            "packs": [{
                "pack_name": "category_auth",
                "config_kind": "local_file",
                "config_sha256": "a" * 64,
                "raw_exit_code": 0,
                "sarif_exists": True,
                "sarif_valid": True,
                "stderr_class": "none",
                "failure_class": None,
                "bounded_stderr_tail": "",
                "sandbox_denial_count": 0,
                "proxy_event_count": 0,
            }],
        }
        version_two = {
            **common,
            "schema_version": 2,
            "combined_sarif_validation_status": "full_valid",
            "packs": [{
                "pack_name": "category_auth",
                "config_kind": "local_file",
                "config_sha256": "a" * 64,
                "raw_exit_code": 0,
                "sarif_exists": True,
                "sarif_validation_status": "full_valid",
                "failure_class": None,
                "bounded_stderr_tail": "",
                "sandbox_denial_count": 0,
                "proxy_event_count": 0,
            }],
        }

        assert _validate_semgrep_run_summary(version_one) is None
        assert _validate_semgrep_run_summary(version_two) is None
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_default_schema_fully_validates_valid_sarif(self, _identity, tmp_path):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        valid_sarif = json.dumps(
            {
                "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
                "version": "2.1.0",
                "runs": [{"tool": {"driver": {"name": "semgrep"}}}],
            }
        )
        _write_pack(tmp_path, "category_auth", 0)
        (tmp_path / "semgrep_category_auth.sarif").write_text(
            valid_sarif, encoding="utf-8",
        )
        (tmp_path / "combined.sarif").write_text(valid_sarif, encoding="utf-8")

        assert _scanner_mod.validate_sarif(
            tmp_path / "semgrep_category_auth.sarif",
        ) is True

        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=0,
            sandbox_engagement_state="engaged",
        )

        assert summary["packs"][0]["failure_class"] is None
        assert summary["packs"][0]["sarif_validation_status"] == "full_valid"
        assert summary["combined_sarif_validation_status"] == "full_valid"
        assert summary["all_semgrep_failed"] is False
        assert summary["aggregate_exit_code"] == 0

    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_missing_binary_is_not_misclassified_as_config_error(self, _identity, _validate, tmp_path):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        _write_pack(tmp_path, "category_auth", 127, "semgrep: command not found")

        summary = _scanner_mod.build_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=4,
            sandbox_engagement_state="engaged",
        )

        assert summary["packs"][0]["failure_class"] == "semgrep_missing"

    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_invalid_config_is_not_misclassified_as_sandbox_error(self, _identity, _validate, tmp_path):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        _write_pack(tmp_path, "category_auth", -1, "invalid yaml configuration is invalid")

        summary = _scanner_mod.build_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[("category_auth", str(config))],
            aggregate_exit_code=4,
            sandbox_engagement_state="engaged",
        )

        assert summary["packs"][0]["failure_class"] == "invalid_config"

    def test_failure_classes_do_not_collapse_to_sandbox(self):
        classify = _scanner_mod._classify_semgrep_failure
        assert classify(
            -1, "registry cache missing",
            sarif_validation_status="missing", config_kind="registry",
        ) == "registry_cache_missing"
        assert classify(
            -1, "permission denied",
            sarif_validation_status="missing", config_kind="local_directory",
        ) == "config_unreadable"
        assert classify(
            -1, "sandbox read denied",
            sarif_validation_status="missing", config_kind="local_file",
        ) == "sandbox_read_denied"
        assert classify(
            0, "", sarif_validation_status="invalid", config_kind="local_file",
        ) == "sarif_invalid"

    def test_all_dispatched_packs_fail_without_sarif_paths(self):
        configs = [("category_auth", "p/auth"), ("semgrep_auth", "p/auth")]
        assert _scanner_mod._all_semgrep_packs_failed(
            configs,
            ["semgrep_auth", "category_auth"],
        )
        assert not _scanner_mod._all_semgrep_packs_failed(
            configs,
            ["category_auth"],
        )
    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    @patch.object(_scanner_mod, "_semgrep_executable_identity", return_value=_EXECUTABLE_IDENTITY)
    def test_operator_pack_name_is_redacted_and_findings_exit_is_accepted(
        self, _identity, _validate, tmp_path,
    ):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        unsafe_name = "extra_opaque-fixture-api-key=top-secret.yml"
        _write_pack(tmp_path, unsafe_name, 1)
        (tmp_path / "combined.sarif").write_text(
            '{"version":"2.1.0","runs":[]}', encoding="utf-8",
        )

        summary = _scanner_mod.write_semgrep_run_summary(
            out_dir=tmp_path,
            configs=[(unsafe_name, str(config))],
            aggregate_exit_code=0,
            sandbox_engagement_state="engaged",
        )

        pack = summary["packs"][0]
        assert pack["raw_exit_code"] == 1
        assert pack["failure_class"] is None
        assert summary["aggregate_exit_code"] == 0
        assert pack["pack_name"].startswith("custom_")
        rendered = json.dumps(summary)
        assert "top-secret" not in rendered
        assert "api-key" not in rendered
        assert "opaque-fixture" not in rendered

    @patch.object(_scanner_mod, "validate_sarif", return_value=True)
    def test_identity_probe_does_not_block_summary_when_isolation_is_invalid(
        self, _validate, tmp_path,
    ):
        config = tmp_path / "rules.yml"
        config.write_text("rules: []\n", encoding="utf-8")
        executable = tmp_path / "semgrep"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        _write_pack(tmp_path, "category_auth", 1)

        with patch.object(_scanner_mod.shutil, "which", return_value=str(executable)), patch.object(
            _scanner_mod.RaptorConfig,
            "get_safe_env",
            side_effect=RuntimeError("invalid isolated temp"),
        ):
            summary = _scanner_mod.write_semgrep_run_summary(
                out_dir=tmp_path,
                configs=[("category_auth", str(config))],
                aggregate_exit_code=0,
                sandbox_engagement_state="engaged",
            )

        assert summary["scanner"]["version"] is None
        assert (tmp_path / "semgrep-run-summary.json").is_file()
