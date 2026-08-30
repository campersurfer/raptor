"""No-provider contracts for the bounded qualification controller."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from qualification.controller import (
    ProcessCapture,
    QualificationController,
    QualificationControllerError,
    validate_private_temp,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_git_repo(path: Path) -> Path:
    path.mkdir()
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Qualification Test",
        "GIT_AUTHOR_EMAIL": "qualification-test@example.invalid",
        "GIT_COMMITTER_NAME": "Qualification Test",
        "GIT_COMMITTER_EMAIL": "qualification-test@example.invalid",
    }
    subprocess.run(["git", "init", "--quiet", str(path)], check=True, env=environment)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(path), "commit", "--quiet", "-m", "fixture"],
        check=True,
        env=environment,
    )
    return path


def _passing_preflight() -> dict:
    return {
        "credential_isolation_required": True,
        "private_temp_present": True,
        "temp_aliases_equal": True,
        "private_temp_absolute": True,
        "private_temp_canonical": True,
        "private_temp_non_symlink": True,
        "private_temp_current_user_owned": True,
        "private_temp_mode": "0700",
        "dispatcher_started": True,
        "socket_path_bytes": 42,
        "child_started": True,
        "token_fd_available": True,
        "dispatcher_socket_available": True,
        "same_uid_and_fd_token_gate": True,
        "provider_credentials_in_child_environment": False,
        "direct_fallback_occurred": False,
        "process_exit_code": 0,
        "failure_stage": None,
        "failure_class": None,
        "long_private_temp_exercised": True,
        "legacy_uds_path_lower_bound_bytes": 220,
        "raw_run_id_absent_from_socket_path": True,
        "full_run_id_preserved_in_audit": True,
        "dispatcher_cleanup_complete": True,
        "private_temp_cleanup_complete": True,
    }


def test_private_temp_attestation_requires_exact_current_user_directory(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)

    attestation = validate_private_temp(root.resolve())

    assert attestation.valid is True
    assert attestation.as_record() == {
        "credential_isolation_required": True,
        "private_temp_present": True,
        "temp_aliases_equal": True,
        "private_temp_absolute": True,
        "private_temp_canonical": True,
        "private_temp_non_symlink": True,
        "private_temp_current_user_owned": True,
        "private_temp_mode": "0700",
    }

    root.chmod(0o755)
    invalid = validate_private_temp(root.resolve())
    assert invalid.valid is False
    assert invalid.mode == "0755"


def test_no_provider_preflight_uses_real_dispatcher_and_cleans_everything():
    controller = QualificationController(
        repo_root=REPO_ROOT,
        candidate_python=Path(sys.executable),
    )

    record = controller.run_no_provider_preflight()

    assert record["failure_class"] is None
    assert record["private_temp_present"] is True
    assert record["temp_aliases_equal"] is True
    assert record["private_temp_absolute"] is True
    assert record["private_temp_canonical"] is True
    assert record["private_temp_non_symlink"] is True
    assert record["private_temp_current_user_owned"] is True
    assert record["private_temp_mode"] == "0700"
    assert record["long_private_temp_exercised"] is True
    assert record["legacy_uds_path_lower_bound_bytes"] > 100
    assert record["dispatcher_started"] is True
    assert record["socket_path_bytes"] <= 100
    assert record["child_started"] is True
    assert record["child_spawned"] is True
    assert set(record["output_digests"]) == {"stdout_sha256", "stderr_sha256"}
    assert all(len(digest) == 64 for digest in record["output_digests"].values())
    assert record["token_fd_available"] is True
    assert record["dispatcher_socket_available"] is True
    assert record["same_uid_and_fd_token_gate"] is True
    assert record["provider_credentials_in_child_environment"] is False
    assert record["provider_turn_count"] == 0
    assert record["direct_fallback_occurred"] is False
    assert record["process_exit_code"] == 0
    assert record["raw_run_id_absent_from_socket_path"] is True
    assert record["full_run_id_preserved_in_audit"] is True
    assert record["dispatcher_cleanup_complete"] is True
    assert record["private_temp_cleanup_complete"] is True
    rendered = json.dumps(record, sort_keys=True)
    assert "/private/" not in rendered
    assert "dispatcher-audit.jsonl" not in rendered


def test_safe_worker_environment_keeps_private_temp_and_removes_provider_keys(
    tmp_path, monkeypatch,
):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-provider-key")
    controller = QualificationController(repo_root=REPO_ROOT)

    environment = controller._safe_worker_environment(root.resolve())

    assert environment["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] == "1"
    assert {environment[name] for name in ("RAPTOR_PRIVATE_TMPDIR", "TMPDIR", "TMP", "TEMP")} == {
        str(root.resolve())
    }
    assert "GEMINI_API_KEY" not in environment


def test_canary_controller_retains_only_bounded_attestation_and_refuses_retry(
    tmp_path, monkeypatch,
):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    calls: list[list[str]] = []
    attestation = {
        "schema_version": 3,
        "status": "passed",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "provider_turns_started": 2,
        "provider_turns_completed": 2,
        "terminal_call_count": 1,
        "fixture_read": True,
        "source_verified": True,
        "sink_verified": True,
        "language_verified": True,
        "semantic_relation_verified": True,
        "worker_cleanup_verified": True,
        "untrusted_extra": "must-not-persist",
    }

    def fake_run_json(argv, *, env, timeout):
        del env, timeout
        calls.append(list(argv))
        return (
            ProcessCapture(
                return_code=0,
                timed_out=False,
                stdout_sha256="a" * 64,
                stderr_sha256="b" * 64,
            ),
            attestation,
        )

    monkeypatch.setattr(controller, "_run_bounded_json", fake_run_json)

    record = controller.run_canary()

    assert record["status"] == "passed"
    assert record["exactly_one_valid_submit_context_map"] is True
    assert "untrusted_extra" not in record["attestation"]
    assert calls[0][1:] == [
        "raptor.py", "semantic-canary", "--model", "gemini-2.5-flash", "--format", "json",
    ]
    with pytest.raises(QualificationControllerError, match="already reached a terminal result"):
        controller.run_canary()
    assert len(calls) == 1


def test_integrated_controller_refuses_second_live_attempt(tmp_path, monkeypatch):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    monkeypatch.setattr(controller, "run_no_provider_preflight", _passing_preflight)
    calls: list[list[str]] = []

    def fake_run(argv, *, env, timeout=300):
        del env, timeout
        calls.append(list(argv))
        return ProcessCapture(
            return_code=2,
            timed_out=False,
            stdout_sha256="a" * 64,
            stderr_sha256="b" * 64,
        )

    monkeypatch.setattr(controller, "_run", fake_run)

    first = controller.run_integrated()

    assert first["integrated_live_command_count"] == 1
    assert first["failure_class"] == "agentic_launcher_exit_2_unattributed"
    assert len(calls) == 1
    assert calls[0][1:] == [
        "raptor.py", "agentic", "--repo", calls[0][4], "--out", calls[0][6],
        "--sandbox", "strict", "--no-codeql", "--threat-model", "--validate",
        "--model", "gemini-2.5-flash", "--max-findings", "1", "--no-exploits",
        "--no-patches", "--phase-timeout", "180",
    ]
    with pytest.raises(QualificationControllerError, match="already reached a terminal result"):
        controller.run_integrated()
    assert len(calls) == 1


def test_integrated_exit_two_uses_typed_private_temp_artifact(tmp_path, monkeypatch):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    monkeypatch.setattr(controller, "run_no_provider_preflight", _passing_preflight)

    def fake_run(argv, *, env, timeout=300):
        del env, timeout
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "credential-isolation-startup.json").write_text(
            json.dumps({
                "dispatcher_started": True,
                "child_started": False,
                "provider_turn_count": 0,
                "scanner_started": False,
                "direct_fallback_occurred": False,
                "failure_stage": "credential_isolation_private_temp_contract",
                "failure_class": "credential_isolation_private_temp_contract_invalid",
            }),
            encoding="utf-8",
        )
        return ProcessCapture(
            return_code=2,
            timed_out=False,
            stdout_sha256="a" * 64,
            stderr_sha256="b" * 64,
        )

    monkeypatch.setattr(controller, "_run", fake_run)

    record = controller.run_integrated()

    assert record["integrated_live_command_count"] == 1
    assert record["child_started"] is False
    assert record["provider_turn_count"] == 0
    assert record["scanner_started"] is False
    assert record["direct_fallback_occurred"] is False
    assert record["failure_stage"] == "private_temp_validation"
    assert record["failure_class"] == "qualification_private_temp_contract_invalid"
