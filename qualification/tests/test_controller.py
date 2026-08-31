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
    _direct_scan_passed,
    _pack_failure_evidence,
    validate_private_temp,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
def _clean_git_repo(path: Path, *, with_sarif_schema: bool = False) -> Path:
    path.mkdir()
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    if with_sarif_schema:
        schema_path = path / "engine/schemas/sarif-2.1.0.json"
        schema_path.parent.mkdir(parents=True)
        schema_path.write_text("{}\n", encoding="utf-8")
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


_SEMGREP_IDENTITY = {
    "basename": "semgrep",
    "executable": True,
    "version": "1.99.0",
    "sha256": "d" * 64,
    "path_kind": "system",
}


def _passing_candidate_runtime() -> dict:
    return {
        "passed": True,
        "failure_class": None,
        "probe_return_code": 0,
        "probe_timed_out": False,
        "probe_output_digests": {
            "stdout_sha256": "a" * 64,
            "stderr_sha256": "b" * 64,
        },
        "candidate_python": {
            "implementation": "cpython",
            "version": {"major": 3, "minor": 13, "patch": 9},
            "executable_basename": "python3.13",
            "path_kind": "system",
            "resolved_executable_sha256": "c" * 64,
        },
        "jsonschema": {
            "importable": True,
            "version": "4.26.0",
            "expected_version": "4.26.0",
            "version_matches": True,
        },
        "sarif_schema": {
            "relative_path": "engine/schemas/sarif-2.1.0.json",
            "present": True,
            "sha256": "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a",
            "expected_sha256": "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a",
            "hash_matches": True,
        },
        "semgrep": dict(_SEMGREP_IDENTITY),
    }


def _runtime_failure(failure_class: str) -> dict:
    runtime = _passing_candidate_runtime()
    runtime["passed"] = False
    runtime["failure_class"] = failure_class
    if failure_class == "candidate_runtime_dependency_missing":
        runtime["jsonschema"].update({
            "importable": False,
            "version": None,
            "version_matches": False,
        })
    elif failure_class == "candidate_runtime_dependency_version_mismatch":
        runtime["jsonschema"].update({
            "version": "4.25.1",
            "version_matches": False,
        })
    return runtime
def _isolated_environment(root: Path) -> dict[str, str]:
    root_text = str(root.resolve())
    return {
        "RAPTOR_REQUIRE_CREDENTIAL_ISOLATION": "1",
        "RAPTOR_PRIVATE_TMPDIR": root_text,
        "TMPDIR": root_text,
        "TMP": root_text,
        "TEMP": root_text,
    }


def test_private_temp_attestation_requires_exact_current_user_directory(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    environment = _isolated_environment(root)

    attestation = validate_private_temp(root.resolve(), environment)

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

    missing_alias = dict(environment)
    missing_alias.pop("TMP")
    assert validate_private_temp(root.resolve(), missing_alias).valid is False

    mismatched_alias = dict(environment)
    mismatched_alias["TEMP"] = str(tmp_path / "other")
    assert validate_private_temp(root.resolve(), mismatched_alias).valid is False

    isolation_not_required = dict(environment)
    isolation_not_required["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] = "0"
    isolation_attestation = validate_private_temp(root.resolve(), isolation_not_required)
    assert isolation_attestation.credential_isolation_required is False
    assert isolation_attestation.valid is False

    root.chmod(0o755)
    invalid = validate_private_temp(root.resolve(), environment)
    assert invalid.valid is False
    assert invalid.mode == "0755"


def test_no_provider_preflight_uses_real_dispatcher_and_cleans_everything(monkeypatch):
    controller = QualificationController(
        repo_root=REPO_ROOT,
        candidate_python=Path(sys.executable),
    )

    monkeypatch.setattr(
        controller,
        "_candidate_runtime_preflight",
        lambda environment: _passing_candidate_runtime(),
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


def test_no_provider_preflight_forwards_attested_safe_environment(
    monkeypatch,
):
    from core.llm.dispatcher import server as dispatcher_server
    from core.llm.dispatcher import spawn as dispatcher_spawn

    controller = QualificationController(
        repo_root=REPO_ROOT,
        candidate_python=Path(sys.executable),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-reach-child")
    original_safe_environment = controller._safe_worker_environment
    forwarded_environments: list[dict[str, str]] = []
    probed_environments: list[dict[str, str]] = []

    def record_candidate_runtime(environment):
        assert environment is forwarded_environments[-1]
        probed_environments.append(environment)
        return _passing_candidate_runtime()

    def record_safe_environment(private_root):
        environment = original_safe_environment(private_root)
        forwarded_environments.append(environment)
        return environment

    original_spawn_worker = dispatcher_spawn.spawn_worker

    def reject_provider_forwarding(dispatcher, cmd, *, label, env=None, **kwargs):
        assert env is forwarded_environments[-1]
        assert label == "qualification-no-provider"
        assert len(cmd) == 2
        assert Path(cmd[1]).name == "no_provider_child.py"
        assert env is not None
        assert not any(
            name.startswith(("ANTHROPIC_", "OPENAI_", "GEMINI_"))
            for name in env
        )
        return original_spawn_worker(dispatcher, cmd, label=label, env=env, **kwargs)

    def scanner_sink(*args, **kwargs):
        raise AssertionError("offline preflight must not invoke the scanner sink")
    def provider_http_sink(*args, **kwargs):
        raise AssertionError("offline preflight must not construct a provider HTTP client")

    monkeypatch.setattr(controller, "_safe_worker_environment", record_safe_environment)
    monkeypatch.setattr(controller, "_candidate_runtime_preflight", record_candidate_runtime)
    monkeypatch.setattr(dispatcher_spawn, "spawn_worker", reject_provider_forwarding)
    monkeypatch.setattr(controller, "_run", scanner_sink)
    monkeypatch.setattr(dispatcher_server.httpx, "Client", provider_http_sink)

    record = controller.run_no_provider_preflight()

    assert record["failure_class"] is None
    assert record["credential_isolation_required"] is True
    assert len(forwarded_environments) == 1
    assert probed_environments == forwarded_environments


@pytest.mark.parametrize("broken_alias", ("TMP", "TEMP"))
def test_no_provider_preflight_rejects_unattested_worker_environment(
    monkeypatch, broken_alias,
):
    from core.llm.dispatcher import spawn as dispatcher_spawn

    controller = QualificationController(
        repo_root=REPO_ROOT,
        candidate_python=Path(sys.executable),
    )

    def malformed_safe_environment(private_root):
        environment = _isolated_environment(private_root)
        if broken_alias == "TMP":
            environment.pop(broken_alias)
        else:
            environment[broken_alias] = str(private_root / "other")
        return environment

    def spawn_sink(*args, **kwargs):
        raise AssertionError("invalid worker environment must fail before spawn")

    def scanner_sink(*args, **kwargs):
        raise AssertionError("invalid worker environment must fail before scanner")

    monkeypatch.setattr(controller, "_safe_worker_environment", malformed_safe_environment)
    monkeypatch.setattr(dispatcher_spawn, "spawn_worker", spawn_sink)
    monkeypatch.setattr(controller, "_run", scanner_sink)

    record = controller.run_no_provider_preflight()

    assert record["failure_class"] == "qualification_private_temp_contract_invalid"
    assert record["dispatcher_started"] is False
    assert record["child_spawned"] is False
    assert record["scanner_started"] is False

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


def test_direct_controller_reuses_attested_environment_and_scanner_failure_class(
    tmp_path, monkeypatch,
):
    repo = _clean_git_repo(tmp_path / "repo", with_sarif_schema=True)
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    original_safe_environment = controller._safe_worker_environment
    worker_environments: list[dict[str, str]] = []
    probed_environments: list[dict[str, str]] = []

    def record_safe_environment(private_root):
        environment = original_safe_environment(private_root)
        worker_environments.append(environment)
        return environment

    def record_candidate_runtime(environment):
        assert environment is worker_environments[-1]
        probed_environments.append(environment)
        return _passing_candidate_runtime()

    def fake_run(argv, *, env, timeout=300):
        del timeout
        assert env is worker_environments[-1]
        assert env is probed_environments[-1]
        scan_out = Path(argv[argv.index("--out") + 1])
        scan_out.mkdir(parents=True, exist_ok=True)
        (scan_out / "semgrep-run-summary.json").write_text(
            json.dumps({
                "schema_version": 2,
                "scanner": _SEMGREP_IDENTITY,
                "packs_dispatched": 1,
                "packs_succeeded": 0,
                "packs_failed": 1,
                "all_semgrep_failed": True,
                "aggregate_exit_code": 4,
                "failure_class": "semgrep_missing",
                "packs": [{
                    "pack_name": "category_auth",
                    "config_kind": "local_file",
                    "config_sha256": "e" * 64,
                    "raw_exit_code": 127,
                    "sarif_exists": False,
                    "sarif_validation_status": "missing",
                    "failure_class": "semgrep_missing",
                    "bounded_stderr_tail": "semgrep: command not found",
                    "sandbox_denial_count": 0,
                    "proxy_event_count": 0,
                }],
                "combined_sarif_exists": False,
                "combined_sarif_validation_status": "missing",
                "sandbox_engagement": {"state": "engaged", "denial_count": 0},
            }),
            encoding="utf-8",
        )
        return ProcessCapture(
            return_code=4,
            timed_out=False,
            stdout_sha256="a" * 64,
            stderr_sha256="b" * 64,
        )

    monkeypatch.setattr(controller, "_safe_worker_environment", record_safe_environment)
    monkeypatch.setattr(controller, "_candidate_runtime_preflight", record_candidate_runtime)
    monkeypatch.setattr(controller, "_run", fake_run)

    record = controller.run_direct()

    assert probed_environments == worker_environments
    assert record["credential_isolation_required"] is True
    assert record["failure_class"] == "semgrep_missing"
    assert record["semgrep"]["runtime_identity_matches_preflight"] is True
def test_candidate_probe_uses_supplied_interpreter_and_exact_environment(
    tmp_path, monkeypatch,
):
    repo = _clean_git_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate-python"
    candidate.symlink_to(Path(sys.executable))
    controller = QualificationController(repo_root=repo, candidate_python=candidate)
    assert controller.candidate_python == candidate.absolute()
    assert controller.candidate_python != candidate.resolve()
    exact_environment = {"RAPTOR_REQUIRE_CREDENTIAL_ISOLATION": "1"}

    def fake_bounded_json(argv, *, env, timeout):
        assert argv[0] == str(candidate.absolute())
        assert argv[1] == "-c"
        assert env is exact_environment
        assert timeout == 20
        return ProcessCapture(0, False, "a" * 64, "b" * 64), {
            "schema_version": 1,
            "candidate_python": {
                "implementation": "pypy",
                "version": [3, 11, 12],
                "basename": "candidate-python",
                "path_kind": "governed_private",
                "sha256": "c" * 64,
            },
            "jsonschema": {"importable": True, "version": "4.26.0"},
            "sarif_schema": {
                "relative_path": "engine/schemas/sarif-2.1.0.json",
                "present": True,
                "sha256": "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a",
            },
            "semgrep": _SEMGREP_IDENTITY,
        }

    monkeypatch.setattr(controller, "_run_bounded_json", fake_bounded_json)

    runtime = controller._candidate_runtime_preflight(exact_environment)

    assert sys.implementation.name == "cpython"
    assert runtime["passed"] is True
    assert runtime["candidate_python"] == {
        "implementation": "pypy",
        "version": {"major": 3, "minor": 11, "patch": 12},
        "executable_basename": "candidate-python",
        "path_kind": "governed_private",
        "resolved_executable_sha256": "c" * 64,
    }


@pytest.mark.parametrize(
    ("jsonschema_payload", "expected_failure"),
    [
        ({"importable": False, "version": None}, "candidate_runtime_dependency_missing"),
        ({"importable": True, "version": "4.25.1"}, "candidate_runtime_dependency_version_mismatch"),
    ],
)
def test_candidate_probe_classifies_exact_dependency_state(
    tmp_path, monkeypatch, jsonschema_payload, expected_failure,
):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=tmp_path / "candidate")

    def fake_bounded_json(argv, *, env, timeout):
        del argv, env, timeout
        return ProcessCapture(0, False, "a" * 64, "b" * 64), {
            "schema_version": 1,
            "candidate_python": {
                "implementation": "cpython",
                "version": [3, 13, 9],
                "basename": "candidate",
                "path_kind": "governed_private",
                "sha256": "c" * 64,
            },
            "jsonschema": jsonschema_payload,
            "sarif_schema": {
                "relative_path": "engine/schemas/sarif-2.1.0.json",
                "present": True,
                "sha256": "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a",
            },
            "semgrep": _SEMGREP_IDENTITY,
        }

    monkeypatch.setattr(controller, "_run_bounded_json", fake_bounded_json)

    runtime = controller._candidate_runtime_preflight({})

    assert runtime["passed"] is False
    assert runtime["failure_class"] == expected_failure


@pytest.mark.parametrize(
    "failure_class",
    [
        "candidate_runtime_dependency_missing",
        "candidate_runtime_dependency_version_mismatch",
    ],
)
def test_candidate_dependency_failure_blocks_direct_dispatch_and_provider_sinks(
    tmp_path, monkeypatch, failure_class,
):
    import jsonschema
    from core.llm.dispatcher import server as dispatcher_server

    assert jsonschema is not None
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))

    def scanner_sink(*args, **kwargs):
        raise AssertionError("candidate dependency failure must block scanner dispatch")

    def provider_sink(*args, **kwargs):
        raise AssertionError("direct qualification must not construct a provider HTTP client")

    monkeypatch.setattr(
        controller,
        "_candidate_runtime_preflight",
        lambda environment: _runtime_failure(failure_class),
    )
    monkeypatch.setattr(controller, "_run", scanner_sink)
    monkeypatch.setattr(dispatcher_server.httpx, "Client", provider_sink)

    record = controller.run_direct()

    assert record["status"] == "failed"
    assert record["failure_stage"] == "candidate_runtime_preflight"
    assert record["failure_class"] == failure_class
    assert record["packs_dispatched"] == 0
    assert record["provider_turn_count"] == 0
    assert record["scanner_started"] is False
    assert record["private_temp_cleanup_complete"] is True


def test_missing_candidate_dependency_blocks_no_provider_dispatch_sinks(
    tmp_path, monkeypatch,
):
    from core.llm.dispatcher import server as dispatcher_server
    from core.llm.dispatcher import spawn as dispatcher_spawn

    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))

    def sink(*args, **kwargs):
        raise AssertionError("candidate runtime failure must stop before dispatch sinks")

    monkeypatch.setattr(
        controller,
        "_candidate_runtime_preflight",
        lambda environment: _runtime_failure("candidate_runtime_dependency_missing"),
    )
    monkeypatch.setattr(controller, "_run", sink)
    monkeypatch.setattr(dispatcher_spawn, "spawn_worker", sink)
    monkeypatch.setattr(dispatcher_server.httpx, "Client", sink)

    record = controller.run_no_provider_preflight()

    assert record["failure_stage"] == "candidate_runtime_preflight"
    assert record["failure_class"] == "candidate_runtime_dependency_missing"
    assert record["packs_dispatched"] == 0
    assert record["provider_turn_count"] == 0
    assert record["scanner_started"] is False
    assert record["dispatcher_started"] is False


def test_pack_diagnostics_survive_cleanup_with_full_redaction(
    tmp_path, monkeypatch,
):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    scan_paths: list[Path] = []

    def fake_run(argv, *, env, timeout=300):
        del env, timeout
        scan_out = Path(argv[argv.index("--out") + 1])
        scan_paths.append(scan_out)
        scan_out.mkdir(parents=True, exist_ok=True)
        unsafe_tail = (
            "api_key=alpha\n"
            "Authorization: Bearer beta\n"
            "Cookie: session=gamma\n"
            "/Users/alice/a /home/alice/b /private/tmp/c "
            "/var/folders/d /private/var/folders/e"
        )
        (scan_out / "semgrep-run-summary.json").write_text(json.dumps({
            "schema_version": 2,
            "scanner": _SEMGREP_IDENTITY,
            "packs_dispatched": 1,
            "packs_succeeded": 0,
            "packs_failed": 1,
            "all_semgrep_failed": True,
            "aggregate_exit_code": 4,
            "failure_class": "timeout",
            "packs": [{
                "pack_name": "category_auth",
                "config_kind": "local_file",
                "config_sha256": "e" * 64,
                "raw_exit_code": -1,
                "sarif_exists": False,
                "sarif_validation_status": "missing",
                "failure_class": "timeout",
                "bounded_stderr_tail": unsafe_tail,
                "sandbox_denial_count": 2,
                "proxy_event_count": 1,
            }],
            "combined_sarif_exists": False,
            "combined_sarif_validation_status": "missing",
            "sandbox_engagement": {"state": "engaged", "denial_count": 2},
        }), encoding="utf-8")
        return ProcessCapture(4, False, "a" * 64, "b" * 64)

    monkeypatch.setattr(
        controller, "_candidate_runtime_preflight",
        lambda environment: _passing_candidate_runtime(),
    )
    monkeypatch.setattr(controller, "_run", fake_run)

    record = controller.run_direct()

    assert scan_paths and not scan_paths[0].exists()
    diagnostic = record["semgrep"]["pack_diagnostics"][0]
    assert set(diagnostic) == {
        "pack_name", "config_kind", "config_sha256", "raw_exit_code",
        "sarif_exists", "sarif_validation_status", "failure_class",
        "bounded_stderr_tail", "sandbox_denial_count", "proxy_event_count",
    }
    assert diagnostic["raw_exit_code"] == -1
    assert diagnostic["sarif_validation_status"] == "missing"
    assert diagnostic["failure_class"] == "timeout"
    assert len(diagnostic["bounded_stderr_tail"]) <= 800
    rendered = json.dumps(record, sort_keys=True)
    for forbidden in (
        "alpha", "beta", "gamma", "/Users/", "/home/", "/private/tmp/",
        "/var/folders/", "/private/var/folders/",
    ):
        assert forbidden not in rendered
    assert record["failure_class"] == "timeout"
    assert record["semgrep"]["failure_class_counts"] == {"timeout": 1}
    assert record["private_temp_cleanup_complete"] is True


def test_uniform_mixed_and_unavailable_pack_failures_are_explicit():
    uniform = _pack_failure_evidence(
        [{"failure_class": "timeout"}, {"failure_class": "timeout"}],
        2,
        None,
    )
    mixed = _pack_failure_evidence(
        [{"failure_class": "timeout"}, {"failure_class": "semgrep_missing"}],
        2,
        None,
    )
    unavailable = _pack_failure_evidence(
        [{"failure_class": None}],
        1,
        None,
    )

    assert uniform == (
        "timeout", {"timeout": 2}, "uniform_pack_failure_class", 0,
    )
    assert mixed == (
        "mixed_pack_failures",
        {"semgrep_missing": 1, "timeout": 1},
        "mixed_pack_failure_classes",
        0,
    )
    assert unavailable == (
        "pack_failure_class_unavailable", {}, "no_per_pack_failure_class", 1,
    )


def test_direct_gate_requires_sixteen_fully_validated_packs_and_combined_sarif():
    diagnostics = [{
        "pack_name": f"pack_{index}",
        "config_kind": "local_file",
        "config_sha256": "e" * 64,
        "raw_exit_code": index % 2,
        "sarif_exists": True,
        "sarif_validation_status": "full_valid",
        "failure_class": None,
        "bounded_stderr_tail": "",
        "sandbox_denial_count": 0,
        "proxy_event_count": 0,
    } for index in range(16)]
    record = {
        "promotable": False,
        "provider_turn_count": 0,
        "process_exit_code": 0,
        "scanner_exit_code": 0,
        "scanner_started": True,
        "candidate_runtime": _passing_candidate_runtime(),
        "semgrep": {
            "summary_schema_version": 2,
            "runtime_identity_matches_preflight": True,
            "packs_dispatched": 16,
            "packs_succeeded": 16,
            "packs_failed": 0,
            "aggregate_exit_code": 0,
            "failure_class": None,
            "sandbox_engagement": "engaged",
            "combined_sarif_validation_status": "full_valid",
            "per_pack_full_validation": True,
            "pack_diagnostics": diagnostics,
        },
        "sarif_validation": {
            "combined_full_validation": True,
            "per_pack_full_validation": True,
            "jsonschema_version_matches": True,
            "schema_hash_matches": True,
        },
        "artifact_contracts": {
            name: {"validation": "structural_contract_passed"}
            for name in ("scan_manifest", "scan_metrics", "verification")
        },
    }

    assert _direct_scan_passed(record) is True

    fifteen = json.loads(json.dumps(record))
    fifteen["semgrep"]["pack_diagnostics"].pop()
    assert _direct_scan_passed(fifteen) is False

    unavailable = json.loads(json.dumps(record))
    unavailable["semgrep"]["pack_diagnostics"][0][
        "sarif_validation_status"
    ] = "full_validation_unavailable"
    assert _direct_scan_passed(unavailable) is False

    invalid_combined = json.loads(json.dumps(record))
    invalid_combined["semgrep"]["combined_sarif_validation_status"] = "invalid"
    assert _direct_scan_passed(invalid_combined) is False
def test_integrated_controller_refuses_second_live_attempt(tmp_path, monkeypatch):
    repo = _clean_git_repo(tmp_path / "repo")
    controller = QualificationController(repo_root=repo, candidate_python=Path(sys.executable))
    monkeypatch.setattr(controller, "run_no_provider_preflight", _passing_preflight)
    original_launcher_environment = controller._launcher_environment
    launcher_environments: list[dict[str, str]] = []
    calls: list[list[str]] = []
    forwarded_environments: list[dict[str, str]] = []

    def record_launcher_environment(private_root):
        environment = original_launcher_environment(private_root)
        launcher_environments.append(environment)
        return environment

    def fake_run(argv, *, env, timeout=300):
        del timeout
        calls.append(list(argv))
        forwarded_environments.append(env)
        return ProcessCapture(
            return_code=2,
            timed_out=False,
            stdout_sha256="a" * 64,
            stderr_sha256="b" * 64,
        )

    monkeypatch.setattr(controller, "_launcher_environment", record_launcher_environment)
    monkeypatch.setattr(controller, "_run", fake_run)

    first = controller.run_integrated()

    assert first["integrated_live_command_count"] == 1
    assert first["credential_isolation_required"] is True
    assert first["failure_class"] == "agentic_launcher_exit_2_unattributed"
    assert len(calls) == 1
    assert forwarded_environments == launcher_environments
    assert forwarded_environments[0]["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] == "1"
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
