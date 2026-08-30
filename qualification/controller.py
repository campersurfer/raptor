"""Bounded controller for one Raptor qualification attempt.

The controller owns only disposable fixtures, output, private temporary roots,
and redacted evidence. It never persists provider credentials, prompts,
responses, absolute paths, dispatcher paths, or capability-token material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_PRIVATE_TEMP_ENV = (
    "RAPTOR_PRIVATE_TMPDIR",
    "TMPDIR",
    "TMP",
    "TEMP",
)
_PROVIDER_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "GOOGLE_",
    "MISTRAL_",
    "GROQ_",
    "TOGETHER_",
    "OPENROUTER_",
    "FIREWORKS_",
    "DEEPINFRA_",
    "PERPLEXITY_",
    "REPLICATE_",
    "COHERE_",
    "AZURE_OPENAI_",
    "AWS_",
)
_SAFE_ARTIFACTS = (
    "scan/scan-manifest.json",
    "scan/scan_metrics.json",
    "scan/semgrep-run-summary.json",
    "scan/verification.json",
    "scan/combined.sarif",
    "understand-prepass-summary.json",
    "raptor_agentic_report.json",
    "credential-isolation-startup.json",
)
_CANARY_ATTESTATION_FIELDS = (
    "schema_version",
    "status",
    "provider",
    "model",
    "sdk_version",
    "request_schema_sha256",
    "provider_turns_started",
    "provider_turns_completed",
    "terminal_call_count",
    "fixture_read",
    "source_verified",
    "sink_verified",
    "language_verified",
    "semantic_relation_verified",
    "failure_class",
    "failure_stage",
    "worker_cleanup_verified",
)


class QualificationControllerError(RuntimeError):
    """A bounded controller invariant failed before a live command."""


@dataclass(frozen=True)
class PrivateTempAttestation:
    """Boolean-only private-temp contract result."""

    present: bool
    aliases_equal: bool
    absolute: bool
    canonical: bool
    non_symlink: bool
    current_user_owned: bool
    mode: str

    @property
    def valid(self) -> bool:
        return (
            self.present
            and self.aliases_equal
            and self.absolute
            and self.canonical
            and self.non_symlink
            and self.current_user_owned
            and self.mode == "0700"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "credential_isolation_required": True,
            "private_temp_present": self.present,
            "temp_aliases_equal": self.aliases_equal,
            "private_temp_absolute": self.absolute,
            "private_temp_canonical": self.canonical,
            "private_temp_non_symlink": self.non_symlink,
            "private_temp_current_user_owned": self.current_user_owned,
            "private_temp_mode": self.mode,
        }


@dataclass(frozen=True)
class ProcessCapture:
    """Process exit and raw-output hashes retained after ephemeral cleanup."""

    return_code: int | None
    timed_out: bool
    stdout_sha256: str
    stderr_sha256: str


@dataclass(frozen=True)
class ExecutionIdentity:
    """Exact clean commit/tree that a record may bind."""

    commit: str
    tree: str


class QualificationController:
    """Run bounded qualification paths without retaining sensitive material."""

    def __init__(
        self,
        *,
        repo_root: Path = _REPO_ROOT,
        candidate_python: Path | None = None,
        runner: Any = subprocess.run,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.candidate_python = (candidate_python or Path(sys.executable)).resolve()
        self._runner = runner
        self._integrated_terminal = False
        self._canary_terminal = False

    def execution_identity(self) -> ExecutionIdentity:
        """Return the clean checked-out commit and tree for new evidence."""
        dirty = self._git("status", "--porcelain")
        if dirty.stdout.strip():
            raise QualificationControllerError("qualification execution tree is dirty")
        return ExecutionIdentity(
            commit=self._git("rev-parse", "HEAD").stdout.strip(),
            tree=self._git("rev-parse", "HEAD^{tree}").stdout.strip(),
        )

    def run_no_provider_preflight(self) -> dict[str, Any]:
        """Exercise a real local dispatcher without invoking a provider."""
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        fixture = self._new_fixture(workspace)
        output_root = workspace / "out"
        output_root.mkdir(mode=0o700)
        attestation = validate_private_temp(private_root)
        record: dict[str, Any] = {
            "schema_version": 1,
            "record_kind": "qualification_private_temp_preflight",
            "immutable": True,
            "promotable": False,
            "fixture": {
                "kind": "fresh_inert_cpp_git",
                "source_tree_sha256": self._fixture_tree_hash(fixture),
            },
            **attestation.as_record(),
            "dispatcher_started": False,
            "socket_path_bytes": None,
            "child_spawned": False,
            "child_started": False,
            "provider_turn_count": 0,
            "scanner_started": False,
            "direct_fallback_occurred": False,
            "tcp_fallback_occurred": False,
            "process_exit_code": None,
            "failure_stage": None,
            "failure_class": None,
            "long_private_temp_exercised": True,
            "legacy_uds_path_lower_bound_bytes": None,
            "raw_run_id_absent_from_socket_path": False,
            "full_run_id_preserved_in_audit": False,
            "retention": {
                "fixture_and_raw_run_output": "ephemeral",
                "raw_output_persisted": False,
            },
        }
        dispatcher = None
        process = None
        capture_root: Path | None = None
        stdout_path: Path | None = None
        stderr_path: Path | None = None
        report_path = output_root / "no-provider-child.json"
        try:
            if not attestation.valid:
                record.update({
                    "process_exit_code": 2,
                    "failure_stage": "private_temp_validation",
                    "failure_class": "qualification_private_temp_contract_invalid",
                })
                return record

            from core.llm.dispatcher.auth import CredentialStore
            from core.llm.dispatcher.server import LLMDispatcher
            from core.llm.dispatcher.spawn import spawn_worker

            run_id = f"qualification-{uuid.uuid4().hex}-{uuid.uuid4().hex}"
            record["legacy_uds_path_lower_bound_bytes"] = len(
                os.fsencode(str(private_root / f"raptor-llm-{run_id}.sock"))
            )
            credentials = CredentialStore.__new__(CredentialStore)
            credentials._keys = {
                "anthropic": None,
                "openai": None,
                "gemini": None,
            }
            audit_path = output_root / "dispatcher-audit.jsonl"
            dispatcher = LLMDispatcher(
                run_id=run_id,
                audit_path=audit_path,
                token_ttl_s=30,
                token_budget=1,
                creds=credentials,
            )
            socket_path = dispatcher.socket_path
            if socket_path is None:
                raise QualificationControllerError("dispatcher omitted socket path")
            record["dispatcher_started"] = True
            record["socket_path_bytes"] = len(os.fsencode(str(socket_path)))
            record["raw_run_id_absent_from_socket_path"] = run_id not in str(socket_path)

            child = workspace / "no_provider_child.py"
            _write_no_provider_child(child, report_path)
            capture_root = Path(tempfile.mkdtemp(prefix="raptor-qualification-output-"))
            stdout_path = capture_root / "stdout"
            stderr_path = capture_root / "stderr"
            with _private_output_file(stdout_path) as stdout, _private_output_file(stderr_path) as stderr:
                process = spawn_worker(
                    dispatcher,
                    [str(self.candidate_python), str(child)],
                    label="qualification-no-provider",
                    env=self._safe_worker_environment(private_root),
                    stdout=stdout,
                    stderr=stderr,
                )
                record["child_spawned"] = True
                record["process_exit_code"] = process.wait(timeout=30)
            record["output_digests"] = {
                "stdout_sha256": _sha256_file(stdout_path),
                "stderr_sha256": _sha256_file(stderr_path),
            }
            child_report = _load_json(report_path)
            record["child_started"] = child_report.get("child_started") is True
            record["token_fd_available"] = child_report.get("token_fd_available") is True
            record["dispatcher_socket_available"] = child_report.get("dispatcher_socket_available") is True
            record["same_uid_and_fd_token_gate"] = child_report.get("fd_token_gate") is True
            record["provider_credentials_in_child_environment"] = (
                child_report.get("provider_credentials_in_child_environment") is True
            )
            if record["process_exit_code"] != 0 or not all(
                record[name] is True
                for name in (
                    "child_started",
                    "token_fd_available",
                    "dispatcher_socket_available",
                    "same_uid_and_fd_token_gate",
                )
            ) or record["provider_credentials_in_child_environment"]:
                record.update({
                    "failure_stage": "credential_isolation_no_provider_preflight",
                    "failure_class": "credential_isolation_child_spawn_failed",
                })
            else:
                record["full_run_id_preserved_in_audit"] = (
                    audit_path.is_file() and run_id in audit_path.read_text(encoding="utf-8")
                )
                if not record["full_run_id_preserved_in_audit"]:
                    record.update({
                        "failure_stage": "credential_isolation_no_provider_preflight",
                        "failure_class": "credential_isolation_audit_run_id_missing",
                    })
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.wait(timeout=5)
            record.update({
                "process_exit_code": 2,
                "failure_stage": "credential_isolation_no_provider_preflight",
                "failure_class": "credential_isolation_child_timeout",
            })
        except Exception:
            if dispatcher is None:
                failure_class = "credential_isolation_dispatcher_startup_failed"
            elif process is None:
                failure_class = "credential_isolation_child_spawn_failed"
            else:
                failure_class = "credential_isolation_child_execution_failed"
            record.update({
                "process_exit_code": 2 if record["process_exit_code"] is None else record["process_exit_code"],
                "failure_stage": "credential_isolation_no_provider_preflight",
                "failure_class": failure_class,
            })
        finally:
            if capture_root is not None:
                if (
                    "output_digests" not in record
                    and stdout_path is not None
                    and stderr_path is not None
                    and stdout_path.is_file()
                    and stderr_path.is_file()
                ):
                    record["output_digests"] = {
                        "stdout_sha256": _sha256_file(stdout_path),
                        "stderr_sha256": _sha256_file(stderr_path),
                    }
                shutil.rmtree(capture_root, ignore_errors=True)
            if dispatcher is not None:
                socket_dir = dispatcher._sock_dir
                dispatcher.shutdown()
                record["dispatcher_cleanup_complete"] = (
                    socket_dir is None or not socket_dir.exists()
                )
            else:
                record["dispatcher_cleanup_complete"] = True
            self._cleanup_workspace(workspace, record)
        return record

    def run_canary(self) -> dict[str, Any]:
        """Run at most one standalone Gemini semantic canary per instance."""
        if self._canary_terminal:
            raise QualificationControllerError("semantic canary already reached a terminal result")
        self._canary_terminal = True
        identity = self.execution_identity()
        argv = [
            str(self.candidate_python),
            "raptor.py",
            "semantic-canary",
            "--model",
            "gemini-2.5-flash",
            "--format",
            "json",
        ]
        capture, payload = self._run_bounded_json(
            argv,
            env=self._trusted_environment(),
            timeout=300,
        )
        attestation = {
            name: payload.get(name)
            for name in _CANARY_ATTESTATION_FIELDS
            if name in payload
        }
        passed = (
            capture.return_code == 0
            and not capture.timed_out
            and attestation.get("status") == "passed"
            and attestation.get("provider") == "gemini"
            and attestation.get("model") == "gemini-2.5-flash"
            and type(attestation.get("provider_turns_completed")) is int
            and attestation["provider_turns_completed"] >= 1
            and attestation.get("terminal_call_count") == 1
            and attestation.get("fixture_read") is True
            and attestation.get("source_verified") is True
            and attestation.get("sink_verified") is True
            and attestation.get("language_verified") is True
            and attestation.get("semantic_relation_verified") is True
            and attestation.get("worker_cleanup_verified") is True
        )
        return {
            "schema_version": 1,
            "record_kind": "semantic_canary_attestation",
            "immutable": True,
            "promotable": False,
            "status": "passed" if passed else "failed",
            "execution": {"commit": identity.commit, "tree": identity.tree},
            "command": {
                "argv": _redact_argv(argv),
                "argv_sha256": _sha256_json(argv),
            },
            "return_code": capture.return_code,
            "timed_out": capture.timed_out,
            "stdout_sha256": capture.stdout_sha256,
            "stderr_sha256": capture.stderr_sha256,
            "raw_output_persisted": False,
            "exactly_one_valid_submit_context_map": passed,
            "attestation": attestation,
            "failure_stage": None if passed else attestation.get("failure_stage") or "semantic_canary",
            "failure_class": None if passed else attestation.get("failure_class") or "semantic_canary_contract_failed",
        }

    def run_direct(self) -> dict[str, Any]:
        """Run one direct strict-Semgrep qualification without a provider."""
        identity = self.execution_identity()
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        fixture = self._new_fixture(workspace)
        output_root = workspace / "out"
        output_root.mkdir(mode=0o700)
        scan_out = output_root / "scan"
        attestation = validate_private_temp(private_root)
        argv = [
            str(self.candidate_python),
            "packages/static-analysis/scanner.py",
            "--repo", str(fixture),
            "--out", str(scan_out),
            "--sandbox", "strict",
            "--no-codeql",
            "--policy-groups", "all",
        ]
        record = self._base_record(
            kind="minimal_cpp_direct_semgrep_qualification",
            identity=identity,
            fixture=fixture,
            argv=argv,
            attestation=attestation,
        )
        try:
            if not attestation.valid:
                return self._set_contract_failure(record)
            capture = self._run(argv, env=self._safe_worker_environment(private_root))
            record.update({
                "process_exit_code": capture.return_code,
                "scanner_exit_code": capture.return_code,
                "output_digests": _capture_hashes(capture),
            })
            self._add_scan_evidence(record, scan_out)
            if capture.timed_out:
                record.update({
                    "status": "failed",
                    "failure_stage": "direct_semgrep_qualification",
                    "failure_class": "qualification_controller_timeout",
                })
            elif not _direct_scan_passed(record):
                record.update({
                    "status": "failed",
                    "failure_stage": "direct_semgrep_qualification",
                    "failure_class": "direct_semgrep_contract_failed",
                })
            else:
                record["status"] = "qualified"
        finally:
            self._cleanup_workspace(workspace, record)
        return record

    def run_integrated(self) -> dict[str, Any]:
        """Run at most one provider-backed agentic qualification per instance."""
        if self._integrated_terminal:
            raise QualificationControllerError("integrated qualification already reached a terminal result")
        self._integrated_terminal = True
        preflight = self.run_no_provider_preflight()
        identity = self.execution_identity()
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        fixture = self._new_fixture(workspace)
        output_root = workspace / "out"
        output_root.mkdir(mode=0o700)
        attestation = validate_private_temp(private_root)
        argv = [
            str(self.candidate_python),
            "raptor.py",
            "agentic",
            "--repo", str(fixture),
            "--out", str(output_root),
            "--sandbox", "strict",
            "--no-codeql",
            "--threat-model",
            "--validate",
            "--model", "gemini-2.5-flash",
            "--max-findings", "1",
            "--no-exploits",
            "--no-patches",
            "--phase-timeout", "180",
        ]
        record = self._base_record(
            kind="minimal_cpp_integrated_agentic_qualification",
            identity=identity,
            fixture=fixture,
            argv=argv,
            attestation=attestation,
        )
        record["integrated_live_command_count"] = 0
        record["no_provider_preflight"] = _preflight_summary(preflight)
        try:
            if not attestation.valid:
                return self._set_contract_failure(record)
            if preflight.get("failure_class") is not None:
                record.update({
                    "status": "failed",
                    "failure_stage": "credential_isolation_no_provider_preflight",
                    "failure_class": "credential_isolation_preflight_failed",
                })
                return record
            record["integrated_live_command_count"] = 1
            capture = self._run(argv, env=self._launcher_environment(private_root), timeout=420)
            record.update({
                "process_exit_code": capture.return_code,
                "agentic_exit_code": capture.return_code,
                "output_digests": _capture_hashes(capture),
            })
            self._add_integrated_evidence(record, output_root)
            if capture.timed_out:
                record.update({
                    "status": "failed",
                    "failure_stage": "integrated_agentic_qualification",
                    "failure_class": "qualification_controller_timeout",
                })
            elif _integrated_passed(record):
                record["status"] = "qualified"
            else:
                _classify_integrated_failure(record, output_root)
        finally:
            self._cleanup_workspace(workspace, record)
        return record

    def _base_record(
        self,
        *,
        kind: str,
        identity: ExecutionIdentity,
        fixture: Path,
        argv: Sequence[str],
        attestation: PrivateTempAttestation,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_kind": kind,
            "immutable": True,
            "promotable": False,
            "status": "failed",
            "execution": {"commit": identity.commit, "tree": identity.tree},
            "fixture": {
                "kind": "fresh_inert_cpp_git",
                "source_tree_sha256": self._fixture_tree_hash(fixture),
            },
            "command": {
                "argv": _redact_argv(argv),
                "argv_sha256": _sha256_json(list(argv)),
            },
            **attestation.as_record(),
            "dispatcher_started": False,
            "socket_path_bytes": None,
            "child_started": False,
            "provider_turn_count": 0,
            "scanner_started": False,
            "direct_fallback_occurred": False,
            "tcp_fallback_occurred": False,
            "failure_stage": None,
            "failure_class": None,
            "retention": {
                "fixture_and_raw_run_output": "ephemeral",
                "raw_output_persisted": False,
            },
        }

    def _set_contract_failure(self, record: dict[str, Any]) -> dict[str, Any]:
        record.update({
            "process_exit_code": 2,
            "failure_stage": "private_temp_validation",
            "failure_class": "qualification_private_temp_contract_invalid",
        })
        return record

    def _workspace(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="raptor-qualification-"))

    def _new_private_root(self, workspace: Path) -> Path:
        root = workspace / ("private-temp-" + "x" * 120)
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        return root.resolve()

    def _new_fixture(self, workspace: Path) -> Path:
        fixture = workspace / "fixture"
        fixture.mkdir(mode=0o700)
        (fixture / "main.cpp").write_text(
            "int main() { return 0; }\n", encoding="utf-8"
        )
        (fixture / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(qualification LANGUAGES CXX)\n"
            "add_executable(qualification main.cpp)\n",
            encoding="utf-8",
        )
        fixture_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Raptor Qualification",
            "GIT_AUTHOR_EMAIL": "qualification@example.invalid",
            "GIT_COMMITTER_NAME": "Raptor Qualification",
            "GIT_COMMITTER_EMAIL": "qualification@example.invalid",
        }
        self._runner(["git", "init", "--quiet", str(fixture)], check=True, env=fixture_env)
        self._runner(["git", "-C", str(fixture), "add", "."], check=True, env=fixture_env)
        self._runner(
            ["git", "-C", str(fixture), "commit", "--quiet", "-m", "qualification fixture"],
            check=True,
            env=fixture_env,
        )
        return fixture

    def _fixture_tree_hash(self, fixture: Path) -> str:
        return self._runner(
            ["git", "-C", str(fixture), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _git(self, *args: str) -> Any:
        return self._runner(
            ["git", *args],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def _safe_worker_environment(self, private_root: Path) -> dict[str, str]:
        from core.config import RaptorConfig

        original = {name: os.environ.get(name) for name in _PRIVATE_TEMP_ENV}
        original_required = os.environ.get("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION")
        root_text = str(private_root)
        try:
            os.environ["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] = "1"
            for name in _PRIVATE_TEMP_ENV:
                os.environ[name] = root_text
            env = RaptorConfig.get_safe_env()
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            if original_required is None:
                os.environ.pop("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", None)
            else:
                os.environ["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] = original_required
        if any(_is_provider_environment_name(name) for name in env):
            raise QualificationControllerError("safe worker environment retained a provider credential name")
        return env

    def _trusted_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("RAPTOR_LLM_SOCKET", None)
        env.pop("RAPTOR_LLM_TOKEN_FD", None)
        return env

    def _launcher_environment(self, private_root: Path) -> dict[str, str]:
        """Preserve the trusted launcher's credentials, never its worker's."""
        env = self._trusted_environment()
        root_text = str(private_root)
        env["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] = "1"
        for name in _PRIVATE_TEMP_ENV:
            env[name] = root_text
        return env

    def _run(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str],
        timeout: float = 300,
    ) -> ProcessCapture:
        capture_root = Path(tempfile.mkdtemp(prefix="raptor-qualification-output-"))
        stdout_path = capture_root / "stdout"
        stderr_path = capture_root / "stderr"
        return_code: int | None = None
        timed_out = False
        try:
            with _private_output_file(stdout_path) as stdout, _private_output_file(stderr_path) as stderr:
                try:
                    completed = self._runner(
                        list(argv),
                        cwd=self.repo_root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=timeout,
                    )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            return ProcessCapture(
                return_code=return_code,
                timed_out=timed_out,
                stdout_sha256=_sha256_file(stdout_path),
                stderr_sha256=_sha256_file(stderr_path),
            )
        finally:
            shutil.rmtree(capture_root, ignore_errors=True)

    def _run_bounded_json(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[ProcessCapture, dict[str, Any]]:
        capture_root = Path(tempfile.mkdtemp(prefix="raptor-qualification-output-"))
        stdout_path = capture_root / "stdout"
        stderr_path = capture_root / "stderr"
        return_code: int | None = None
        timed_out = False
        payload: dict[str, Any] = {}
        try:
            with _private_output_file(stdout_path) as stdout, _private_output_file(stderr_path) as stderr:
                try:
                    completed = self._runner(
                        list(argv),
                        cwd=self.repo_root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=timeout,
                    )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            if stdout_path.stat().st_size <= 65536:
                try:
                    decoded = json.loads(stdout_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    decoded = None
                if isinstance(decoded, dict):
                    payload = decoded
            capture = ProcessCapture(
                return_code=return_code,
                timed_out=timed_out,
                stdout_sha256=_sha256_file(stdout_path),
                stderr_sha256=_sha256_file(stderr_path),
            )
            return capture, payload
        finally:
            shutil.rmtree(capture_root, ignore_errors=True)

    def _add_scan_evidence(self, record: dict[str, Any], scan_out: Path) -> None:
        summary = _load_json_if_file(scan_out / "semgrep-run-summary.json")
        scanner_started = isinstance(summary, dict)
        record["scanner_started"] = scanner_started
        record["consumed_artifact_hashes"] = _artifact_hashes(scan_out, prefix="scan")
        if not isinstance(summary, dict):
            return
        sandbox = summary.get("sandbox_engagement")
        sandbox_state = sandbox.get("state") if isinstance(sandbox, dict) else None
        pack_validations = {
            str(pack.get("pack_name")): pack.get("sarif_valid") is True
            for pack in summary.get("packs", [])
            if isinstance(pack, dict) and isinstance(pack.get("pack_name"), str)
        }
        combined_valid = _validate_sarif(scan_out / "combined.sarif")
        record["semgrep"] = {
            "packs_dispatched": summary.get("packs_dispatched"),
            "packs_succeeded": summary.get("packs_succeeded"),
            "packs_failed": summary.get("packs_failed"),
            "aggregate_exit_code": summary.get("aggregate_exit_code"),
            "combined_sarif_valid": combined_valid,
            "per_pack_sarif_valid": pack_validations,
            "per_pack_full_validation": bool(pack_validations) and all(pack_validations.values()),
            "sandbox_engagement": sandbox_state,
        }
        record["sarif_validation"] = {
            "canonical_schema": "engine/schemas/sarif-2.1.0.json",
            "schema_sha256": _sha256_file(self.repo_root / "engine/schemas/sarif-2.1.0.json"),
            "per_pack_full_validation": bool(pack_validations) and all(pack_validations.values()),
            "combined_full_validation": combined_valid,
            "jsonschema_distribution": "jsonschema",
            "jsonschema_version": _distribution_version("jsonschema"),
        }
        record["artifact_contracts"] = _artifact_contracts(scan_out)

    def _add_integrated_evidence(self, record: dict[str, Any], output_root: Path) -> None:
        startup = _load_json_if_file(output_root / "credential-isolation-startup.json")
        if isinstance(startup, dict):
            for name in (
                "dispatcher_started",
                "child_started",
                "provider_turn_count",
                "scanner_started",
                "direct_fallback_occurred",
            ):
                if isinstance(startup.get(name), bool) or isinstance(startup.get(name), int):
                    record[name] = startup[name]
            record["startup_failure"] = {
                "failure_stage": startup.get("failure_stage"),
                "failure_class": startup.get("failure_class"),
            }
        prepass = _load_json_if_file(output_root / "understand-prepass-summary.json")
        if isinstance(prepass, dict):
            record["child_started"] = True
            record["dispatcher_started"] = True
            record["provider_turn_count"] = (
                prepass.get("terminal_call_count")
                if isinstance(prepass.get("terminal_call_count"), int)
                else 0
            )
            record["understand_prepass"] = {
                name: prepass.get(name)
                for name in (
                    "provider",
                    "model",
                    "ran",
                    "terminal_call_count",
                    "context_map_valid",
                    "semantic_complete",
                )
            }
        self._add_scan_evidence(record, output_root / "scan")
        record["consumed_artifact_hashes"] = _artifact_hashes(output_root)
        report = _load_json_if_file(output_root / "raptor_agentic_report.json")
        record["agentic_report_relative_paths_only"] = (
            isinstance(report, dict) and not _contains_absolute_path(report)
        )
        if record.get("scanner_started") is True:
            record["child_started"] = True
            record["dispatcher_started"] = True

    def _cleanup_workspace(self, workspace: Path, record: dict[str, Any]) -> None:
        shutil.rmtree(workspace, ignore_errors=True)
        record["private_temp_cleanup_complete"] = not workspace.exists()


def validate_private_temp(private_root: Path) -> PrivateTempAttestation:
    """Validate exactly the same bounded contract RaptorConfig enforces."""
    root_text = str(private_root)
    values = {name: root_text for name in _PRIVATE_TEMP_ENV}
    present = private_root.exists()
    aliases_equal = all(value == root_text for value in values.values())
    absolute = private_root.is_absolute()
    canonical = False
    non_symlink = False
    current_user_owned = False
    mode = "unknown"
    try:
        observed = os.lstat(private_root)
        resolved = private_root.resolve(strict=True)
        canonical = str(resolved) == root_text
        non_symlink = not stat.S_ISLNK(observed.st_mode) and stat.S_ISDIR(observed.st_mode)
        current_user_owned = observed.st_uid == getattr(os, "geteuid", os.getuid)()
        mode = f"{stat.S_IMODE(observed.st_mode):04o}"
    except OSError:
        pass
    return PrivateTempAttestation(
        present=present,
        aliases_equal=aliases_equal,
        absolute=absolute,
        canonical=canonical,
        non_symlink=non_symlink,
        current_user_owned=current_user_owned,
        mode=mode,
    )


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write bounded evidence atomically without granting secret-bearing modes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_no_provider_child(path: Path, report: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "import socket\n"
        "from pathlib import Path\n"
        f"report = Path({str(report)!r})\n"
        "socket_path = os.environ.get('RAPTOR_LLM_SOCKET', '')\n"
        "token_fd = os.environ.get('RAPTOR_LLM_TOKEN_FD')\n"
        "provider_names = ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY')\n"
        "state = {\n"
        "  'child_started': True,\n"
        "  'dispatcher_socket_available': bool(socket_path) and Path(socket_path).exists(),\n"
        "  'token_fd_available': False,\n"
        "  'fd_token_gate': False,\n"
        "  'provider_credentials_in_child_environment': any(bool(os.environ.get(name)) for name in provider_names),\n"
        "}\n"
        "try:\n"
        "  token = os.read(int(token_fd), 512) if token_fd else b''\n"
        "  state['token_fd_available'] = bool(token)\n"
        "  if token and socket_path:\n"
        "    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    connection.connect(socket_path)\n"
        "    request = (b'POST /openai/v1/chat/completions HTTP/1.1\\r\\n'\n"
        "      b'Host: localhost\\r\\n'\n"
        "      b'X-Raptor-Token: ' + token + b'\\r\\n'\n"
        "      b'Content-Length: 2\\r\\n\\r\\n{}')\n"
        "    connection.sendall(request)\n"
        "    response = connection.recv(4096)\n"
        "    connection.close()\n"
        "    state['fd_token_gate'] = b' 503 ' in response and b'provider not configured: openai' in response\n"
        "except Exception:\n"
        "  pass\n"
        "report.write_text(json.dumps(state), encoding='utf-8')\n",
        encoding="utf-8",
    )


def _redact_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for index, value in enumerate(argv):
        if index == 0:
            redacted.append("candidate-python")
            continue
        if redact_next:
            redacted.append("<fresh_inert_cpp_git_fixture>" if argv[index - 1] == "--repo" else "<fresh_empty_output>")
            redact_next = False
            continue
        redacted.append(value)
        if value in {"--repo", "--out"}:
            redact_next = True
    return redacted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _private_output_file(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    return os.fdopen(descriptor, "wb")


def _capture_hashes(capture: ProcessCapture) -> dict[str, str]:
    return {
        "stdout_sha256": capture.stdout_sha256,
        "stderr_sha256": capture.stderr_sha256,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QualificationControllerError("expected bounded object artifact")
    return payload


def _load_json_if_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError, QualificationControllerError):
        return None


def _artifact_hashes(root: Path, *, prefix: str = "") -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SAFE_ARTIFACTS:
        relative_path = Path(relative)
        if prefix and relative_path.parts[:1] != (prefix,):
            continue
        path = root / relative_path if not prefix else root / relative_path.relative_to(prefix)
        if path.is_file():
            key = str(relative_path if prefix else relative_path)
            hashes[key] = _sha256_file(path)
    return hashes


def _validate_sarif(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from core.sarif.parser import validate_sarif

        return validate_sarif(path) is True
    except Exception:
        return False


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _artifact_contracts(scan_out: Path) -> dict[str, dict[str, str]]:
    manifest = _load_json_if_file(scan_out / "scan-manifest.json")
    metrics = _load_json_if_file(scan_out / "scan_metrics.json")
    verification = _load_json_if_file(scan_out / "verification.json")
    return {
        "scan_manifest": {
            "status": "not_declared",
            "validation": "structural_contract_passed"
            if isinstance(manifest, dict) and {"agent", "version", "repo_path", "policy_groups"}.issubset(manifest)
            else "structural_contract_failed",
        },
        "scan_metrics": {
            "status": "not_declared",
            "validation": "structural_contract_passed"
            if isinstance(metrics, dict)
            else "structural_contract_failed",
        },
        "verification": {
            "status": "declared",
            "validation": "structural_contract_passed"
            if isinstance(verification, dict)
            and verification.get("schema_version") == 1
            and isinstance(verification.get("combined_sarif"), dict)
            and isinstance(verification.get("packs"), list)
            else "structural_contract_failed",
        },
    }


def _is_provider_environment_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _PROVIDER_ENV_PREFIXES)


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    return False


def _direct_scan_passed(record: dict[str, Any]) -> bool:
    semgrep = record.get("semgrep")
    validation = record.get("sarif_validation")
    contracts = record.get("artifact_contracts")
    return (
        record.get("process_exit_code") == 0
        and record.get("scanner_started") is True
        and isinstance(semgrep, dict)
        and semgrep.get("packs_dispatched") == 16
        and semgrep.get("packs_succeeded") == 16
        and semgrep.get("packs_failed") == 0
        and semgrep.get("aggregate_exit_code") == 0
        and semgrep.get("sandbox_engagement") == "engaged"
        and semgrep.get("combined_sarif_valid") is True
        and semgrep.get("per_pack_full_validation") is True
        and isinstance(validation, dict)
        and validation.get("combined_full_validation") is True
        and validation.get("per_pack_full_validation") is True
        and isinstance(contracts, dict)
        and all(contract.get("validation") == "structural_contract_passed" for contract in contracts.values())
    )


def _integrated_passed(record: dict[str, Any]) -> bool:
    prepass = record.get("understand_prepass")
    return (
        record.get("agentic_exit_code") == 0
        and record.get("dispatcher_started") is True
        and record.get("child_started") is True
        and record.get("direct_fallback_occurred") is False
        and record.get("tcp_fallback_occurred") is False
        and record.get("provider_turn_count", 0) >= 1
        and isinstance(prepass, dict)
        and prepass.get("provider") == "gemini"
        and prepass.get("model") == "gemini-2.5-flash"
        and prepass.get("ran") is True
        and prepass.get("terminal_call_count") == 1
        and prepass.get("context_map_valid") is True
        and prepass.get("semantic_complete") is True
        and record.get("scanner_started") is True
        and _direct_scan_passed(record)
        and record.get("agentic_report_relative_paths_only") is True
    )


def _classify_integrated_failure(record: dict[str, Any], output_root: Path) -> None:
    startup = _load_json_if_file(output_root / "credential-isolation-startup.json")
    if isinstance(startup, dict):
        runtime_class = startup.get("failure_class")
        if runtime_class == "credential_isolation_private_temp_contract_invalid":
            record.update({
                "status": "failed",
                "failure_stage": "private_temp_validation",
                "failure_class": "qualification_private_temp_contract_invalid",
            })
            return
        if isinstance(runtime_class, str):
            record.update({
                "status": "failed",
                "failure_stage": startup.get("failure_stage") or "credential_isolation_startup",
                "failure_class": runtime_class,
            })
            return
    if record.get("child_started") is True:
        record.update({
            "status": "failed",
            "failure_stage": "agentic_child",
            "failure_class": "agentic_child_internal_error",
        })
    else:
        record.update({
            "status": "failed",
            "failure_stage": "agentic_launcher",
            "failure_class": "agentic_launcher_exit_2_unattributed",
        })


def _preflight_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        name: record.get(name)
        for name in (
            "credential_isolation_required",
            "private_temp_present",
            "temp_aliases_equal",
            "private_temp_absolute",
            "private_temp_canonical",
            "private_temp_non_symlink",
            "private_temp_current_user_owned",
            "private_temp_mode",
            "dispatcher_started",
            "socket_path_bytes",
            "child_spawned",
            "output_digests",
            "child_started",
            "token_fd_available",
            "dispatcher_socket_available",
            "same_uid_and_fd_token_gate",
            "provider_credentials_in_child_environment",
            "direct_fallback_occurred",
            "tcp_fallback_occurred",
            "process_exit_code",
            "failure_stage",
            "failure_class",
            "long_private_temp_exercised",
            "legacy_uds_path_lower_bound_bytes",
            "raw_run_id_absent_from_socket_path",
            "full_run_id_preserved_in_audit",
            "dispatcher_cleanup_complete",
            "private_temp_cleanup_complete",
        )
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "canary", "direct", "integrated"))
    parser.add_argument("--candidate-python", required=True)
    parser.add_argument("--record", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    controller = QualificationController(candidate_python=Path(args.candidate_python))
    if args.mode == "preflight":
        record = controller.run_no_provider_preflight()
    elif args.mode == "canary":
        record = controller.run_canary()
    elif args.mode == "direct":
        record = controller.run_direct()
    else:
        record = controller.run_integrated()
    write_record(Path(args.record), record)
    print(json.dumps({
        "record_kind": record["record_kind"],
        "status": record.get("status"),
        "failure_class": record.get("failure_class"),
    }, sort_keys=True))
    succeeded = record.get("status") in {"qualified", "passed"}
    preflight_passed = args.mode == "preflight" and record.get("failure_class") is None
    return 0 if succeeded or preflight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
