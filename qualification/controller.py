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
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from packages.semgrep.runtime import (
    EXPECTED_SEMGREP_VERSION,
    SemgrepRuntimeError,
    VerifiedSemgrepLauncher,
    verify_explicit_launcher,
)


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
    "system_instruction_sha256",
    "request_schema_sha256",
    "provider_turns_started",
    "provider_turns_completed",
    "terminal_call_count",
    "fixture_read",
    "source_verified",
    "sink_verified",
    "language_verified",
    "semantic_relation_verified",
    "semantic_checks",
    "semantic_failure_reasons",
    "section_counts",
    "worker_cleanup_verified",
    "failure_class",
    "failure_stage",
)
_CANARY_OUTER_SCHEMA_VERSION = 4
_CANARY_MAX_SEMANTIC_FAILURE_REASONS = 16
_CANARY_MAX_SECTION_COUNT = 1_024
_CANARY_SECTION_COUNT_KEYS = (
    "sources",
    "sinks",
    "trust_boundaries",
    "entry_points",
    "sink_details",
    "boundary_details",
    "unchecked_flows",
)
_CANARY_SEMANTIC_CHECK_KEYS = (
    "source_type_verified",
    "source_entry_verified",
    "source_trust_verified",
    "top_level_sink_type_verified",
    "top_level_sink_callsite_verified",
    "sink_detail_type_verified",
    "sink_detail_callsite_verified",
    "sink_detail_wrapper_name_verified",
    "sink_detail_operation_verified",
    "entry_point_callsite_verified",
    "entry_point_relation_name_verified",
    "flow_entry_reference_known",
    "flow_sink_reference_known",
    "flow_links_expected_entry_and_sink",
    "no_declared_boundary_verified",
)
_CANARY_SEMANTIC_FAILURE_REASONS = frozenset({
    "source_type_mismatch",
    "source_entry_mismatch",
    "source_trust_mismatch",
    "top_level_sink_type_mismatch",
    "top_level_sink_callsite_mismatch",
    "sink_detail_type_mismatch",
    "sink_detail_callsite_mismatch",
    "sink_detail_wrapper_name_mismatch",
    "sink_detail_operation_mismatch",
    "entry_point_callsite_mismatch",
    "entry_point_relation_name_mismatch",
    "flow_entry_reference_unknown",
    "flow_sink_reference_unknown",
    "flow_not_linked",
    "declared_boundary_present",
    "semantic_evidence_mismatch",
})


_EXPECTED_JSONSCHEMA_VERSION = "4.26.0"
_SARIF_SCHEMA_RELATIVE_PATH = "engine/schemas/sarif-2.1.0.json"
_EXPECTED_SARIF_SCHEMA_SHA256 = "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a"
_MAX_CANDIDATE_PROBE_OUTPUT_BYTES = 65536
_MAX_PACK_DIAGNOSTICS = 16
_MAX_PACK_STDERR_CHARS = 800
_SARIF_VALIDATION_STATUSES = frozenset({
    "full_valid", "invalid", "full_validation_unavailable", "missing",
})
_SEMGREP_RUNTIME_IDENTITY_SCHEMA_VERSION = 1
_SEMGREP_RUNTIME_PATH_KINDS = frozenset({"governed_private", "unknown", "system"})
_SEMGREP_RUNTIME_SMOKE_STATUSES = frozenset({
    "full_valid", "invalid", "full_validation_unavailable", "missing", "not_run",
})
_DIAGNOSTIC_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|"
    r"cookie|password|secret|session(?:_id)?|token)\b\s*(?:=|:)\s*[^\s,;]+"
)
_DIAGNOSTIC_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_DIAGNOSTIC_SENSITIVE_HEADER = re.compile(
    r"(?im)^(?:authorization|proxy-authorization|cookie|set-cookie|"
    r"x-(?:api-key|auth(?:orization)?|token|secret))\s*:\s*[^\r\n]*$"
)
_DIAGNOSTIC_PRIVATE_PATH = re.compile(
    r"(?<!\w)/(?:Users|home|tmp|private/(?:tmp|var/folders)|var/folders)/[^\s'\"]+"
)
class QualificationControllerError(RuntimeError):
    """A bounded controller invariant failed before a live command."""


@dataclass(frozen=True)
class PrivateTempAttestation:
    """Boolean-only private-temp contract result."""

    credential_isolation_required: bool
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
            self.credential_isolation_required
            and self.present
            and self.aliases_equal
            and self.absolute
            and self.canonical
            and self.non_symlink
            and self.current_user_owned
            and self.mode == "0700"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "credential_isolation_required": self.credential_isolation_required,
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
        semgrep_bin: Path | None = None,
        direct_record: Path | None = None,
        runner: Any = subprocess.run,
    ) -> None:
        self.repo_root = repo_root.resolve()
        selected_python = candidate_python or Path(sys.executable)
        self.candidate_python = Path(os.path.abspath(os.fspath(selected_python)))
        self._runner = runner
        self._integrated_terminal = False
        self._canary_terminal = False
        self._semgrep_launcher: VerifiedSemgrepLauncher | None = None
        self._direct_record = (
            Path(os.path.abspath(os.fspath(direct_record)))
            if direct_record is not None else None
        )
        if semgrep_bin is not None:
            try:
                self._semgrep_launcher = verify_explicit_launcher(semgrep_bin)
            except SemgrepRuntimeError as exc:
                raise QualificationControllerError(
                    "governed Semgrep launcher contract failed"
                ) from exc

    def execution_identity(self) -> ExecutionIdentity:
        """Return the clean checked-out commit and tree for new evidence."""
        dirty = self._git("status", "--porcelain")
        if dirty.stdout.strip():
            raise QualificationControllerError("qualification execution tree is dirty")
        return ExecutionIdentity(
            commit=self._git("rev-parse", "HEAD").stdout.strip(),
            tree=self._git("rev-parse", "HEAD^{tree}").stdout.strip(),
        )

    def _candidate_runtime_preflight(
        self,
        environment: dict[str, str],
    ) -> dict[str, Any]:
        """Attest the exact candidate interpreter and governed Semgrep runtime."""
        if self._semgrep_launcher is None:
            return _candidate_runtime_launcher_required_record()
        capture, payload = self._run_bounded_json(
            [
                str(self.candidate_python),
                "-c",
                _candidate_runtime_probe_source(),
                str(self._semgrep_launcher.lexical_path),
            ],
            env=environment,
            timeout=30,
        )
        return _bounded_candidate_runtime_record(capture, payload)
    def run_no_provider_preflight(self) -> dict[str, Any]:
        """Exercise a real local dispatcher without invoking a provider."""
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        fixture = self._new_fixture(workspace)
        output_root = workspace / "out"
        output_root.mkdir(mode=0o700)
        worker_environment = self._safe_worker_environment(private_root)
        attestation = validate_private_temp(private_root, worker_environment)
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
            "packs_dispatched": 0,
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

            candidate_runtime = self._candidate_runtime_preflight(worker_environment)
            record["candidate_runtime"] = candidate_runtime
            if not _candidate_runtime_ready(candidate_runtime):
                record.update({
                    "process_exit_code": 2,
                    "failure_stage": "candidate_runtime_preflight",
                    "failure_class": _candidate_runtime_failure_class(candidate_runtime),
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
                    env=worker_environment,
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

    def _direct_qualification_gate(self, identity: ExecutionIdentity) -> bool:
        """Accept only a clean, matching, fully-qualified direct record."""
        if self._direct_record is None:
            return False
        try:
            record = _load_json(self._direct_record)
        except (OSError, json.JSONDecodeError, QualificationControllerError):
            return False
        return _qualified_direct_record(record, identity)

    def run_canary(self) -> dict[str, Any]:
        """Run at most one Gemini canary after exact runtime attestations."""
        if self._canary_terminal:
            raise QualificationControllerError("semantic canary already reached a terminal result")
        self._canary_terminal = True
        identity = self.execution_identity()
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        environment = self._single_gemini_environment(private_root)
        argv = [
            str(self.candidate_python),
            "raptor.py",
            "semantic-canary",
            "--model",
            "gemini-2.5-flash",
            "--format",
            "json",
        ]
        record: dict[str, Any] = {
            "schema_version": 2,
            "record_kind": "semantic_canary_attestation",
            "immutable": True,
            "promotable": False,
            "status": "failed",
            "execution": {"commit": identity.commit, "tree": identity.tree},
            "command": {
                "argv": _redact_argv(argv),
                "argv_sha256": _sha256_json(argv),
            },
            "raw_output_persisted": False,
            "exactly_one_valid_submit_context_map": False,
            "direct_qualification_attested": False,
            "semgrep_runtime_preflight": None,
            "semgrep_runtime_postflight": None,
            "semgrep_runtime_identity_matches": False,
            "attestation": {},
            "return_code": None,
            "timed_out": False,
            "stdout_sha256": None,
            "stderr_sha256": None,
            "failure_stage": None,
            "failure_class": None,
        }
        try:
            direct_attested = self._direct_qualification_gate(identity)
            record["direct_qualification_attested"] = direct_attested
            if not direct_attested:
                record.update({
                    "failure_stage": "direct_qualification_gate",
                    "failure_class": "direct_qualification_required",
                })
                return record
            preflight = self._candidate_runtime_preflight(environment)
            record["semgrep_runtime_preflight"] = preflight
            if not _candidate_runtime_ready(preflight):
                record.update({
                    "failure_stage": "candidate_runtime_preflight",
                    "failure_class": _candidate_runtime_failure_class(preflight),
                })
                return record
            capture, payload = self._run_bounded_json(
                argv,
                env=environment,
                timeout=300,
            )
            postflight = self._candidate_runtime_preflight(environment)
            record.update({
                "return_code": capture.return_code,
                "timed_out": capture.timed_out,
                "stdout_sha256": capture.stdout_sha256,
                "stderr_sha256": capture.stderr_sha256,
                "semgrep_runtime_postflight": postflight,
                "semgrep_runtime_identity_matches": _executable_identities_match(
                    preflight.get("semgrep"),
                    postflight.get("semgrep"),
                ),
            })
            attestation = _bounded_semantic_canary_attestation(payload)
            bounded_attestation = {} if attestation is None else attestation
            record["attestation"] = bounded_attestation
            canary_passed = (
                capture.return_code == 0
                and not capture.timed_out
                and _semantic_canary_attestation_passed(attestation)
            )
            runtime_postflight_passed = _candidate_runtime_ready(postflight)
            passed = (
                canary_passed
                and runtime_postflight_passed
                and record["semgrep_runtime_identity_matches"] is True
            )
            if passed:
                record.update({
                    "status": "passed",
                    "exactly_one_valid_submit_context_map": True,
                })
            elif not runtime_postflight_passed:
                record.update({
                    "failure_stage": "candidate_runtime_postflight",
                    "failure_class": _candidate_runtime_failure_class(postflight),
                })
            elif not record["semgrep_runtime_identity_matches"]:
                record.update({
                    "failure_stage": "candidate_runtime_postflight",
                    "failure_class": "candidate_runtime_semgrep_identity_mismatch",
                })
            else:
                record.update({
                    "failure_stage": bounded_attestation.get("failure_stage")
                    or "semantic_canary",
                    "failure_class": bounded_attestation.get("failure_class")
                    or "semantic_canary_contract_failed",
                })
        finally:
            self._cleanup_workspace(workspace, record)
        return record
    def run_direct(self) -> dict[str, Any]:
        """Run one direct strict-Semgrep qualification without a provider."""
        identity = self.execution_identity()
        workspace = self._workspace()
        private_root = self._new_private_root(workspace)
        fixture = self._new_fixture(workspace)
        output_root = workspace / "out"
        output_root.mkdir(mode=0o700)
        scan_out = output_root / "scan"
        worker_environment = self._safe_worker_environment(private_root)
        attestation = validate_private_temp(private_root, worker_environment)
        argv = [
            str(self.candidate_python),
            "packages/static-analysis/scanner.py",
            "--repo", str(fixture),
            "--out", str(scan_out),
            "--sandbox", "strict",
            "--no-codeql",
            "--policy-groups", "all",
        ]
        if self._semgrep_launcher is not None:
            argv.extend(["--semgrep-bin", str(self._semgrep_launcher.lexical_path)])
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
            candidate_runtime = self._candidate_runtime_preflight(worker_environment)
            record["candidate_runtime"] = candidate_runtime
            if not _candidate_runtime_ready(candidate_runtime):
                record.update({
                    "process_exit_code": 2,
                    "scanner_exit_code": None,
                    "packs_dispatched": 0,
                    "failure_stage": "candidate_runtime_preflight",
                    "failure_class": _candidate_runtime_failure_class(candidate_runtime),
                })
                return record
            capture = self._run(argv, env=worker_environment)
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
                    "failure_class": (
                        record["semgrep"]["failure_class"]
                        if (
                            isinstance(record.get("semgrep"), dict)
                            and isinstance(record["semgrep"].get("failure_class"), str)
                        )
                        else "direct_semgrep_contract_failed"
                    ),
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
        launcher_environment = self._launcher_environment(private_root)
        attestation = validate_private_temp(private_root, launcher_environment)
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
        if self._semgrep_launcher is not None:
            argv.extend(["--semgrep-bin", str(self._semgrep_launcher.lexical_path)])
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
            capture = self._run(argv, env=launcher_environment, timeout=420)
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
            "packs_dispatched": 0,
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
    def _single_gemini_environment(self, private_root: Path) -> dict[str, str]:
        """Return the canary environment with only its one Gemini credential."""
        env = {
            name: value
            for name, value in os.environ.items()
            if not _is_provider_environment_name(name)
        }
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            env["GEMINI_API_KEY"] = gemini_key
        env.pop("RAPTOR_LLM_SOCKET", None)
        env.pop("RAPTOR_LLM_TOKEN_FD", None)
        env["RAPTOR_REQUIRE_CREDENTIAL_ISOLATION"] = "1"
        root_text = str(private_root)
        for name in _PRIVATE_TEMP_ENV:
            env[name] = root_text
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
        summary_version = summary.get("schema_version")
        raw_packs = summary.get("packs")
        diagnostics = [
            diagnostic
            for pack in (raw_packs[:_MAX_PACK_DIAGNOSTICS] if isinstance(raw_packs, list) else [])
            if isinstance(pack, dict)
            for diagnostic in (_bounded_pack_diagnostic(pack, summary_version),)
            if diagnostic is not None
        ]
        sandbox = summary.get("sandbox_engagement")
        sandbox_state = sandbox.get("state") if isinstance(sandbox, dict) else None
        dispatched = _bounded_nonnegative_int(summary.get("packs_dispatched"))
        succeeded = _bounded_nonnegative_int(summary.get("packs_succeeded"))
        failed = _bounded_nonnegative_int(summary.get("packs_failed"))
        failure_class, failure_counts, failure_source, unclassified = (
            _pack_failure_evidence(
                diagnostics,
                failed,
                summary.get("failure_class"),
            )
        )
        combined_status = _summary_combined_validation_status(summary, summary_version)
        per_pack_full = (
            dispatched is not None
            and dispatched > 0
            and dispatched == len(diagnostics)
            and all(
                diagnostic["sarif_validation_status"] == "full_valid"
                for diagnostic in diagnostics
            )
        )
        scanner_identity = (
            _bounded_semgrep_runtime_identity(summary.get("scanner"))
            if summary_version == 3
            else _bounded_executable_identity(summary.get("scanner"))
        )
        combined_inputs_complete = (
            summary.get("combined_inputs_complete") is True
            if summary_version == 3 else False
        )
        combined_usable = (
            summary.get("combined_usable") is True
            if summary_version == 3 else False
        )
        candidate_runtime = record.get("candidate_runtime")
        candidate_semgrep = (
            candidate_runtime.get("semgrep")
            if isinstance(candidate_runtime, dict) else None
        )
        runtime_identity_matches = _executable_identities_match(
            candidate_semgrep,
            scanner_identity,
        )
        record["packs_dispatched"] = dispatched
        record["semgrep"] = {
            "summary_schema_version": summary_version,
            "scanner": scanner_identity,
            "runtime_identity_matches_preflight": runtime_identity_matches,
            "packs_dispatched": dispatched,
            "packs_succeeded": succeeded,
            "packs_failed": failed,
            "aggregate_exit_code": summary.get("aggregate_exit_code"),
            "failure_class": failure_class,
            "failure_class_source": failure_source,
            "failure_class_counts": failure_counts,
            "unclassified_pack_count": unclassified,
            "pack_diagnostics": diagnostics,
            "combined_sarif_validation_status": combined_status,
            "combined_inputs_complete": combined_inputs_complete,
            "combined_usable": combined_usable,
            "per_pack_full_validation": per_pack_full,
            "sandbox_engagement": sandbox_state,
        }
        jsonschema_identity = (
            candidate_runtime.get("jsonschema")
            if isinstance(candidate_runtime, dict) else None
        )
        schema_identity = (
            candidate_runtime.get("sarif_schema")
            if isinstance(candidate_runtime, dict) else None
        )
        record["sarif_validation"] = {
            "canonical_schema": _SARIF_SCHEMA_RELATIVE_PATH,
            "schema_sha256": schema_identity.get("sha256")
            if isinstance(schema_identity, dict) else None,
            "schema_hash_matches": schema_identity.get("hash_matches") is True
            if isinstance(schema_identity, dict) else False,
            "per_pack_full_validation": per_pack_full,
            "combined_full_validation": combined_status == "full_valid",
            "combined_validation_status": combined_status,
            "combined_inputs_complete": combined_inputs_complete,
            "combined_usable": combined_usable,
            "jsonschema_distribution": "jsonschema",
            "jsonschema_version": jsonschema_identity.get("version")
            if isinstance(jsonschema_identity, dict) else None,
            "jsonschema_version_matches": jsonschema_identity.get("version_matches") is True
            if isinstance(jsonschema_identity, dict) else False,
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


def validate_private_temp(
    private_root: Path,
    environment: Mapping[str, str],
) -> PrivateTempAttestation:
    """Validate the exact environment mapping passed to a bounded child."""
    root_text = str(private_root)
    values = {name: environment.get(name) for name in _PRIVATE_TEMP_ENV}
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
        credential_isolation_required=(
            environment.get("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION") == "1"
        ),
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
    placeholders = {
        "--repo": "<fresh_inert_cpp_git_fixture>",
        "--out": "<fresh_empty_output>",
        "--semgrep-bin": "<verified-semgrep-bin>",
    }
    redact_next = False
    for index, value in enumerate(argv):
        if index == 0:
            redacted.append("candidate-python")
            continue
        if redact_next:
            redacted.append(placeholders[argv[index - 1]])
            redact_next = False
            continue
        redacted.append(value)
        if value in placeholders:
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


def _safe_sha256(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    return None


def _safe_identifier(value: object, *, limit: int = 160) -> str | None:
    if not isinstance(value, str) or len(value) > limit:
        return None
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value):
        return value
    return None


def _safe_basename(value: object) -> str | None:
    candidate = _safe_identifier(value, limit=128)
    if candidate is None or Path(candidate).name != candidate:
        return None
    return candidate


def _bounded_nonnegative_int(value: object) -> int | None:
    if type(value) is int and 0 <= value <= 1_000_000:
        return value
    return None


def _bounded_canary_semantic_checks(value: object) -> dict[str, bool] | None:
    if not isinstance(value, dict) or set(value) != set(_CANARY_SEMANTIC_CHECK_KEYS):
        return None
    if any(type(value[key]) is not bool for key in _CANARY_SEMANTIC_CHECK_KEYS):
        return None
    return {key: value[key] for key in _CANARY_SEMANTIC_CHECK_KEYS}


def _bounded_canary_semantic_failure_reasons(value: object) -> list[str] | None:
    if (
        not isinstance(value, list)
        or len(value) > _CANARY_MAX_SEMANTIC_FAILURE_REASONS
        or any(
            not isinstance(reason, str)
            or reason not in _CANARY_SEMANTIC_FAILURE_REASONS
            for reason in value
        )
        or value != sorted(set(value))
    ):
        return None
    return list(value)


def _bounded_canary_section_counts(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) != set(_CANARY_SECTION_COUNT_KEYS):
        return None
    if any(
        type(value[key]) is not int
        or not 0 <= value[key] <= _CANARY_MAX_SECTION_COUNT
        for key in _CANARY_SECTION_COUNT_KEYS
    ):
        return None
    return {key: value[key] for key in _CANARY_SECTION_COUNT_KEYS}


def _bounded_semantic_canary_attestation(payload: object) -> dict[str, Any] | None:
    """Retain only the exact v4 controller contract from child JSON."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _CANARY_OUTER_SCHEMA_VERSION
        or payload.get("status") not in {"passed", "failed"}
        or payload.get("provider") != "gemini"
        or payload.get("model") != "gemini-2.5-flash"
    ):
        return None
    sdk_version = _safe_identifier(payload.get("sdk_version"), limit=128)
    system_instruction_sha256 = _safe_sha256(
        payload.get("system_instruction_sha256")
    )
    request_schema_sha256 = _safe_sha256(payload.get("request_schema_sha256"))
    counters = {
        key: payload.get(key)
        for key in (
            "provider_turns_started",
            "provider_turns_completed",
            "terminal_call_count",
        )
    }
    booleans = {
        key: payload.get(key)
        for key in (
            "fixture_read",
            "source_verified",
            "sink_verified",
            "language_verified",
            "semantic_relation_verified",
            "worker_cleanup_verified",
        )
    }
    semantic_checks = _bounded_canary_semantic_checks(
        payload.get("semantic_checks")
    )
    semantic_failure_reasons = _bounded_canary_semantic_failure_reasons(
        payload.get("semantic_failure_reasons")
    )
    section_counts = _bounded_canary_section_counts(payload.get("section_counts"))
    if (
        sdk_version is None
        or system_instruction_sha256 is None
        or request_schema_sha256 is None
        or any(
            type(value) is not int or not 0 <= value <= 3
            for value in counters.values()
        )
        or any(type(value) is not bool for value in booleans.values())
        or semantic_checks is None
        or semantic_failure_reasons is None
        or section_counts is None
    ):
        return None
    status = payload["status"]
    failure: dict[str, str] = {}
    if status == "passed":
        if "failure_class" in payload or "failure_stage" in payload:
            return None
    else:
        failure_class = _safe_identifier(payload.get("failure_class"), limit=128)
        failure_stage = _safe_identifier(payload.get("failure_stage"), limit=128)
        if failure_class is None or failure_stage is None:
            return None
        failure = {
            "failure_class": failure_class,
            "failure_stage": failure_stage,
        }
    return {
        "schema_version": _CANARY_OUTER_SCHEMA_VERSION,
        "status": status,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "sdk_version": sdk_version,
        "system_instruction_sha256": system_instruction_sha256,
        "request_schema_sha256": request_schema_sha256,
        **counters,
        **booleans,
        "semantic_checks": semantic_checks,
        "semantic_failure_reasons": semantic_failure_reasons,
        "section_counts": section_counts,
        **failure,
    }


def _semantic_canary_attestation_passed(attestation: dict[str, Any] | None) -> bool:
    return bool(
        attestation is not None
        and attestation["status"] == "passed"
        and attestation["provider_turns_started"] == 2
        and attestation["provider_turns_completed"] == 2
        and attestation["terminal_call_count"] == 1
        and all(
            attestation[key] is True
            for key in (
                "fixture_read",
                "source_verified",
                "sink_verified",
                "language_verified",
                "semantic_relation_verified",
                "worker_cleanup_verified",
            )
        )
        and all(attestation["semantic_checks"].values())
        and attestation["semantic_failure_reasons"] == []
    )


def _bounded_diagnostic_tail(value: object) -> str:
    text = str(value or "")
    text = _DIAGNOSTIC_SENSITIVE_HEADER.sub("<redacted-header>", text)
    text = _DIAGNOSTIC_BEARER_VALUE.sub("<redacted-bearer>", text)
    text = _DIAGNOSTIC_SENSITIVE_VALUE.sub("<redacted>", text)
    text = _DIAGNOSTIC_PRIVATE_PATH.sub("<runtime-path>", text)
    text = "".join(
        character if character in "\n\t" or ord(character) >= 32 else "?"
        for character in text
    )
    return text[-_MAX_PACK_STDERR_CHARS:]


def _bounded_executable_identity(value: object) -> dict[str, Any] | None:
    """Retain historical v1/v2 scanner identity without treating it as healthy."""
    if not isinstance(value, dict):
        return None
    path_kind = value.get("path_kind")
    return {
        "basename": _safe_basename(value.get("basename")),
        "executable": value.get("executable") is True,
        "version": _safe_identifier(value.get("version"), limit=80),
        "sha256": _safe_sha256(value.get("sha256")),
        "path_kind": path_kind
        if path_kind in {"system", "governed_private", "unknown"} else "unknown",
    }


def _bounded_semgrep_runtime_identity(value: object) -> dict[str, Any] | None:
    """Retain the v3 governed runtime contract without its lexical path."""
    if not isinstance(value, dict):
        return None
    launcher_mode = value.get("launcher_lstat_mode")
    return {
        "identity_schema_version": (
            value.get("identity_schema_version")
            if value.get("identity_schema_version") == _SEMGREP_RUNTIME_IDENTITY_SCHEMA_VERSION
            else None
        ),
        "launcher_basename": _safe_basename(value.get("launcher_basename")),
        "launcher_string_sha256": _safe_sha256(value.get("launcher_string_sha256")),
        "launcher_lstat_mode": launcher_mode
        if isinstance(launcher_mode, str) and re.fullmatch(r"0[0-7]{3}", launcher_mode)
        else None,
        "launcher_symlink": value.get("launcher_symlink") is True,
        "resolved_executable_sha256": _safe_sha256(
            value.get("resolved_executable_sha256")
        ),
        "path_kind": value.get("path_kind")
        if value.get("path_kind") in _SEMGREP_RUNTIME_PATH_KINDS else "unknown",
        "version": _safe_identifier(value.get("version"), limit=80),
        "version_parse_source": value.get("version_parse_source")
        if value.get("version_parse_source") == "stdout" else None,
        "version_probe_return_code": (
            value.get("version_probe_return_code")
            if type(value.get("version_probe_return_code")) is int
            and -255 <= value.get("version_probe_return_code") <= 255
            else None
        ),
        "version_probe_timed_out": value.get("version_probe_timed_out") is True,
        "version_probe_stdout_sha256": _safe_sha256(
            value.get("version_probe_stdout_sha256")
        ),
        "version_probe_stderr_sha256": _safe_sha256(
            value.get("version_probe_stderr_sha256")
        ),
        "engine_smoke_return_code": (
            value.get("engine_smoke_return_code")
            if type(value.get("engine_smoke_return_code")) is int
            and -255 <= value.get("engine_smoke_return_code") <= 255
            else None
        ),
        "engine_smoke_timed_out": value.get("engine_smoke_timed_out") is True,
        "engine_smoke_stdout_sha256": _safe_sha256(
            value.get("engine_smoke_stdout_sha256")
        ),
        "engine_smoke_stderr_sha256": _safe_sha256(
            value.get("engine_smoke_stderr_sha256")
        ),
        "engine_smoke_sarif_status": (
            value.get("engine_smoke_sarif_status")
            if value.get("engine_smoke_sarif_status") in _SEMGREP_RUNTIME_SMOKE_STATUSES
            else "not_run"
        ),
        "engine_smoke_raw_output_persisted": (
            value.get("engine_smoke_raw_output_persisted") is True
        ),
        "dependency_closure_sha256": _safe_sha256(
            value.get("dependency_closure_sha256")
        ),
        "semgrep_core_sha256": _safe_sha256(value.get("semgrep_core_sha256")),
        "failure_class": _safe_identifier(value.get("failure_class"), limit=96),
        "linker_family": value.get("linker_family")
        if value.get("linker_family") in {"dyld", "ld.so", "windows_loader"}
        else None,
        "missing_library_basename": _safe_basename(value.get("missing_library_basename")),
        "healthy": value.get("healthy") is True,
    }


def _semgrep_runtime_identity_healthy(value: object) -> bool:
    identity = _bounded_semgrep_runtime_identity(value)
    if identity is None:
        return False
    return (
        identity["identity_schema_version"] == _SEMGREP_RUNTIME_IDENTITY_SCHEMA_VERSION
        and identity["launcher_basename"] == "semgrep"
        and identity["launcher_string_sha256"] is not None
        and identity["launcher_lstat_mode"] is not None
        and identity["resolved_executable_sha256"] is not None
        and identity["version_parse_source"] == "stdout"
        and identity["path_kind"] == "governed_private"
        and identity["version"] == EXPECTED_SEMGREP_VERSION
        and identity["version_probe_return_code"] == 0
        and identity["version_probe_timed_out"] is False
        and identity["engine_smoke_return_code"] in (0, 1)
        and identity["engine_smoke_timed_out"] is False
        and identity["engine_smoke_sarif_status"] == "full_valid"
        and identity["engine_smoke_raw_output_persisted"] is False
        and identity["dependency_closure_sha256"] is not None
        and identity["semgrep_core_sha256"] is not None
        and identity["failure_class"] is None
        and identity["healthy"] is True
    )


def _executable_identities_match(left: object, right: object) -> bool:
    left_identity = _bounded_semgrep_runtime_identity(left)
    right_identity = _bounded_semgrep_runtime_identity(right)
    if not _semgrep_runtime_identity_healthy(left_identity):
        return False
    if not _semgrep_runtime_identity_healthy(right_identity):
        return False
    # Probe output digests attest each run; temporary roots and diagnostics vary.
    return all(
        left_identity[field] == right_identity[field]
        for field in (
            "identity_schema_version",
            "launcher_basename",
            "launcher_string_sha256",
            "launcher_lstat_mode",
            "launcher_symlink",
            "resolved_executable_sha256",
            "path_kind",
            "version",
            "version_parse_source",
            "version_probe_return_code",
            "version_probe_timed_out",
            "engine_smoke_return_code",
            "engine_smoke_timed_out",
            "engine_smoke_sarif_status",
            "engine_smoke_raw_output_persisted",
            "dependency_closure_sha256",
            "semgrep_core_sha256",
            "failure_class",
            "linker_family",
            "missing_library_basename",
            "healthy",
        )
    )
def _summary_pack_validation_status(pack: Mapping[str, Any], version_value: object) -> str:
    if version_value == 1:
        if pack.get("sarif_exists") is not True:
            return "missing"
        return "full_valid" if pack.get("sarif_valid") is True else "invalid"
    status = pack.get("sarif_validation_status")
    return status if status in _SARIF_VALIDATION_STATUSES else "invalid"


def _summary_combined_validation_status(
    summary: Mapping[str, Any],
    version_value: object,
) -> str:
    if version_value == 1:
        if summary.get("combined_sarif_exists") is not True:
            return "missing"
        return "full_valid" if summary.get("combined_sarif_valid") is True else "invalid"
    status = summary.get("combined_sarif_validation_status")
    return status if status in _SARIF_VALIDATION_STATUSES else "invalid"


def _bounded_pack_diagnostic(
    pack: Mapping[str, Any],
    summary_version: object,
) -> dict[str, Any] | None:
    raw_name = pack.get("pack_name")
    pack_name = _safe_identifier(raw_name, limit=96)
    if pack_name is None:
        pack_name = "custom_" + hashlib.sha256(
            str(raw_name).encode("utf-8", errors="replace")
        ).hexdigest()[:16]
    config_kind = pack.get("config_kind")
    if config_kind not in {"local_file", "local_directory", "registry", "unknown"}:
        config_kind = "unknown"
    failure_class = _safe_identifier(pack.get("failure_class"), limit=96)
    return {
        "pack_name": pack_name,
        "config_kind": config_kind,
        "config_sha256": _safe_sha256(pack.get("config_sha256")),
        "raw_exit_code": pack.get("raw_exit_code")
        if type(pack.get("raw_exit_code")) is int else None,
        "sarif_exists": pack.get("sarif_exists") is True,
        "sarif_validation_status": _summary_pack_validation_status(
            pack, summary_version,
        ),
        "failure_class": failure_class,
        "bounded_stderr_tail": _bounded_diagnostic_tail(
            pack.get("bounded_stderr_tail"),
        ),
        "sandbox_denial_count": _bounded_nonnegative_int(
            pack.get("sandbox_denial_count"),
        ) or 0,
        "proxy_event_count": _bounded_nonnegative_int(
            pack.get("proxy_event_count"),
        ) or 0,
    }


def _pack_failure_evidence(
    diagnostics: Sequence[Mapping[str, Any]],
    failed: int | None,
    scanner_failure: object,
) -> tuple[str | None, dict[str, int], str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        failure_class = diagnostic.get("failure_class")
        if isinstance(failure_class, str):
            counts[failure_class] = counts.get(failure_class, 0) + 1
    counts = dict(sorted(counts.items()))
    classified = sum(counts.values())
    expected_failed = failed if failed is not None else classified
    unclassified = max(0, expected_failed - classified)
    if expected_failed == 0:
        bounded_scanner_failure = _safe_identifier(scanner_failure, limit=96)
        return (
            bounded_scanner_failure,
            counts,
            "scanner" if bounded_scanner_failure else "no_failed_packs",
            0,
        )
    if classified == 0:
        return (
            "pack_failure_class_unavailable",
            counts,
            "no_per_pack_failure_class",
            unclassified,
        )
    if unclassified or len(counts) > 1:
        return "mixed_pack_failures", counts, "mixed_pack_failure_classes", unclassified
    return next(iter(counts)), counts, "uniform_pack_failure_class", 0


def _empty_semgrep_runtime_identity(
    *,
    failure_class: str | None,
) -> dict[str, Any]:
    return {
        "identity_schema_version": _SEMGREP_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "launcher_basename": None,
        "launcher_string_sha256": None,
        "launcher_lstat_mode": None,
        "launcher_symlink": False,
        "version_parse_source": None,
        "resolved_executable_sha256": None,
        "path_kind": "unknown",
        "version": None,
        "version_probe_return_code": None,
        "version_probe_timed_out": False,
        "version_probe_stdout_sha256": None,
        "version_probe_stderr_sha256": None,
        "engine_smoke_return_code": None,
        "engine_smoke_timed_out": False,
        "engine_smoke_stdout_sha256": None,
        "engine_smoke_stderr_sha256": None,
        "engine_smoke_sarif_status": "not_run",
        "engine_smoke_raw_output_persisted": False,
        "dependency_closure_sha256": None,
        "semgrep_core_sha256": None,
        "failure_class": failure_class,
        "linker_family": None,
        "missing_library_basename": None,
        "healthy": False,
    }


def _candidate_runtime_launcher_required_record() -> dict[str, Any]:
    return {
        "passed": False,
        "failure_class": "candidate_runtime_semgrep_launcher_required",
        "probe_return_code": None,
        "probe_timed_out": False,
        "probe_output_digests": {"stdout_sha256": None, "stderr_sha256": None},
        "candidate_python": {
            "implementation": None,
            "version": {"major": None, "minor": None, "patch": None},
            "executable_basename": None,
            "path_kind": "unknown",
            "resolved_executable_sha256": None,
        },
        "jsonschema": {
            "importable": False,
            "version": None,
            "expected_version": _EXPECTED_JSONSCHEMA_VERSION,
            "version_matches": False,
        },
        "sarif_schema": {
            "relative_path": _SARIF_SCHEMA_RELATIVE_PATH,
            "present": False,
            "sha256": None,
            "expected_sha256": _EXPECTED_SARIF_SCHEMA_SHA256,
            "hash_matches": False,
        },
        "semgrep": _empty_semgrep_runtime_identity(
            failure_class="semgrep_runtime_launcher_required",
        ),
    }


def _candidate_runtime_semgrep_failure(value: object) -> str:
    identity = _bounded_semgrep_runtime_identity(value)
    if identity is None:
        return "candidate_runtime_semgrep_version_probe_failed"
    failure = identity.get("failure_class")
    mapping = {
        "semgrep_runtime_launcher_required": "candidate_runtime_semgrep_launcher_required",
        "semgrep_runtime_launcher_invalid": "candidate_runtime_semgrep_launcher_invalid",
        "semgrep_runtime_linker_dependency_missing": (
            "candidate_runtime_semgrep_linker_dependency_missing"
        ),
        "semgrep_runtime_process_aborted": "candidate_runtime_semgrep_process_aborted",
        "semgrep_runtime_version_probe_failed": (
            "candidate_runtime_semgrep_version_probe_failed"
        ),
        "semgrep_runtime_version_unparseable": (
            "candidate_runtime_semgrep_version_unparseable"
        ),
        "semgrep_runtime_engine_smoke_failed": (
            "candidate_runtime_semgrep_engine_smoke_failed"
        ),
        "semgrep_runtime_dependency_closure_invalid": (
            "candidate_runtime_semgrep_dependency_closure_invalid"
        ),
    }
    if failure in mapping:
        return mapping[failure]
    if identity.get("version") != EXPECTED_SEMGREP_VERSION:
        return "candidate_runtime_semgrep_version_probe_failed"
    if identity.get("version_probe_return_code") != 0:
        return "candidate_runtime_semgrep_version_probe_failed"
    if identity.get("engine_smoke_sarif_status") != "full_valid":
        return "candidate_runtime_semgrep_engine_smoke_failed"
    if identity.get("dependency_closure_sha256") is None:
        return "candidate_runtime_semgrep_dependency_closure_invalid"
    return "candidate_runtime_semgrep_version_probe_failed"


def _candidate_runtime_ready(runtime: object) -> bool:
    return (
        isinstance(runtime, dict)
        and runtime.get("passed") is True
        and _semgrep_runtime_identity_healthy(runtime.get("semgrep"))
    )


def _candidate_runtime_failure_class(runtime: object) -> str:
    if isinstance(runtime, dict):
        failure = runtime.get("failure_class")
        if isinstance(failure, str) and failure.startswith("candidate_runtime_"):
            return failure
        return _candidate_runtime_semgrep_failure(runtime.get("semgrep"))
    return "candidate_runtime_probe_invalid"


def _bounded_candidate_runtime_record(
    capture: ProcessCapture,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_python = payload.get("candidate_python")
    raw_jsonschema = payload.get("jsonschema")
    raw_schema = payload.get("sarif_schema")
    raw_semgrep = payload.get("semgrep")
    python_identity = raw_python if isinstance(raw_python, dict) else {}
    raw_version = python_identity.get("version")
    version_parts = (
        list(raw_version)
        if isinstance(raw_version, list)
        and len(raw_version) == 3
        and all(type(part) is int and part >= 0 for part in raw_version)
        else []
    )
    candidate_python = {
        "implementation": _safe_identifier(python_identity.get("implementation"), limit=32),
        "version": {
            "major": version_parts[0] if version_parts else None,
            "minor": version_parts[1] if version_parts else None,
            "patch": version_parts[2] if version_parts else None,
        },
        "executable_basename": _safe_basename(python_identity.get("basename")),
        "path_kind": python_identity.get("path_kind")
        if python_identity.get("path_kind") in {"system", "governed_private", "unknown"}
        else "unknown",
        "resolved_executable_sha256": _safe_sha256(python_identity.get("sha256")),
    }
    jsonschema_payload = raw_jsonschema if isinstance(raw_jsonschema, dict) else {}
    jsonschema_version = _safe_identifier(jsonschema_payload.get("version"), limit=40)
    jsonschema_identity = {
        "importable": jsonschema_payload.get("importable") is True,
        "version": jsonschema_version,
        "expected_version": _EXPECTED_JSONSCHEMA_VERSION,
        "version_matches": jsonschema_version == _EXPECTED_JSONSCHEMA_VERSION,
    }
    schema_payload = raw_schema if isinstance(raw_schema, dict) else {}
    schema_sha256 = _safe_sha256(schema_payload.get("sha256"))
    schema_identity = {
        "relative_path": _SARIF_SCHEMA_RELATIVE_PATH,
        "present": schema_payload.get("present") is True
        and schema_payload.get("relative_path") == _SARIF_SCHEMA_RELATIVE_PATH,
        "sha256": schema_sha256,
        "expected_sha256": _EXPECTED_SARIF_SCHEMA_SHA256,
        "hash_matches": schema_sha256 == _EXPECTED_SARIF_SCHEMA_SHA256,
    }
    semgrep_identity = _bounded_semgrep_runtime_identity(raw_semgrep) or (
        _empty_semgrep_runtime_identity(failure_class="semgrep_runtime_probe_invalid")
    )
    candidate_identity_valid = (
        candidate_python["implementation"] is not None
        and bool(version_parts)
        and candidate_python["executable_basename"] is not None
    )
    if capture.timed_out:
        failure_class = "candidate_runtime_probe_timeout"
    elif capture.return_code != 0:
        failure_class = "candidate_runtime_probe_failed"
    elif payload.get("schema_version") != 2 or not candidate_identity_valid:
        failure_class = "candidate_runtime_probe_invalid"
    elif not jsonschema_identity["importable"]:
        failure_class = "candidate_runtime_dependency_missing"
    elif not jsonschema_identity["version_matches"]:
        failure_class = "candidate_runtime_dependency_version_mismatch"
    elif not schema_identity["present"]:
        failure_class = "candidate_runtime_sarif_schema_missing"
    elif not schema_identity["hash_matches"]:
        failure_class = "candidate_runtime_sarif_schema_hash_mismatch"
    elif not _semgrep_runtime_identity_healthy(semgrep_identity):
        failure_class = _candidate_runtime_semgrep_failure(semgrep_identity)
    else:
        failure_class = None
    return {
        "passed": failure_class is None,
        "failure_class": failure_class,
        "probe_return_code": capture.return_code,
        "probe_timed_out": capture.timed_out,
        "probe_output_digests": {
            "stdout_sha256": capture.stdout_sha256,
            "stderr_sha256": capture.stderr_sha256,
        },
        "candidate_python": candidate_python,
        "jsonschema": jsonschema_identity,
        "sarif_schema": schema_identity,
        "semgrep": semgrep_identity,
    }


def _candidate_runtime_probe_source() -> str:
    return r"""
import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from core.sandbox import SANDBOX_ENGAGE_EXIT_CODE, SandboxSetupError, run
from packages.semgrep.runtime import (
    SemgrepRuntimeError,
    collect_runtime_health,
    verify_explicit_launcher,
)

OUTPUT_LIMIT = 65536
try:
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))
except (ImportError, OSError, ValueError):
    pass


def digest(path):
    try:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    except OSError:
        return None


def path_kind(path):
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return "unknown"
    rendered = str(resolved)
    if rendered.startswith((
        "/bin/", "/sbin/", "/usr/", "/opt/", "/System/Library/",
        "/Library/Frameworks/",
    )):
        return "system"
    current_uid = getattr(os, "geteuid", os.getuid)()
    for parent in (resolved.parent, *resolved.parents):
        try:
            observed = parent.stat()
        except OSError:
            continue
        if observed.st_uid == current_uid and stat.S_IMODE(observed.st_mode) == 0o700:
            return "governed_private"
    return "unknown"


def strict_runtime_runner(argv, **kwargs):
    if argv[-1] == "--version":
        probe_root = Path(tempfile.mkdtemp(
            prefix="semgrep-runtime-version-",
            dir=str(launcher.private_root),
        ))
        cleanup_root = True
    else:
        source = Path(argv[-1]).resolve()
        probe_root = source.parent
        cleanup_root = False
    try:
        return run(
            argv,
            timeout=kwargs["timeout"],
            env=kwargs["env"],
            stdin=kwargs["stdin"],
            capture_output=True,
            text=kwargs["text"],
            check=kwargs["check"],
            cwd=str(probe_root),
            target=str(probe_root),
            output=str(probe_root),
            proxy_hosts=None,
            caller_label="qualification-semgrep-runtime",
            fake_home=True,
            profile="strict",
            restrict_reads=True,
            readable_paths=[str(launcher.private_root)],
            tool_paths=[str(launcher.private_root)],
        )
    except (SandboxSetupError, RuntimeError) as exc:
        return subprocess.CompletedProcess(
            argv,
            SANDBOX_ENGAGE_EXIT_CODE,
            b"",
            str(exc).encode("utf-8", errors="replace"),
        )
    finally:
        if cleanup_root:
            import shutil
            shutil.rmtree(probe_root, ignore_errors=True)


python_path = Path(sys.executable)
try:
    import jsonschema
    jsonschema_importable = True
except ImportError:
    jsonschema_importable = False
try:
    jsonschema_version = importlib.metadata.version("jsonschema")
except importlib.metadata.PackageNotFoundError:
    jsonschema_version = None
schema_relative = "engine/schemas/sarif-2.1.0.json"
schema_path = Path(schema_relative)
semgrep_fields = (
    "identity_schema_version", "launcher_basename", "launcher_string_sha256",
    "launcher_lstat_mode", "launcher_symlink", "resolved_executable_sha256",
    "path_kind", "version", "version_parse_source", "version_probe_return_code",
    "version_probe_timed_out", "version_probe_stdout_sha256",
    "version_probe_stderr_sha256", "engine_smoke_return_code",
    "engine_smoke_timed_out", "engine_smoke_stdout_sha256",
    "engine_smoke_stderr_sha256", "engine_smoke_sarif_status",
    "engine_smoke_raw_output_persisted", "dependency_closure_sha256",
    "semgrep_core_sha256", "failure_class", "linker_family",
    "missing_library_basename", "healthy",
)
if len(sys.argv) != 2:
    semgrep_runtime = {"failure_class": "semgrep_runtime_launcher_invalid", "healthy": False}
else:
    try:
        launcher = verify_explicit_launcher(Path(sys.argv[1]))
        semgrep_runtime = collect_runtime_health(
            launcher,
            environment=dict(os.environ),
            runner=strict_runtime_runner,
            engine_runner=strict_runtime_runner,
        )
    except SemgrepRuntimeError:
        semgrep_runtime = {"failure_class": "semgrep_runtime_launcher_invalid", "healthy": False}
semgrep = {field: semgrep_runtime.get(field) for field in semgrep_fields}
payload = {
    "schema_version": 2,
    "candidate_python": {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:3]),
        "basename": python_path.name,
        "path_kind": path_kind(python_path),
        "sha256": digest(python_path.resolve()),
    },
    "jsonschema": {
        "importable": jsonschema_importable,
        "version": jsonschema_version,
    },
    "sarif_schema": {
        "relative_path": schema_relative,
        "present": schema_path.is_file(),
        "sha256": digest(schema_path) if schema_path.is_file() else None,
    },
    "semgrep": semgrep,
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
if len(encoded.encode("utf-8")) > OUTPUT_LIMIT:
    raise SystemExit(3)
print(encoded)
"""
def _direct_scan_passed(record: dict[str, Any]) -> bool:
    semgrep = record.get("semgrep")
    validation = record.get("sarif_validation")
    contracts = record.get("artifact_contracts")
    runtime = record.get("candidate_runtime")
    jsonschema_identity = runtime.get("jsonschema") if isinstance(runtime, dict) else None
    schema_identity = runtime.get("sarif_schema") if isinstance(runtime, dict) else None
    runtime_semgrep = runtime.get("semgrep") if isinstance(runtime, dict) else None
    scanner_identity = semgrep.get("scanner") if isinstance(semgrep, dict) else None
    diagnostics = semgrep.get("pack_diagnostics") if isinstance(semgrep, dict) else None
    return (
        record.get("promotable") is False
        and record.get("provider_turn_count") == 0
        and record.get("process_exit_code") == 0
        and record.get("scanner_exit_code") == 0
        and record.get("scanner_started") is True
        and _candidate_runtime_ready(runtime)
        and isinstance(jsonschema_identity, dict)
        and jsonschema_identity.get("importable") is True
        and jsonschema_identity.get("version") == _EXPECTED_JSONSCHEMA_VERSION
        and jsonschema_identity.get("version_matches") is True
        and isinstance(schema_identity, dict)
        and schema_identity.get("relative_path") == _SARIF_SCHEMA_RELATIVE_PATH
        and schema_identity.get("present") is True
        and schema_identity.get("sha256") == _EXPECTED_SARIF_SCHEMA_SHA256
        and schema_identity.get("hash_matches") is True
        and _semgrep_runtime_identity_healthy(runtime_semgrep)
        and isinstance(semgrep, dict)
        and semgrep.get("summary_schema_version") == 3
        and _semgrep_runtime_identity_healthy(scanner_identity)
        and semgrep.get("runtime_identity_matches_preflight") is True
        and semgrep.get("packs_dispatched") == 16
        and semgrep.get("packs_succeeded") == 16
        and semgrep.get("packs_failed") == 0
        and semgrep.get("aggregate_exit_code") == 0
        and semgrep.get("failure_class") is None
        and semgrep.get("sandbox_engagement") == "engaged"
        and semgrep.get("combined_sarif_validation_status") == "full_valid"
        and semgrep.get("combined_inputs_complete") is True
        and semgrep.get("combined_usable") is True
        and semgrep.get("per_pack_full_validation") is True
        and isinstance(diagnostics, list)
        and len(diagnostics) == 16
        and all(
            isinstance(diagnostic, dict)
            and diagnostic.get("raw_exit_code") in (0, 1)
            and diagnostic.get("sarif_validation_status") == "full_valid"
            and diagnostic.get("failure_class") is None
            for diagnostic in diagnostics
        )
        and isinstance(validation, dict)
        and validation.get("combined_full_validation") is True
        and validation.get("per_pack_full_validation") is True
        and validation.get("combined_inputs_complete") is True
        and validation.get("combined_usable") is True
        and validation.get("jsonschema_version_matches") is True
        and validation.get("schema_hash_matches") is True
        and isinstance(contracts, dict)
        and bool(contracts)
        and all(
            isinstance(contract, dict)
            and contract.get("validation") == "structural_contract_passed"
            for contract in contracts.values()
        )
    )
def _qualified_direct_record(
    record: object,
    identity: ExecutionIdentity,
) -> bool:
    """Verify a direct record before permitting the provider canary."""
    if not isinstance(record, dict):
        return False
    execution = record.get("execution")
    retention = record.get("retention")
    return (
        record.get("record_kind") == "minimal_cpp_direct_semgrep_qualification"
        and record.get("status") == "qualified"
        and record.get("immutable") is True
        and record.get("promotable") is False
        and isinstance(execution, dict)
        and execution.get("commit") == identity.commit
        and execution.get("tree") == identity.tree
        and record.get("private_temp_cleanup_complete") is True
        and isinstance(retention, dict)
        and retention.get("raw_output_persisted") is False
        and _direct_scan_passed(record)
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
    parser.add_argument("--semgrep-bin", required=True, metavar="ABSOLUTE_EXECUTABLE")
    parser.add_argument("--record", required=True)
    parser.add_argument("--direct-record", metavar="QUALIFIED_DIRECT_RECORD")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    controller = QualificationController(
        candidate_python=Path(args.candidate_python),
        semgrep_bin=Path(args.semgrep_bin),
        direct_record=Path(args.direct_record) if args.direct_record else None,
    )
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
