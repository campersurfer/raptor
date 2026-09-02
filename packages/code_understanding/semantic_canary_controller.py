"""Process-level deadline controller for the semantic-canary CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from core.llm.config import ModelConfig
SECTION_COUNT_KEYS = (
    "sources",
    "sinks",
    "trust_boundaries",
    "entry_points",
    "sink_details",
    "boundary_details",
    "unchecked_flows",
)
SEMANTIC_CHECK_KEYS = (
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
SEMANTIC_FAILURE_REASONS = frozenset({
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

_CANARY_HARD_DEADLINE_S = 135.0
_CANARY_PROVIDER_TIMEOUT_S = 60.0
_CLEANUP_RESERVE_S = 5.0
_MIN_CLEANUP_RESERVE_S = 0.2
_MAX_PROGRESS_EVENTS = 16
_MAX_COUNTER = 3
_MAX_IDENTITY_CHARS = 128
_MAX_IPC_MESSAGE_BYTES = 1_024
_MAX_IPC_BUFFER_BYTES = _MAX_IPC_MESSAGE_BYTES * _MAX_PROGRESS_EVENTS
_MAX_MODEL_CONFIG_BYTES = 4_096
_INNER_ATTESTATION_SCHEMA_VERSION = 3
_PROCESS_ATTESTATION_SCHEMA_VERSION = 4
_MAX_SEMANTIC_FAILURE_REASONS = 16
_MAX_ATTESTATION_SECTION_COUNT = 1_024
_ALLOWED_LIFECYCLE_EVENTS = frozenset(
    {
        "provider_turn_started",
        "provider_turn_completed",
        "tool_call_dispatched",
        "tool_call_completed",
        "terminal_call_dispatched",
        "controller_timeout",
        "worker_terminated",
    }
)
_ALLOWED_FAILURE_CLASSES = frozenset(
    {
        "auth",
        "cost_limit",
        "internal",
        "invalid_model",
        "local_dependency_missing",
        "local_schema_invalid",
        "map_validation",
        "model_resolution",
        "provider_init",
        "quota",
        "schema",
        "semantic_evidence",
        "terminal_contract",
        "timeout",
        "transport",
        "unsupported_model",
        "worker_cleanup",
        "worker_lost",
        "worker_start",
    }
)


@dataclass(frozen=True)
class _ControllerOutcome:
    attestation: dict[str, Any]
    worker_pgid: int | None


def _bounded_identity(value: Any) -> str:
    return str(value)[:_MAX_IDENTITY_CHARS]


def _sdk_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "unavailable"


def _bounded_sha256(value: Any) -> str | None:
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def _bounded_counter(value: Any) -> int:
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_COUNTER
        else 0
    )


def _bounded_failure_class(value: Any) -> str:
    return value if value in _ALLOWED_FAILURE_CLASSES else "internal"


def _default_semantic_checks() -> dict[str, bool]:
    return {key: False for key in SEMANTIC_CHECK_KEYS}


def _parse_semantic_checks(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, dict) or set(value) != set(SEMANTIC_CHECK_KEYS):
        return None
    if any(type(value[key]) is not bool for key in SEMANTIC_CHECK_KEYS):
        return None
    return {key: value[key] for key in SEMANTIC_CHECK_KEYS}


def _parse_semantic_failure_reasons(value: Any) -> list[str] | None:
    if not isinstance(value, list) or len(value) > _MAX_SEMANTIC_FAILURE_REASONS:
        return None
    if any(not isinstance(reason, str) or reason not in SEMANTIC_FAILURE_REASONS for reason in value):
        return None
    if value != sorted(set(value)):
        return None
    return list(value)


def _parse_section_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) != set(SECTION_COUNT_KEYS):
        return None
    if any(
        type(value[key]) is not int
        or not 0 <= value[key] <= _MAX_ATTESTATION_SECTION_COUNT
        for key in SECTION_COUNT_KEYS
    ):
        return None
    return {key: value[key] for key in SECTION_COUNT_KEYS}


def _safe_semantic_checks(value: Any) -> dict[str, bool]:
    return _parse_semantic_checks(value) or _default_semantic_checks()


def _safe_semantic_failure_reasons(value: Any) -> list[str]:
    return _parse_semantic_failure_reasons(value) or []


def _safe_section_counts(value: Any) -> dict[str, int]:
    return _parse_section_counts(value) or {key: 0 for key in SECTION_COUNT_KEYS}


def _controller_attestation(
    *,
    model: ModelConfig,
    status: str,
    request_schema_sha256: str | None,
    provider_turns_started: int,
    provider_turns_completed: int,
    terminal_call_count: int,
    fixture_read: bool,
    source_verified: bool,
    sink_verified: bool,
    language_verified: bool,
    semantic_relation_verified: bool,
    worker_cleanup_verified: bool,
    system_instruction_sha256: str | None = None,
    semantic_checks: dict[str, bool] | None = None,
    semantic_failure_reasons: list[str] | None = None,
    section_counts: dict[str, int] | None = None,
    failure_class: str | None = None,
    failure_stage: str | None = None,
) -> dict[str, Any]:
    attestation: dict[str, Any] = {
        "schema_version": _PROCESS_ATTESTATION_SCHEMA_VERSION,
        "status": status,
        "provider": _bounded_identity(model.provider),
        "model": _bounded_identity(model.model_name),
        "sdk_version": _bounded_identity(_sdk_version()),
        "system_instruction_sha256": _bounded_sha256(system_instruction_sha256),
        "request_schema_sha256": _bounded_sha256(request_schema_sha256),
        "provider_turns_started": _bounded_counter(provider_turns_started),
        "provider_turns_completed": _bounded_counter(provider_turns_completed),
        "terminal_call_count": _bounded_counter(terminal_call_count),
        "fixture_read": bool(fixture_read),
        "source_verified": bool(source_verified),
        "sink_verified": bool(sink_verified),
        "language_verified": bool(language_verified),
        "semantic_relation_verified": bool(semantic_relation_verified),
        "semantic_checks": _safe_semantic_checks(semantic_checks),
        "semantic_failure_reasons": _safe_semantic_failure_reasons(
            semantic_failure_reasons
        ),
        "section_counts": _safe_section_counts(section_counts),
        "worker_cleanup_verified": bool(worker_cleanup_verified),
    }
    if failure_class is not None:
        attestation["failure_class"] = _bounded_failure_class(failure_class)
        attestation["failure_stage"] = failure_stage or "provider_init"
    return attestation


def _redirect_worker_output() -> None:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())


def _encode_ipc_message(message: dict[str, Any]) -> bytes | None:
    try:
        return json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _send(write_fd: int, message: dict[str, Any]) -> bool:
    encoded = _encode_ipc_message(message)
    if encoded is None or len(encoded) >= _MAX_IPC_MESSAGE_BYTES:
        return False
    try:
        _write_all(write_fd, encoded + b"\n")
    except (BrokenPipeError, OSError):
        return False
    return True

def _worker_diagnostic_messages(raw: Any) -> list[dict[str, Any]]:
    attestation = raw if isinstance(raw, dict) else {}
    return [
        {
            "type": "diagnostic",
            "semantic_checks": _parse_semantic_checks(
                attestation.get("semantic_checks")
            ),
        },
        {
            "type": "diagnostic",
            "semantic_failure_reasons": _parse_semantic_failure_reasons(
                attestation.get("semantic_failure_reasons")
            ),
        },
    ]


def _worker_attestation(
    result: Any, provider_turns_started: int, provider_turns_completed: int
) -> dict[str, Any]:
    raw = result.attestation if isinstance(result.attestation, dict) else {}
    return {
        "inner_schema_version": (
            raw.get("schema_version")
            if raw.get("schema_version") == _INNER_ATTESTATION_SCHEMA_VERSION
            else None
        ),
        "status": (
            "passed"
            if result.success and raw.get("status") == "passed"
            else "failed"
        ),
        "system_instruction_sha256": _bounded_sha256(
            raw.get("system_instruction_sha256")
        ),
        "request_schema_sha256": _bounded_sha256(raw.get("request_schema_sha256")),
        "section_counts": _parse_section_counts(raw.get("section_counts")),
        "provider_turns_started": _bounded_counter(provider_turns_started),
        "provider_turns_completed": _bounded_counter(provider_turns_completed),
        "terminal_call_count": _bounded_counter(raw.get("terminal_call_count")),
        "fixture_read": raw.get("fixture_read") is True,
        "source_verified": raw.get("source_verified") is True,
        "sink_verified": raw.get("sink_verified") is True,
        "language_verified": raw.get("language_verified") is True,
        "semantic_relation_verified": raw.get("semantic_relation_verified") is True,
        "failure_class": _bounded_failure_class(raw.get("failure_class")),
    }
def _read_worker_model(config_fd: int) -> ModelConfig:
    payload = bytearray()
    try:
        while True:
            chunk = os.read(config_fd, _MAX_MODEL_CONFIG_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _MAX_MODEL_CONFIG_BYTES:
                raise ValueError("semantic canary worker configuration is too large")
    finally:
        os.close(config_fd)
    if not payload:
        raise ValueError("semantic canary worker configuration is missing")
    config = json.loads(payload)
    if not isinstance(config, dict):
        raise ValueError("semantic canary worker configuration is invalid")
    return ModelConfig(**config)


def _write_all(write_fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(write_fd, view)
        if written <= 0:
            raise OSError("zero-length pipe write")
        view = view[written:]

def _spawn_worker(
    model: ModelConfig, fixture_root: Path, write_fd: int
) -> subprocess.Popen[Any]:
    payload = json.dumps(
        asdict(model), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > _MAX_MODEL_CONFIG_BYTES:
        raise ValueError("semantic canary worker configuration is too large")
    config_read_fd, config_write_fd = os.pipe()
    try:
        _write_all(config_write_fd, payload)
    finally:
        os.close(config_write_fd)
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "packages.code_understanding.semantic_canary_controller",
                "--worker",
                "--write-fd",
                str(write_fd),
                "--config-fd",
                str(config_read_fd),
                "--fixture-root",
                str(fixture_root),
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_fd, config_read_fd),
            start_new_session=True,
        )
    finally:
        os.close(config_read_fd)


def _worker_entry(write_fd: int, model: ModelConfig, fixture_root: str) -> None:
    _redirect_worker_output()
    provider_turns_started = 0
    provider_turns_completed = 0

    def lifecycle(event: str, **metadata: Any) -> None:
        nonlocal provider_turns_started, provider_turns_completed
        if event not in _ALLOWED_LIFECYCLE_EVENTS:
            raise RuntimeError("unsupported semantic canary lifecycle event")
        message: dict[str, Any] = {"type": "event", "event": event}
        if event == "provider_turn_started":
            provider_turns_started += 1
            request_schema_sha256 = _bounded_sha256(metadata.get("request_schema_sha256"))
            if request_schema_sha256 is not None:
                message["request_schema_sha256"] = request_schema_sha256
        elif event == "provider_turn_completed":
            provider_turns_completed += 1
        elif event == "tool_call_completed":
            message["fixture_read"] = metadata.get("fixture_read") is True
        _send(write_fd, message)

    try:
        from packages.code_understanding.semantic_canary import run_semantic_canary

        result = run_semantic_canary(
            model,
            fixture_root=Path(fixture_root),
            lifecycle_events=lifecycle,
        )
        diagnostics_sent = all(
            _send(write_fd, message)
            for message in _worker_diagnostic_messages(result.attestation)
        )
        attestation = _worker_attestation(
            result,
            provider_turns_started,
            provider_turns_completed,
        )
        if not diagnostics_sent:
            attestation = {"status": "failed", "failure_class": "internal"}
        _send(write_fd, {"type": "result", "attestation": attestation})
    except Exception:
        _send(write_fd, {"type": "result", "failure_class": "internal"})
    finally:
        os.close(write_fd)
    os._exit(0)


def _worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--write-fd", type=int, required=True)
    parser.add_argument("--config-fd", type=int, required=True)
    parser.add_argument("--fixture-root", required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        return 2
    try:
        model = _read_worker_model(args.config_fd)
    except Exception:
        _send(args.write_fd, {"type": "result", "failure_class": "worker_start"})
        os.close(args.write_fd)
        return 1
    _worker_entry(args.write_fd, model, args.fixture_root)
    return 0


def _cleanup_reserve(total_deadline_s: float) -> float:
    return min(_CLEANUP_RESERVE_S, max(_MIN_CLEANUP_RESERVE_S, total_deadline_s / 8))


def _process_group_is_gone(process_group: int | None) -> bool:
    if process_group is None:
        return True
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _worker_reaped(worker_pid: int) -> bool:
    try:
        reaped_pid, _ = os.waitpid(worker_pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return reaped_pid == worker_pid


def _worker_cleanup_complete(worker_pid: int, process_group: int | None) -> bool:
    return _worker_reaped(worker_pid) and _process_group_is_gone(process_group)


def _wait_for_worker_cleanup(
    worker_pid: int, process_group: int | None, deadline: float
) -> bool:
    while time.monotonic() < deadline:
        if _worker_cleanup_complete(worker_pid, process_group):
            return True
        time.sleep(0.01)
    return _worker_cleanup_complete(worker_pid, process_group)


def _signal_worker(worker_pid: int, process_group: int | None, signal_number: int) -> None:
    try:
        if process_group is not None:
            os.killpg(process_group, signal_number)
        else:
            os.kill(worker_pid, signal_number)
    except ProcessLookupError:
        pass


def _terminate_worker(worker_pid: int, process_group: int | None, deadline: float) -> bool:
    _signal_worker(worker_pid, process_group, signal.SIGTERM)
    grace_deadline = min(deadline, time.monotonic() + 0.25)
    if _wait_for_worker_cleanup(worker_pid, process_group, grace_deadline):
        return True
    _signal_worker(worker_pid, process_group, signal.SIGKILL)
    return _wait_for_worker_cleanup(worker_pid, process_group, deadline)


def _confirm_worker_cleanup(worker_pid: int, process_group: int | None, deadline: float) -> bool:
    if _wait_for_worker_cleanup(worker_pid, process_group, deadline):
        return True
    return _terminate_worker(worker_pid, process_group, deadline)


def _failure_stage(
    *,
    failure_class: str,
    provider_turns_started: int,
    provider_turns_completed: int,
    terminal_call_count: int,
    fixture_read: bool,
) -> str:
    if provider_turns_started > provider_turns_completed:
        return "provider_turn_2" if provider_turns_started >= 2 else "provider_turn_1"
    if failure_class in {"provider_init", "model_resolution", "unsupported_model", "worker_start"}:
        return "provider_init"
    if fixture_read and provider_turns_started == provider_turns_completed == 1:
        return "fixture_read"
    if terminal_call_count or provider_turns_started >= 2:
        return "terminal_validation"
    return "controller_timeout" if failure_class == "timeout" else "provider_init"


def _canonical_success_failure(
    *,
    provider_turns_started: int,
    provider_turns_completed: int,
    terminal_call_count: int,
    fixture_read: bool,
    source_verified: bool,
    sink_verified: bool,
    language_verified: bool,
    semantic_relation_verified: bool,
    diagnostics_valid: bool,
    semantic_checks: dict[str, bool],
    semantic_failure_reasons: list[str],
) -> str | None:
    if not diagnostics_valid:
        return "internal"
    if (
        provider_turns_started != 2
        or provider_turns_completed != 2
        or terminal_call_count != 1
        or not fixture_read
    ):
        return "terminal_contract"
    if not (
        source_verified
        and sink_verified
        and language_verified
        and semantic_relation_verified
        and all(semantic_checks.values())
        and not semantic_failure_reasons
    ):
        return "semantic_evidence"
    return None
def _receive_messages(read_fd: int, buffered: bytes) -> tuple[list[dict[str, Any]], bytes, bool, bool]:
    try:
        chunk = os.read(read_fd, _MAX_IPC_BUFFER_BYTES)
    except BlockingIOError:
        return [], buffered, False, False
    if not chunk:
        return [], buffered, True, False
    buffered += chunk
    if len(buffered) > _MAX_IPC_BUFFER_BYTES:
        return [], b"", False, True
    messages: list[dict[str, Any]] = []
    while b"\n" in buffered:
        line, buffered = buffered.split(b"\n", 1)
        if not line or len(line) >= _MAX_IPC_MESSAGE_BYTES:
            return [], buffered, False, True
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return [], buffered, False, True
        if not isinstance(message, dict):
            return [], buffered, False, True
        messages.append(message)
    return messages, buffered, False, False


def _run_semantic_canary_controller(
    model: ModelConfig,
    *,
    total_deadline_s: float | None = None,
) -> _ControllerOutcome:
    if not isinstance(model, ModelConfig):
        invalid = ModelConfig(provider="gemini", model_name="")
        return _ControllerOutcome(
            _controller_attestation(
                model=invalid,
                status="failed",
                request_schema_sha256=None,
                provider_turns_started=0,
                provider_turns_completed=0,
                terminal_call_count=0,
                fixture_read=False,
                source_verified=False,
                sink_verified=False,
                language_verified=False,
                semantic_relation_verified=False,
                worker_cleanup_verified=True,
                failure_class="invalid_model",
                failure_stage="provider_init",
            ),
            None,
        )

    deadline_s = _CANARY_HARD_DEADLINE_S if total_deadline_s is None else total_deadline_s
    if (
        isinstance(deadline_s, bool)
        or not isinstance(deadline_s, (int, float))
        or not math.isfinite(float(deadline_s))
        or deadline_s <= 0
    ):
        raise ValueError("semantic canary deadline must be a positive finite number")
    canary_model = replace(model, timeout=_CANARY_PROVIDER_TIMEOUT_S)
    started_at = time.monotonic()
    cleanup_deadline = started_at + float(deadline_s)
    worker_deadline = cleanup_deadline - _cleanup_reserve(float(deadline_s))
    state: dict[str, Any] = {
        "provider_turns_started": 0,
        "provider_turns_completed": 0,
        "terminal_call_count": 0,
        "fixture_read": False,
        "request_schema_sha256": None,
        "semantic_checks": None,
        "semantic_failure_reasons": None,
        "semantic_checks_received": False,
        "semantic_failure_reasons_received": False,
        "events": 0,
    }
    worker_pgid: int | None = None
    worker_attestation: dict[str, Any] | None = None
    timeout_hit = False
    worker_lost = False

    with tempfile.TemporaryDirectory(prefix="raptor-semantic-canary-") as fixture_root:
        read_fd, write_fd = os.pipe()
        try:
            worker_process = _spawn_worker(canary_model, Path(fixture_root), write_fd)
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            return _ControllerOutcome(
                _controller_attestation(
                    model=canary_model,
                    status="failed",
                    request_schema_sha256=None,
                    provider_turns_started=0,
                    provider_turns_completed=0,
                    terminal_call_count=0,
                    fixture_read=False,
                    source_verified=False,
                    sink_verified=False,
                    language_verified=False,
                    semantic_relation_verified=False,
                    worker_cleanup_verified=True,
                    failure_class="worker_start",
                    failure_stage="provider_init",
                ),
                None,
            )
        worker_pid = worker_process.pid
        worker_pgid = worker_pid
        os.close(write_fd)
        os.set_blocking(read_fd, False)
        buffered = b""
        try:
            while worker_attestation is None and not timeout_hit and not worker_lost:
                now = time.monotonic()
                if now >= worker_deadline:
                    timeout_hit = True
                    state["events"] += 1  # controller_timeout
                    break
                ready, _, _ = select.select([read_fd], [], [], min(0.05, worker_deadline - now))
                if ready:
                    messages, buffered, eof, invalid = _receive_messages(read_fd, buffered)
                    if invalid:
                        worker_lost = True
                        continue
                    for message in messages:
                        if worker_attestation is not None:
                            worker_lost = True
                            break
                        message_type = message.get("type")
                        if message_type == "event":
                            event = message.get("event")
                            if (
                                event not in _ALLOWED_LIFECYCLE_EVENTS
                                or state["events"] >= _MAX_PROGRESS_EVENTS
                            ):
                                worker_lost = True
                                break
                            state["events"] += 1
                            if event == "provider_turn_started":
                                state["provider_turns_started"] += 1
                                state["request_schema_sha256"] = (
                                    _bounded_sha256(message.get("request_schema_sha256"))
                                    or state["request_schema_sha256"]
                                )
                            elif event == "provider_turn_completed":
                                state["provider_turns_completed"] += 1
                            elif event == "terminal_call_dispatched":
                                state["terminal_call_count"] += 1
                            elif (
                                event == "tool_call_completed"
                                and message.get("fixture_read") is True
                            ):
                                state["fixture_read"] = True
                        elif message_type == "diagnostic":
                            if set(message) == {"type", "semantic_checks"}:
                                parsed_checks = _parse_semantic_checks(
                                    message.get("semantic_checks")
                                )
                                if (
                                    state["semantic_checks_received"]
                                    or parsed_checks is None
                                ):
                                    worker_lost = True
                                    break
                                state["semantic_checks"] = parsed_checks
                                state["semantic_checks_received"] = True
                            elif set(message) == {
                                "type",
                                "semantic_failure_reasons",
                            }:
                                parsed_reasons = _parse_semantic_failure_reasons(
                                    message.get("semantic_failure_reasons")
                                )
                                if (
                                    state["semantic_failure_reasons_received"]
                                    or parsed_reasons is None
                                ):
                                    worker_lost = True
                                    break
                                state["semantic_failure_reasons"] = parsed_reasons
                                state["semantic_failure_reasons_received"] = True
                            else:
                                worker_lost = True
                                break
                        elif message_type == "result":
                            if isinstance(message.get("attestation"), dict):
                                worker_attestation = message["attestation"]
                            else:
                                worker_attestation = {
                                    "status": "failed",
                                    "failure_class": _bounded_failure_class(
                                        message.get("failure_class")
                                    ),
                                }
                        else:
                            worker_lost = True
                            break
                    if eof and worker_attestation is None:
                        worker_lost = True
                elif _worker_reaped(worker_pid):
                    worker_lost = True

            state_semantic_checks = _safe_semantic_checks(
                state["semantic_checks"]
            )
            state_semantic_failure_reasons = _safe_semantic_failure_reasons(
                state["semantic_failure_reasons"]
            )
            if timeout_hit:
                cleanup_verified = _terminate_worker(
                    worker_pid, worker_pgid, cleanup_deadline
                )
                failure_class = "timeout"
                attestation = _controller_attestation(
                    model=canary_model,
                    status="failed",
                    request_schema_sha256=state["request_schema_sha256"],
                    provider_turns_started=state["provider_turns_started"],
                    provider_turns_completed=state["provider_turns_completed"],
                    terminal_call_count=state["terminal_call_count"],
                    fixture_read=state["fixture_read"],
                    source_verified=False,
                    sink_verified=False,
                    language_verified=False,
                    semantic_relation_verified=False,
                    semantic_checks=state_semantic_checks,
                    semantic_failure_reasons=state_semantic_failure_reasons,
                    worker_cleanup_verified=cleanup_verified,
                    failure_class=failure_class,
                    failure_stage=_failure_stage(
                        failure_class=failure_class,
                        provider_turns_started=state["provider_turns_started"],
                        provider_turns_completed=state["provider_turns_completed"],
                        terminal_call_count=state["terminal_call_count"],
                        fixture_read=state["fixture_read"],
                    ),
                )
            else:
                cleanup_verified = _confirm_worker_cleanup(
                    worker_pid, worker_pgid, cleanup_deadline
                )
                if worker_lost or worker_attestation is None:
                    failure_class = (
                        "worker_lost" if cleanup_verified else "worker_cleanup"
                    )
                    attestation = _controller_attestation(
                        model=canary_model,
                        status="failed",
                        request_schema_sha256=state["request_schema_sha256"],
                        provider_turns_started=state["provider_turns_started"],
                        provider_turns_completed=state["provider_turns_completed"],
                        terminal_call_count=state["terminal_call_count"],
                        fixture_read=state["fixture_read"],
                        source_verified=False,
                        sink_verified=False,
                        language_verified=False,
                        semantic_relation_verified=False,
                        semantic_checks=state_semantic_checks,
                        semantic_failure_reasons=state_semantic_failure_reasons,
                        worker_cleanup_verified=cleanup_verified,
                        failure_class=failure_class,
                        failure_stage=_failure_stage(
                            failure_class=failure_class,
                            provider_turns_started=state["provider_turns_started"],
                            provider_turns_completed=state["provider_turns_completed"],
                            terminal_call_count=state["terminal_call_count"],
                            fixture_read=state["fixture_read"],
                        ),
                    )
                else:
                    system_instruction_sha256 = _bounded_sha256(
                        worker_attestation.get("system_instruction_sha256")
                    )
                    request_schema_sha256 = _bounded_sha256(
                        worker_attestation.get("request_schema_sha256")
                    )
                    section_counts = _parse_section_counts(
                        worker_attestation.get("section_counts")
                    )
                    semantic_checks = _parse_semantic_checks(
                        state["semantic_checks"]
                    )
                    semantic_failure_reasons = _parse_semantic_failure_reasons(
                        state["semantic_failure_reasons"]
                    )
                    diagnostics_valid = (
                        state["semantic_checks_received"]
                        and state["semantic_failure_reasons_received"]
                        and semantic_checks is not None
                        and semantic_failure_reasons is not None
                        and section_counts is not None
                    )
                    safe_semantic_checks = _safe_semantic_checks(semantic_checks)
                    safe_semantic_failure_reasons = (
                        _safe_semantic_failure_reasons(semantic_failure_reasons)
                    )
                    safe_section_counts = _safe_section_counts(section_counts)
                    provider_turns_started = _bounded_counter(
                        worker_attestation.get("provider_turns_started")
                    )
                    provider_turns_completed = _bounded_counter(
                        worker_attestation.get("provider_turns_completed")
                    )
                    terminal_call_count = _bounded_counter(
                        worker_attestation.get("terminal_call_count")
                    )
                    fixture_read = worker_attestation.get("fixture_read") is True
                    source_verified = worker_attestation.get("source_verified") is True
                    sink_verified = worker_attestation.get("sink_verified") is True
                    language_verified = worker_attestation.get("language_verified") is True
                    semantic_relation_verified = (
                        worker_attestation.get("semantic_relation_verified") is True
                    )
                    inner_schema_valid = (
                        worker_attestation.get("inner_schema_version")
                        == _INNER_ATTESTATION_SCHEMA_VERSION
                    )
                    event_result_consistent = (
                        provider_turns_started == state["provider_turns_started"]
                        and provider_turns_completed == state["provider_turns_completed"]
                        and terminal_call_count == state["terminal_call_count"]
                        and fixture_read == state["fixture_read"]
                        and request_schema_sha256 == state["request_schema_sha256"]
                    )
                    if not cleanup_verified:
                        failure_class = "worker_cleanup"
                    elif not inner_schema_valid:
                        failure_class = "internal"
                    elif worker_attestation.get("status") != "passed":
                        failure_class = _bounded_failure_class(
                            worker_attestation.get("failure_class")
                        )
                    elif (
                        system_instruction_sha256 is None
                        or request_schema_sha256 is None
                        or not event_result_consistent
                    ):
                        failure_class = "internal"
                    else:
                        failure_class = _canonical_success_failure(
                            provider_turns_started=provider_turns_started,
                            provider_turns_completed=provider_turns_completed,
                            terminal_call_count=terminal_call_count,
                            fixture_read=fixture_read,
                            source_verified=source_verified,
                            sink_verified=sink_verified,
                            language_verified=language_verified,
                            semantic_relation_verified=semantic_relation_verified,
                            diagnostics_valid=diagnostics_valid,
                            semantic_checks=safe_semantic_checks,
                            semantic_failure_reasons=safe_semantic_failure_reasons,
                        )
                    attestation = _controller_attestation(
                        model=canary_model,
                        status="passed" if failure_class is None else "failed",
                        system_instruction_sha256=system_instruction_sha256,
                        request_schema_sha256=request_schema_sha256,
                        provider_turns_started=provider_turns_started,
                        provider_turns_completed=provider_turns_completed,
                        terminal_call_count=terminal_call_count,
                        fixture_read=fixture_read,
                        source_verified=source_verified,
                        sink_verified=sink_verified,
                        language_verified=language_verified,
                        semantic_relation_verified=semantic_relation_verified,
                        semantic_checks=safe_semantic_checks,
                        semantic_failure_reasons=safe_semantic_failure_reasons,
                        section_counts=safe_section_counts,
                        worker_cleanup_verified=cleanup_verified,
                        failure_class=failure_class,
                        failure_stage=(
                            None
                            if failure_class is None
                            else _failure_stage(
                                failure_class=failure_class,
                                provider_turns_started=provider_turns_started,
                                provider_turns_completed=provider_turns_completed,
                                terminal_call_count=terminal_call_count,
                                fixture_read=fixture_read,
                            )
                        ),
                    )
        finally:
            if not _worker_cleanup_complete(worker_pid, worker_pgid):
                _terminate_worker(worker_pid, worker_pgid, cleanup_deadline)
            try:
                worker_process.poll()
            except (ChildProcessError, OSError):
                pass
            os.close(read_fd)

    return _ControllerOutcome(attestation=attestation, worker_pgid=worker_pgid)


def failed_semantic_canary_cli_result(model_name: str, failure_class: str):
    """Return the controller's safe CLI schema before a worker is started."""
    from packages.code_understanding.semantic_canary import SemanticCanaryResult

    model = ModelConfig(provider="gemini", model_name=_bounded_identity(model_name))
    bounded_failure = _bounded_failure_class(failure_class)
    return SemanticCanaryResult(
        success=False,
        attestation=_controller_attestation(
            model=model,
            status="failed",
            request_schema_sha256=None,
            provider_turns_started=0,
            provider_turns_completed=0,
            terminal_call_count=0,
            fixture_read=False,
            source_verified=False,
            sink_verified=False,
            language_verified=False,
            semantic_relation_verified=False,
            worker_cleanup_verified=True,
            failure_class=bounded_failure,
            failure_stage=_failure_stage(
                failure_class=bounded_failure,
                provider_turns_started=0,
                provider_turns_completed=0,
                terminal_call_count=0,
                fixture_read=False,
            ),
        ),
    )


def run_semantic_canary_controller(model: ModelConfig):
    """Run the canonical canary in a process that owns its hard deadline."""
    from packages.code_understanding.semantic_canary import SemanticCanaryResult

    outcome = _run_semantic_canary_controller(model)
    return SemanticCanaryResult(
        success=outcome.attestation["status"] == "passed",
        attestation=outcome.attestation,
    )


if __name__ == "__main__":
    raise SystemExit(_worker_main())
