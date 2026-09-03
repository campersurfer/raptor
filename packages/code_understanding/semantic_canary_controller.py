"""Process-level deadline controller for the semantic-canary CLI."""

from __future__ import annotations

import argparse
import hashlib
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
_MAX_PROVIDER_TURNS = 2
_MAX_CANARY_TOOL_CALLS = 2
_MAX_COUNTER = _MAX_CANARY_TOOL_CALLS
_MAX_IDENTITY_CHARS = 128
_MAX_IPC_MESSAGE_BYTES = 2_048
_MAX_IPC_BUFFER_BYTES = _MAX_IPC_MESSAGE_BYTES * _MAX_PROGRESS_EVENTS
_MAX_MODEL_CONFIG_BYTES = 4_096
_INNER_ATTESTATION_SCHEMA_VERSION = 4
_PROCESS_ATTESTATION_SCHEMA_VERSION = 5
_MAX_SEMANTIC_FAILURE_REASONS = 16
_MAX_ATTESTATION_SECTION_COUNT = 1_024
_ALLOWED_LIFECYCLE_EVENTS = frozenset(
    {
        "provider_turn_started",
        "provider_turn_completed",
        "provider_turn_failed",
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
    return value if isinstance(value, str) and value in _ALLOWED_FAILURE_CLASSES else "internal"

_PROGRESS_COUNTER_KEYS = (
    "provider_turns_started",
    "provider_turns_completed",
    "provider_turns_failed",
    "tool_calls_dispatched",
    "tool_calls_completed",
    "fixture_read_calls_dispatched",
    "fixture_read_calls_completed",
    "terminal_call_count",
)
_PROGRESS_KEYS = frozenset((*_PROGRESS_COUNTER_KEYS, "fixture_read_verified"))
_PROVIDER_FAILURE_ORIGINS = frozenset({
    "local_request_build", "worker_uds_connect", "dispatcher_token_auth",
    "dispatcher_provider_config", "dispatcher_upstream_connect", "upstream_http",
    "dispatcher_response_stream", "sdk_response_decode", "provider_empty_response",
    "unknown",
})
_PROVIDER_FAILURE_CATEGORIES = frozenset({
    "auth", "quota", "schema", "timeout", "upstream_4xx", "upstream_5xx",
    "connection_refused", "connection_reset", "protocol", "response_decode",
    "empty_response", "transport_unknown",
})


def _default_progress() -> dict[str, Any]:
    return {**{key: 0 for key in _PROGRESS_COUNTER_KEYS}, "fixture_read_verified": False}


def _parse_progress(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != _PROGRESS_KEYS:
        return None
    if any(type(value[key]) is not int for key in _PROGRESS_COUNTER_KEYS):
        return None
    if type(value["fixture_read_verified"]) is not bool:
        return None
    progress = dict(value)
    if (
        not 0 <= progress["provider_turns_started"] <= _MAX_PROVIDER_TURNS
        or not 0 <= progress["provider_turns_completed"] <= progress["provider_turns_started"]
        or not 0 <= progress["provider_turns_failed"] <= progress["provider_turns_started"]
        or progress["provider_turns_completed"] + progress["provider_turns_failed"] > progress["provider_turns_started"]
        or not 0 <= progress["tool_calls_dispatched"] <= _MAX_CANARY_TOOL_CALLS
        or not 0 <= progress["tool_calls_completed"] <= progress["tool_calls_dispatched"]
        or not 0 <= progress["fixture_read_calls_dispatched"] <= 1
        or not 0 <= progress["fixture_read_calls_completed"] <= progress["fixture_read_calls_dispatched"]
        or progress["fixture_read_calls_dispatched"] > progress["tool_calls_dispatched"]
        or progress["fixture_read_calls_completed"] > progress["tool_calls_completed"]
        or not 0 <= progress["terminal_call_count"] <= 1
        or (progress["fixture_read_verified"] and progress["fixture_read_calls_completed"] != 1)
    ):
        return None
    return progress


def _parse_provider_failure(value: Any) -> dict[str, Any] | None:
    fields = (
        "origin", "category", "turn_ordinal", "http_status_code",
        "exception_type", "retryable", "failure_fingerprint_sha256",
    )
    if not isinstance(value, dict) or set(value) != set(fields):
        return None
    origin = value["origin"]
    category = value["category"]
    status = value["http_status_code"]
    exception_type = value["exception_type"]
    fingerprint = value["failure_fingerprint_sha256"]
    if (
        not isinstance(origin, str)
        or origin not in _PROVIDER_FAILURE_ORIGINS
        or not isinstance(category, str)
        or category not in _PROVIDER_FAILURE_CATEGORIES
        or type(value["turn_ordinal"]) is not int
        or not 1 <= value["turn_ordinal"] <= _MAX_PROVIDER_TURNS
        or status is not None
        and (type(status) is not int or not 100 <= status <= 599)
        or not isinstance(exception_type, str)
        or not 1 <= len(exception_type) <= _MAX_IDENTITY_CHARS
        or type(value["retryable"]) is not bool
        or not isinstance(fingerprint, str)
    ):
        return None
    fingerprint_fields = {key: value[key] for key in fields[:-1]}
    expected = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected:
        return None
    return {key: value[key] for key in fields}


def _reconcile_progress(
    observed: dict[str, Any], inner: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    if _parse_progress(observed) is None or _parse_progress(inner) is None:
        return None, False
    if any(inner[key] != observed[key] for key in _PROGRESS_COUNTER_KEYS):
        return None, False
    if inner["fixture_read_verified"] is not observed["fixture_read_verified"]:
        return None, False
    return dict(observed), True


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
    progress: dict[str, Any] | None = None,
    inner_attestation_state: str = "missing",
    provider_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_progress = _parse_progress(
        progress
        if progress is not None
        else {
            "provider_turns_started": provider_turns_started,
            "provider_turns_completed": provider_turns_completed,
            "provider_turns_failed": 0,
            "tool_calls_dispatched": 0,
            "tool_calls_completed": 0,
            "fixture_read_calls_dispatched": int(fixture_read),
            "fixture_read_calls_completed": int(fixture_read),
            "fixture_read_verified": bool(fixture_read),
            "terminal_call_count": terminal_call_count,
        }
    )
    if parsed_progress is None:
        raise ValueError("semantic canary progress was malformed")
    inner_state = (
        inner_attestation_state
        if isinstance(inner_attestation_state, str) and inner_attestation_state in {"validated", "missing", "invalid", "incomplete"}
        else "invalid"
    )
    attestation: dict[str, Any] = {
        "schema_version": _PROCESS_ATTESTATION_SCHEMA_VERSION,
        "status": "passed" if status == "passed" else "failed",
        "provider": _bounded_identity(model.provider),
        "model": _bounded_identity(model.model_name),
        "sdk_version": _bounded_identity(_sdk_version()),
        "system_instruction_sha256": _bounded_sha256(system_instruction_sha256),
        "request_schema_sha256": _bounded_sha256(request_schema_sha256),
        **parsed_progress,
        "provider_turn_count": parsed_progress["provider_turns_started"],
        "fixture_read": parsed_progress["fixture_read_verified"],
        "inner_attestation_state": inner_state,
        "provider_failure": _parse_provider_failure(provider_failure),
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
    result: Any,
    progress_or_provider_turns_started: dict[str, Any] | int,
    provider_turns_completed: int | None = None,
) -> dict[str, Any]:
    raw = result.attestation if isinstance(getattr(result, "attestation", None), dict) else {}
    if isinstance(progress_or_provider_turns_started, dict):
        observed_progress = _parse_progress(progress_or_provider_turns_started)
    else:
        observed_progress = _parse_progress({
            "provider_turns_started": progress_or_provider_turns_started,
            "provider_turns_completed": provider_turns_completed,
            "provider_turns_failed": 0,
            "tool_calls_dispatched": 0,
            "tool_calls_completed": 0,
            "fixture_read_calls_dispatched": 0,
            "fixture_read_calls_completed": 0,
            "fixture_read_verified": False,
            "terminal_call_count": 0,
        })
    schema_version = raw.get("schema_version")
    if not raw or "schema_version" not in raw:
        inner_state = "missing"
    elif type(schema_version) is not int or schema_version not in (2, 3, _INNER_ATTESTATION_SCHEMA_VERSION):
        inner_state = "invalid"
    elif schema_version in (2, 3):
        inner_state = "incomplete"
    else:
        inner_progress = _parse_progress({key: raw.get(key) for key in _PROGRESS_KEYS})
        required_fields = {
            "schema_version", "status", "fixture_sha256", "provider", "model",
            "sdk_version", "system_instruction_sha256", "request_schema_sha256",
            "provider_turn_count", "fixture_read", "language_verified",
            "source_verified", "sink_verified", "semantic_relation_verified",
            "semantic_checks", "semantic_failure_reasons", "section_counts",
            "provider_failure", *_PROGRESS_KEYS,
        }
        if raw.get("status") == "failed":
            required_fields.add("failure_class")
        required_valid = (
            set(raw) == required_fields
            and isinstance(raw.get("status"), str) and raw.get("status") in {"passed", "failed"}
            and bool(getattr(result, "success", False)) == (raw.get("status") == "passed")
            and _bounded_sha256(raw.get("fixture_sha256")) is not None
            and all(isinstance(raw.get(key), str) and len(raw[key]) <= _MAX_IDENTITY_CHARS for key in ("provider", "model", "sdk_version"))
            and _bounded_sha256(raw.get("system_instruction_sha256")) is not None
            and _bounded_sha256(raw.get("request_schema_sha256")) is not None
            and _parse_section_counts(raw.get("section_counts")) is not None
            and _parse_semantic_checks(raw.get("semantic_checks")) is not None
            and _parse_semantic_failure_reasons(raw.get("semantic_failure_reasons")) is not None
            and all(type(raw.get(key)) is bool for key in (
                "fixture_read", "source_verified", "sink_verified",
                "language_verified", "semantic_relation_verified",
            ))
            and type(raw.get("provider_turn_count")) is int
            and raw.get("provider_turn_count") == raw.get("provider_turns_started")
            and raw.get("fixture_read") == raw.get("fixture_read_verified")
            and (
                raw.get("provider_failure") is None
                or _parse_provider_failure(raw.get("provider_failure")) is not None
            )
            and (
                raw.get("status") == "passed"
                or isinstance(raw.get("failure_class"), str)
                and raw.get("failure_class") in _ALLOWED_FAILURE_CLASSES
            )
            and inner_progress is not None
            and observed_progress is not None
            and inner_progress == observed_progress
        )
        inner_state = "validated" if required_valid else "incomplete"
    inner_progress = (
        _parse_progress({key: raw.get(key) for key in _PROGRESS_KEYS})
        if inner_state == "validated"
        else None
    )
    return {
        "inner_schema_version": (
            schema_version if inner_state != "invalid" and type(schema_version) is int else None
        ),
        "inner_attestation_state": inner_state,
        "status": raw.get("status") if isinstance(raw.get("status"), str) and raw.get("status") in {"passed", "failed"} else "failed",
        "system_instruction_sha256": _bounded_sha256(raw.get("system_instruction_sha256")),
        "request_schema_sha256": _bounded_sha256(raw.get("request_schema_sha256")),
        "section_counts": _parse_section_counts(raw.get("section_counts")),
        "progress": inner_progress,
        "fixture_read": raw.get("fixture_read") is True,
        "source_verified": raw.get("source_verified") is True,
        "sink_verified": raw.get("sink_verified") is True,
        "language_verified": raw.get("language_verified") is True,
        "semantic_relation_verified": raw.get("semantic_relation_verified") is True,
        "failure_class": _bounded_failure_class(raw.get("failure_class")),
        "provider_failure": _parse_provider_failure(raw.get("provider_failure")),
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
    progress = _default_progress()
    event_sequence = 0

    def lifecycle(event: str, **metadata: Any) -> None:
        nonlocal event_sequence
        if not isinstance(event, str) or event not in _ALLOWED_LIFECYCLE_EVENTS:
            raise RuntimeError("unsupported semantic canary lifecycle event")
        next_progress = dict(progress)
        message: dict[str, Any] = {
            "type": "event", "event": event, "event_sequence": event_sequence + 1,
        }
        if event == "provider_turn_started":
            first_turn = progress["provider_turns_started"] == 0
            next_progress["provider_turns_started"] += 1
            if first_turn:
                request_schema_sha256 = _bounded_sha256(metadata.get("request_schema_sha256"))
                if request_schema_sha256 is None:
                    raise RuntimeError("first provider turn omitted request schema")
                message["request_schema_sha256"] = request_schema_sha256
        elif event == "provider_turn_completed":
            next_progress["provider_turns_completed"] += 1
        elif event == "provider_turn_failed":
            next_progress["provider_turns_failed"] += 1
            provider_failure = _parse_provider_failure(metadata.get("provider_failure"))
            if provider_failure is None:
                raise RuntimeError("provider failure event was malformed")
            message["provider_failure"] = provider_failure
        elif event == "tool_call_dispatched":
            next_progress["tool_calls_dispatched"] += 1
            fixture_read_call = metadata.get("fixture_read_call") is True
            if fixture_read_call:
                next_progress["fixture_read_calls_dispatched"] += 1
            message["fixture_read_call"] = fixture_read_call
        elif event == "tool_call_completed":
            next_progress["tool_calls_completed"] += 1
            fixture_read_call = metadata.get("fixture_read_call") is True
            fixture_read_verified = metadata.get("fixture_read_verified") is True
            if fixture_read_verified and not fixture_read_call:
                raise RuntimeError("verified fixture read omitted completed call")
            next_progress["fixture_read_calls_completed"] += int(fixture_read_call)
            if fixture_read_verified:
                next_progress["fixture_read_verified"] = True
            message["fixture_read_call"] = fixture_read_call
            message["fixture_read_verified"] = fixture_read_verified
        elif event == "terminal_call_dispatched":
            next_progress["terminal_call_count"] += 1
        parsed_progress = _parse_progress(next_progress)
        if parsed_progress is None or not _send(write_fd, message):
            raise RuntimeError("semantic canary lifecycle progress was invalid")
        progress.update(parsed_progress)
        event_sequence += 1

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
        attestation = _worker_attestation(result, progress)
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
    except PermissionError:
        try:
            os.kill(worker_pid, signal_number)
        except (ProcessLookupError, PermissionError):
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
    if isinstance(failure_class, str) and failure_class in {"provider_init", "model_resolution", "unsupported_model", "worker_start"}:
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


def _apply_process_event(state: dict[str, Any], message: dict[str, Any]) -> bool:
    event = message.get("event")
    sequence = message.get("event_sequence")
    if (
        not isinstance(event, str) or event not in _ALLOWED_LIFECYCLE_EVENTS
        or type(sequence) is not int
        or sequence != state["last_event_sequence"] + 1
        or state["events"] >= _MAX_PROGRESS_EVENTS
    ):
        return False
    progress = dict(state["progress"])
    expected = {"type", "event", "event_sequence"}
    if event == "provider_turn_started":
        expected |= {"request_schema_sha256"} if progress["provider_turns_started"] == 0 else set()
        if set(message) != expected:
            return False
        request_schema_sha256 = message.get("request_schema_sha256")
        if progress["provider_turns_started"] == 0:
            if _bounded_sha256(request_schema_sha256) is None:
                return False
            state["request_schema_sha256"] = request_schema_sha256
        progress["provider_turns_started"] += 1
    elif event == "provider_turn_completed":
        if set(message) != expected:
            return False
        progress["provider_turns_completed"] += 1
    elif event == "provider_turn_failed":
        if set(message) != expected | {"provider_failure"}:
            return False
        failure = _parse_provider_failure(message.get("provider_failure"))
        if failure is None or failure["turn_ordinal"] != progress["provider_turns_started"]:
            return False
        progress["provider_turns_failed"] += 1
        state["provider_failure"] = failure
    elif event == "tool_call_dispatched":
        if set(message) != expected | {"fixture_read_call"} or type(message.get("fixture_read_call")) is not bool:
            return False
        progress["tool_calls_dispatched"] += 1
        progress["fixture_read_calls_dispatched"] += int(message["fixture_read_call"])
    elif event == "tool_call_completed":
        if (
            set(message)
            != expected | {"fixture_read_call", "fixture_read_verified"}
            or type(message.get("fixture_read_call")) is not bool
            or type(message.get("fixture_read_verified")) is not bool
            or message["fixture_read_verified"] and not message["fixture_read_call"]
        ):
            return False
        progress["tool_calls_completed"] += 1
        progress["fixture_read_calls_completed"] += int(
            message["fixture_read_call"]
        )
        if message["fixture_read_verified"]:
            progress["fixture_read_verified"] = True
    elif event == "terminal_call_dispatched":
        if set(message) != expected:
            return False
        progress["terminal_call_count"] += 1
    else:
        return False
    parsed_progress = _parse_progress(progress)
    if parsed_progress is None:
        return False
    state["progress"] = parsed_progress
    state["events"] += 1
    state["last_event_sequence"] = sequence
    return True


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
        "progress": _default_progress(),
        "request_schema_sha256": None,
        "provider_failure": None,
        "semantic_checks": None,
        "semantic_failure_reasons": None,
        "semantic_checks_received": False,
        "semantic_failure_reasons_received": False,
        "events": 0,
        "last_event_sequence": 0,
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
                            if not _apply_process_event(state, message):
                                worker_lost = True
                                break
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

            state_semantic_checks = _safe_semantic_checks(state["semantic_checks"])
            state_semantic_failure_reasons = _safe_semantic_failure_reasons(
                state["semantic_failure_reasons"]
            )
            progress = state["progress"]
            common = {
                "model": canary_model,
                "request_schema_sha256": state["request_schema_sha256"],
                "provider_turns_started": progress["provider_turns_started"],
                "provider_turns_completed": progress["provider_turns_completed"],
                "terminal_call_count": progress["terminal_call_count"],
                "fixture_read": progress["fixture_read_verified"],
                "semantic_checks": state_semantic_checks,
                "semantic_failure_reasons": state_semantic_failure_reasons,
                "progress": progress,
                "provider_failure": state["provider_failure"],
            }
            if timeout_hit:
                cleanup_verified = _terminate_worker(worker_pid, worker_pgid, cleanup_deadline)
                attestation = _controller_attestation(
                    **common,
                    status="failed",
                    source_verified=False,
                    sink_verified=False,
                    language_verified=False,
                    semantic_relation_verified=False,
                    worker_cleanup_verified=cleanup_verified,
                    failure_class="timeout",
                    failure_stage=_failure_stage(
                        failure_class="timeout",
                        provider_turns_started=progress["provider_turns_started"],
                        provider_turns_completed=progress["provider_turns_completed"],
                        terminal_call_count=progress["terminal_call_count"],
                        fixture_read=progress["fixture_read_verified"],
                    ),
                )
            else:
                cleanup_verified = _confirm_worker_cleanup(worker_pid, worker_pgid, cleanup_deadline)
                if worker_lost or worker_attestation is None:
                    failure_class = "worker_lost" if cleanup_verified else "worker_cleanup"
                    attestation = _controller_attestation(
                        **common,
                        status="failed",
                        source_verified=False,
                        sink_verified=False,
                        language_verified=False,
                        semantic_relation_verified=False,
                        worker_cleanup_verified=cleanup_verified,
                        failure_class=failure_class,
                        failure_stage=_failure_stage(
                            failure_class=failure_class,
                            provider_turns_started=progress["provider_turns_started"],
                            provider_turns_completed=progress["provider_turns_completed"],
                            terminal_call_count=progress["terminal_call_count"],
                            fixture_read=progress["fixture_read_verified"],
                        ),
                    )
                else:
                    inner_state = worker_attestation.get("inner_attestation_state")
                    inner_progress = worker_attestation.get("progress")
                    reconciled_progress, monotonic = (
                        _reconcile_progress(progress, inner_progress)
                        if isinstance(inner_progress, dict)
                        else (None, False)
                    )
                    system_instruction_sha256 = _bounded_sha256(
                        worker_attestation.get("system_instruction_sha256")
                    )
                    request_schema_sha256 = _bounded_sha256(
                        worker_attestation.get("request_schema_sha256")
                    )
                    section_counts = _parse_section_counts(worker_attestation.get("section_counts"))
                    semantic_checks = _parse_semantic_checks(state["semantic_checks"])
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
                    provider_failure = _parse_provider_failure(
                        worker_attestation.get("provider_failure")
                    )
                    if not cleanup_verified:
                        failure_class = "worker_cleanup"
                    elif inner_state != "validated" or not monotonic or reconciled_progress is None:
                        failure_class = "internal"
                    elif (
                        system_instruction_sha256 is None
                        or request_schema_sha256 is None
                        or request_schema_sha256 != state["request_schema_sha256"]
                        or (
                            provider_failure != state["provider_failure"]
                            and not (
                                state["provider_failure"] is None
                                and provider_failure is not None
                                and progress["provider_turns_started"] == 0
                            )
                        )
                    ):
                        failure_class = "internal"
                    elif worker_attestation.get("status") != "passed":
                        failure_class = _bounded_failure_class(worker_attestation.get("failure_class"))
                    elif provider_failure is not None:
                        failure_class = "internal"
                    else:
                        failure_class = _canonical_success_failure(
                            provider_turns_started=reconciled_progress["provider_turns_started"],
                            provider_turns_completed=reconciled_progress["provider_turns_completed"],
                            terminal_call_count=reconciled_progress["terminal_call_count"],
                            fixture_read=reconciled_progress["fixture_read_verified"],
                            source_verified=worker_attestation.get("source_verified") is True,
                            sink_verified=worker_attestation.get("sink_verified") is True,
                            language_verified=worker_attestation.get("language_verified") is True,
                            semantic_relation_verified=worker_attestation.get("semantic_relation_verified") is True,
                            diagnostics_valid=diagnostics_valid,
                            semantic_checks=state_semantic_checks,
                            semantic_failure_reasons=state_semantic_failure_reasons,
                        )
                    output_progress = reconciled_progress or progress
                    output_failure = (
                        state["provider_failure"] or provider_failure
                        if failure_class != "internal" else None
                    )
                    attestation = _controller_attestation(
                        **(common | {
                            "request_schema_sha256": request_schema_sha256,
                            "provider_turns_started": output_progress["provider_turns_started"],
                            "provider_turns_completed": output_progress["provider_turns_completed"],
                            "terminal_call_count": output_progress["terminal_call_count"],
                            "fixture_read": output_progress["fixture_read_verified"],
                            "progress": output_progress,
                            "provider_failure": output_failure,
                        }),
                        status="passed" if failure_class is None else "failed",
                        system_instruction_sha256=system_instruction_sha256,
                        source_verified=worker_attestation.get("source_verified") is True,
                        sink_verified=worker_attestation.get("sink_verified") is True,
                        language_verified=worker_attestation.get("language_verified") is True,
                        semantic_relation_verified=worker_attestation.get("semantic_relation_verified") is True,
                        section_counts=section_counts,
                        worker_cleanup_verified=cleanup_verified,
                        inner_attestation_state=inner_state,
                        failure_class=failure_class,
                        failure_stage=(
                            None if failure_class is None else _failure_stage(
                                failure_class=failure_class,
                                provider_turns_started=output_progress["provider_turns_started"],
                                provider_turns_completed=output_progress["provider_turns_completed"],
                                terminal_call_count=output_progress["terminal_call_count"],
                                fixture_read=output_progress["fixture_read_verified"],
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
