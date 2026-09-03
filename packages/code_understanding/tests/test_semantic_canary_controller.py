"""Subprocess contracts for the bounded semantic-canary controller."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


_RAPTOR_ROOT = Path(__file__).resolve().parents[3]
_RAPTOR_CLI = _RAPTOR_ROOT / "raptor.py"
_TEST_DEADLINE_S = 45.0
_TEST_PROCESS_TIMEOUT_S = 120.0
_TEST_SECRET = "semantic-canary-controller-test-secret"
_TEST_PROMPT = "semantic-canary-controller-test-prompt"
_V5_BASE_ATTESTATION_FIELDS = {
    "schema_version", "status", "provider", "model", "sdk_version",
    "system_instruction_sha256", "request_schema_sha256",
    "provider_turns_started", "provider_turns_completed", "provider_turns_failed",
    "tool_calls_dispatched", "tool_calls_completed",
    "fixture_read_calls_dispatched", "fixture_read_calls_completed",
    "fixture_read_verified", "terminal_call_count",
    "provider_turn_count", "fixture_read", "inner_attestation_state",
    "provider_failure", "source_verified", "sink_verified", "language_verified",
    "semantic_relation_verified", "semantic_checks", "semantic_failure_reasons",
    "section_counts", "worker_cleanup_verified",
}
_V5_FAILURE_ATTESTATION_FIELDS = {"failure_class", "failure_stage"}


def _sitecustomize_source() -> str:
    return textwrap.dedent(
        """
        import json
        import os
        import re
        import time

        import core.llm.providers
        from core.llm.config import ModelConfig
        from core.llm.tool_use.types import StopReason, ToolCall, ToolResult, TurnResponse

        _SCENARIO = os.environ["RAPTOR_CANARY_TEST_SCENARIO"]
        _SECRET = os.environ["RAPTOR_CANARY_TEST_SECRET"]

        try:
            from packages.code_understanding import semantic_canary_controller as _controller
        except ImportError:
            pass
        else:
            _controller._CANARY_HARD_DEADLINE_S = float(
                os.environ["RAPTOR_CANARY_TEST_DEADLINE_S"]
            )

        from packages.code_understanding import semantic_canary as _semantic_canary

        def _response(calls):
            return TurnResponse(
                content=calls,
                stop_reason=StopReason.NEEDS_TOOL_CALL,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
            )

        def _fixture_content(messages):
            for message in messages:
                for block in message.content:
                    if not isinstance(block, ToolResult) or block.tool_use_id != "read":
                        continue
                    start = block.content.find("{")
                    end = block.content.rfind("}")
                    if start < 0 or end < start:
                        continue
                    payload = json.loads(block.content[start : end + 1])
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content
            raise AssertionError("fixture read result was absent")

        def _line_of(content, fragment):
            return content[: content.index(fragment)].count("\\n") + 1

        def _canonical_map(content):
            source = re.search(r"std::string\\s+(canary_source_[0-9a-f]+)", content).group(1)
            sink = re.search(r"void\\s+(canary_sink_[0-9a-f]+)", content).group(1)
            relation = re.search(r"void\\s+(canary_relation_[0-9a-f]+)", content).group(1)
            source_line = _line_of(content, 'return argc > 1 ? argv[1] : "";')
            sink_line = _line_of(content, "std::system(command.c_str());")
            relation_line = _line_of(content, "void " + relation + "(")
            return {
                "sources": [{
                    "type": "cli_arg",
                    "entry": "fixture.cpp:%d %s argv[1]" % (source_line, source),
                    "trust_level": "attacker_controlled",
                }],
                "sinks": [{"type": "shell_exec", "location": "fixture.cpp:%d" % sink_line}],
                "trust_boundaries": [],
                "meta": {"app_type": "cli", "language": ["C++"]},
                "entry_points": [{
                    "id": "EP-1",
                    "type": "cli_arg",
                    "file": "fixture.cpp",
                    "line": relation_line,
                    "name": relation,
                }],
                "sink_details": [{
                    "id": "SINK-1",
                    "type": "shell_exec",
                    "operation": "std::system(command.c_str());",
                    "file": "fixture.cpp",
                    "line": sink_line,
                    "name": sink,
                }],
                "boundary_details": [],
                "unchecked_flows": [{
                    "entry_point": "EP-1",
                    "sink": "SINK-1",
                    "missing_boundary": "unvalidated command argument",
                }],
            }

        class _HttpAcceptedResponse:
            status_code = 200
            headers = {"x-test-token": _SECRET}

            @property
            def body(self):
                time.sleep(3600)
                return "never reached"

        class _Provider:
            def __init__(self):
                self.calls = 0

            def supports_tool_use(self):
                return True

            def supports_prompt_caching(self):
                return False

            def estimate_tokens(self, text):
                return max(1, len(text) // 4)

            def context_window(self):
                return 200000

            def compute_cost(self, response):
                return response.cost_usd or 0.0

            def turn(self, messages, _tools, **_kwargs):
                self.calls += 1
                if _SCENARIO == "block_first":
                    time.sleep(3600)
                if _SCENARIO == "read_then_block":
                    if self.calls == 1:
                        return _response([
                            ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})
                        ])
                    time.sleep(3600)
                if _SCENARIO == "response_body_block":
                    accepted = _HttpAcceptedResponse()
                    assert accepted.status_code == 200
                    _ = accepted.body
                if _SCENARIO == "read_then_raise":
                    if self.calls == 1:
                        return _response([
                            ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})
                        ])
                    raise RuntimeError("synthetic private provider failure")
                if _SCENARIO == "success":
                    if self.calls == 1:
                        return _response([
                            ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})
                        ])
                    return _response([
                        ToolCall(
                            id="terminal",
                            name="submit_context_map",
                            input={"context_map": _canonical_map(_fixture_content(messages))},
                        )
                    ])
                raise AssertionError("unexpected provider scenario")

        _semantic_canary.resolve_semantic_canary_model = lambda _model: ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            timeout=17,
        )
        _semantic_canary.run_semantic_canary.__kwdefaults__["provider_factory"] = (
            lambda _model: _Provider()
        )
        """
    )


def _run_cli(tmp_path: Path, scenario: str) -> subprocess.CompletedProcess[str]:
    injection = tmp_path / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(_sitecustomize_source(), encoding="utf-8")
    environment = os.environ.copy()
    pythonpath = [str(injection), str(_RAPTOR_ROOT)]
    if inherited := environment.get("PYTHONPATH"):
        pythonpath.append(inherited)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "RAPTOR_CANARY_TEST_SCENARIO": scenario,
            "RAPTOR_CANARY_TEST_SECRET": _TEST_SECRET,
            "RAPTOR_CANARY_TEST_DEADLINE_S": str(_TEST_DEADLINE_S),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(_RAPTOR_CLI),
            "semantic-canary",
            "--model",
            "gemini-2.5-flash",
            "--format",
            "json",
        ],
        cwd=_RAPTOR_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_TEST_PROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail(
            f"semantic-canary CLI exceeded the bounded test deadline for {scenario}: {error}"
        )
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _single_attestation(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.stdout.count("\n") == 1
    assert completed.stdout.endswith("\n")
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    expected_fields = _V5_BASE_ATTESTATION_FIELDS | (
        _V5_FAILURE_ATTESTATION_FIELDS if payload.get("status") == "failed" else set()
    )
    assert set(payload) == expected_fields
    return payload

def _assert_no_sensitive_output(completed: subprocess.CompletedProcess[str]) -> None:
    combined = completed.stdout + completed.stderr
    assert _TEST_SECRET not in combined
    assert _TEST_PROMPT not in combined
    assert "std::system(command.c_str())" not in combined
    assert "raw-provider-response" not in combined


@pytest.mark.parametrize(
    ("scenario", "expected_stage", "expected_started", "expected_completed", "fixture_read"),
    [
        ("block_first", "provider_turn_1", 1, 0, False),
        ("read_then_block", "provider_turn_2", 2, 1, True),
        ("response_body_block", "provider_turn_1", 1, 0, False),
    ],
)
def test_semantic_canary_cli_hard_deadline_for_inflight_provider_turn(
    tmp_path: Path,
    scenario: str,
    expected_stage: str,
    expected_started: int,
    expected_completed: int,
    fixture_read: bool,
) -> None:
    started = time.monotonic()
    completed = _run_cli(tmp_path, scenario)
    elapsed = time.monotonic() - started

    assert elapsed < _TEST_PROCESS_TIMEOUT_S
    assert completed.returncode != 0
    attestation = _single_attestation(completed)
    assert attestation["status"] == "failed"
    assert attestation["failure_class"] == "timeout"
    assert attestation["failure_stage"] == expected_stage
    assert attestation["provider_turns_started"] == expected_started
    assert attestation["provider_turns_completed"] == expected_completed
    assert attestation["provider_turns_failed"] == 0
    assert attestation["provider_turn_count"] == expected_started
    assert attestation["tool_calls_dispatched"] == int(fixture_read)
    assert attestation["tool_calls_completed"] == int(fixture_read)
    assert attestation["fixture_read_calls_dispatched"] == int(fixture_read)
    assert attestation["fixture_read_calls_completed"] == int(fixture_read)
    assert attestation["fixture_read_verified"] is fixture_read
    assert attestation["inner_attestation_state"] == "missing"
    assert attestation["provider_failure"] is None
    assert attestation["fixture_read"] is fixture_read
    assert attestation["worker_cleanup_verified"] is True
    _assert_no_sensitive_output(completed)


def test_semantic_canary_cli_success_remains_canonical_and_bounded(tmp_path: Path) -> None:
    from packages.code_understanding.semantic_canary import (
        SECTION_COUNT_KEYS,
        SEMANTIC_CHECK_KEYS,
    )

    completed = _run_cli(tmp_path, "success")

    assert completed.returncode == 0
    attestation = _single_attestation(completed)
    assert attestation["schema_version"] == 5
    assert attestation["inner_attestation_state"] == "validated"
    assert attestation["provider_failure"] is None
    assert attestation["status"] == "passed"
    assert len(attestation["system_instruction_sha256"]) == 64
    assert len(attestation["request_schema_sha256"]) == 64
    assert attestation["provider_turns_started"] == 2
    assert attestation["provider_turns_completed"] == 2
    assert attestation["provider_turns_failed"] == 0
    assert attestation["provider_turn_count"] == 2
    assert attestation["tool_calls_dispatched"] == 2
    assert attestation["tool_calls_completed"] == 2
    assert attestation["fixture_read_calls_dispatched"] == 1
    assert attestation["fixture_read_calls_completed"] == 1
    assert attestation["fixture_read_verified"] is True
    assert attestation["terminal_call_count"] == 1
    assert attestation["fixture_read"] is True
    assert attestation["language_verified"] is True
    assert attestation["source_verified"] is True
    assert attestation["sink_verified"] is True
    assert attestation["semantic_relation_verified"] is True
    assert set(attestation["semantic_checks"]) == set(SEMANTIC_CHECK_KEYS)
    assert all(attestation["semantic_checks"].values())
    assert attestation["semantic_failure_reasons"] == []
    assert set(attestation["section_counts"]) == set(SECTION_COUNT_KEYS)
    assert attestation["worker_cleanup_verified"] is True
    _assert_no_sensitive_output(completed)



def test_semantic_canary_cli_retains_progress_after_second_turn_exception(
    tmp_path: Path,
) -> None:
    completed = _run_cli(tmp_path, "read_then_raise")

    assert completed.returncode != 0
    attestation = _single_attestation(completed)
    assert attestation["schema_version"] == 5
    assert attestation["status"] == "failed"
    assert attestation["failure_class"] == "transport"
    assert attestation["failure_stage"] == "provider_turn_2"
    assert attestation["inner_attestation_state"] == "validated"
    assert attestation["provider_turns_started"] == 2
    assert attestation["provider_turns_completed"] == 1
    assert attestation["provider_turns_failed"] == 1
    assert attestation["tool_calls_dispatched"] == 1
    assert attestation["tool_calls_completed"] == 1
    assert attestation["fixture_read_calls_dispatched"] == 1
    assert attestation["fixture_read_calls_completed"] == 1
    assert attestation["fixture_read_verified"] is True
    assert attestation["provider_failure"]["category"] == "transport_unknown"
    assert attestation["worker_cleanup_verified"] is True
    _assert_no_sensitive_output(completed)

def test_controller_attestation_excludes_api_key() -> None:
    from core.llm.config import ModelConfig
    from packages.code_understanding import semantic_canary_controller

    secret = "semantic-canary-attestation-test-secret"
    attestation = semantic_canary_controller._controller_attestation(
        model=ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key=secret,
        ),
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
        failure_class="provider_init",
        failure_stage="provider_init",
    )

    assert secret not in json.dumps(attestation, sort_keys=True)


def test_controller_requires_all_canonical_evidence() -> None:
    from packages.code_understanding import semantic_canary, semantic_canary_controller

    valid_checks = {key: True for key in semantic_canary.SEMANTIC_CHECK_KEYS}
    canonical = {
        "provider_turns_started": 2,
        "provider_turns_completed": 2,
        "terminal_call_count": 1,
        "fixture_read": True,
        "source_verified": True,
        "sink_verified": True,
        "language_verified": True,
        "semantic_relation_verified": True,
        "diagnostics_valid": True,
        "semantic_checks": valid_checks,
        "semantic_failure_reasons": [],
    }

    assert semantic_canary_controller._canonical_success_failure(
        **{**canonical, "source_verified": False}
    ) == "semantic_evidence"
    assert semantic_canary_controller._canonical_success_failure(
        **{
            **canonical,
            "provider_turns_started": 1,
            "provider_turns_completed": 1,
        }
    ) == "terminal_contract"
    assert semantic_canary_controller._canonical_success_failure(
        **{**canonical, "diagnostics_valid": False}
    ) == "internal"


def test_controller_protocol_constants_match_inner_canary_contract() -> None:
    from packages.code_understanding import semantic_canary, semantic_canary_controller

    assert semantic_canary_controller.SECTION_COUNT_KEYS == semantic_canary.SECTION_COUNT_KEYS
    assert semantic_canary_controller.SEMANTIC_CHECK_KEYS == semantic_canary.SEMANTIC_CHECK_KEYS
    assert semantic_canary_controller.SEMANTIC_FAILURE_REASONS == semantic_canary.SEMANTIC_FAILURE_REASONS


def test_controller_sends_bounded_diagnostics_in_separate_ipc_messages() -> None:
    from packages.code_understanding import semantic_canary, semantic_canary_controller

    messages = semantic_canary_controller._worker_diagnostic_messages({
        "semantic_checks": {
            key: True for key in semantic_canary.SEMANTIC_CHECK_KEYS
        },
        "semantic_failure_reasons": sorted(
            semantic_canary.SEMANTIC_FAILURE_REASONS
        ),
        "untrusted_extra": _TEST_SECRET,
    })

    assert [set(message) for message in messages] == [
        {"type", "semantic_checks"},
        {"type", "semantic_failure_reasons"},
    ]
    for message in messages:
        encoded = semantic_canary_controller._encode_ipc_message(message)
        assert encoded is not None
        assert len(encoded) < semantic_canary_controller._MAX_IPC_MESSAGE_BYTES
        assert _TEST_SECRET not in encoded.decode("utf-8")


def test_controller_completes_short_ipc_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    from packages.code_understanding import semantic_canary, semantic_canary_controller

    message = {
        "type": "diagnostic",
        "semantic_checks": {
            key: True for key in semantic_canary.SEMANTIC_CHECK_KEYS
        },
    }
    encoded = semantic_canary_controller._encode_ipc_message(message)
    assert encoded is not None
    assert len(encoded) > 512
    writes: list[bytes] = []

    def short_write(_fd: int, payload: bytes) -> int:
        chunk = bytes(payload)[:17]
        writes.append(chunk)
        return len(chunk)

    monkeypatch.setattr(semantic_canary_controller.os, "write", short_write)

    assert semantic_canary_controller._send(123, message)
    assert b"".join(writes) == encoded + b"\n"
    assert len(writes) > 1

    monkeypatch.setattr(semantic_canary_controller.os, "write", lambda *_: 0)
    assert not semantic_canary_controller._send(123, message)


def test_worker_attestation_marks_historical_v3_inner_schema_incomplete() -> None:
    from packages.code_understanding import semantic_canary_controller

    stale_result = type(
        "StaleResult",
        (),
        {
            "success": True,
            "attestation": {
                "schema_version": 3,
                "status": "passed",
                "request_schema_sha256": "a" * 64,
            },
        },
    )()

    worker_attestation = semantic_canary_controller._worker_attestation(
        stale_result, 2, 2
    )

    assert worker_attestation["inner_schema_version"] == 3
    assert worker_attestation["inner_attestation_state"] == "incomplete"
    assert worker_attestation["progress"] is None

@pytest.mark.parametrize(
    ("attestation", "expected_state"),
    [
        ({}, "missing"),
        ({"status": "failed"}, "missing"),
        ({"schema_version": 2}, "incomplete"),
        ({"schema_version": 5}, "invalid"),
        ({"schema_version": 99}, "invalid"),
        ({"schema_version": 4, "status": "failed"}, "incomplete"),
    ],
)
def test_worker_attestation_reports_explicit_inner_state(
    attestation: dict[str, object], expected_state: str
) -> None:
    from packages.code_understanding import semantic_canary_controller

    result = type(
        "Result",
        (),
        {"success": False, "attestation": attestation},
    )()

    worker_attestation = semantic_canary_controller._worker_attestation(result, 0, 0)

    assert worker_attestation["inner_attestation_state"] == expected_state

def test_controller_rejects_unhashable_enum_fields() -> None:
    from packages.code_understanding import semantic_canary_controller

    malformed_failure = {
        "origin": [],
        "category": {},
        "turn_ordinal": 1,
        "http_status_code": None,
        "exception_type": "Error",
        "retryable": False,
        "failure_fingerprint_sha256": "a" * 64,
    }
    state = {
        "progress": semantic_canary_controller._default_progress(),
        "last_event_sequence": 0,
        "events": 0,
    }

    assert semantic_canary_controller._bounded_failure_class([]) == "internal"
    assert semantic_canary_controller._parse_provider_failure(malformed_failure) is None
    assert not semantic_canary_controller._apply_process_event(
        state, {"event": [], "event_sequence": 1}
    )

def test_fixture_completion_is_counted_without_verification() -> None:
    from packages.code_understanding import semantic_canary_controller

    state = {
        "progress": semantic_canary_controller._default_progress(),
        "last_event_sequence": 0,
        "events": 0,
    }
    assert semantic_canary_controller._apply_process_event(
        state,
        {
            "type": "event",
            "event": "tool_call_dispatched",
            "event_sequence": 1,
            "fixture_read_call": True,
        },
    )
    assert semantic_canary_controller._apply_process_event(
        state,
        {
            "type": "event",
            "event": "tool_call_completed",
            "event_sequence": 2,
            "fixture_read_call": True,
            "fixture_read_verified": False,
        },
    )
    assert state["progress"]["fixture_read_calls_completed"] == 1
    assert state["progress"]["fixture_read_verified"] is False


def test_progress_reconciliation_preserves_observed_event_state() -> None:
    from packages.code_understanding import semantic_canary_controller

    observed = semantic_canary_controller._default_progress()
    observed["provider_turns_started"] = 1
    behind = semantic_canary_controller._default_progress()
    ahead = semantic_canary_controller._default_progress()
    ahead["provider_turns_started"] = 2
    malformed = {"provider_turns_started": 1}

    assert semantic_canary_controller._reconcile_progress(observed, observed) == (
        observed,
        True,
    )
    assert semantic_canary_controller._reconcile_progress(observed, behind) == (
        None,
        False,
    )
    assert semantic_canary_controller._reconcile_progress(observed, ahead) == (
        None,
        False,
    )
    assert semantic_canary_controller._reconcile_progress(observed, malformed) == (
        None,
        False,
    )
def test_terminate_worker_kills_descendant_after_leader_reap() -> None:
    from packages.code_understanding import semantic_canary_controller

    child_program = (
        "import os, signal, time\n"
        "if os.fork() == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(3600)\n"
        "os._exit(0)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_program],
        start_new_session=True,
    )
    try:
        process.wait(timeout=1.0)
        assert semantic_canary_controller._terminate_worker(
            process.pid, process.pid, time.monotonic() + 2.0
        )
        assert semantic_canary_controller._process_group_is_gone(process.pid)
    finally:
        if not semantic_canary_controller._process_group_is_gone(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
