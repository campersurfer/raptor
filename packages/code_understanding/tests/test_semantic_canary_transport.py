"""Hermetic Gemini-over-UDS semantic-canary transport coverage."""

from __future__ import annotations

from collections import deque
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.llm.config import ModelConfig
from core.llm.dispatcher.auth import CredentialStore
from core.llm.dispatcher.server import LLMDispatcher
from core.llm.providers import GeminiProvider
from qualification.controller import _reconcile_provider_failure
from packages.code_understanding import semantic_canary
from packages.code_understanding.semantic_canary import run_semantic_canary


_IDENTIFIER = "a1b2c3d4e5f60718293a4b5c"


def _line_of(content: str, fragment: str) -> int:
    return content[: content.index(fragment)].count("\n") + 1


def _context_map_from_fixture(content: str) -> dict[str, Any]:
    """Derive the canonical terminal map exclusively from a read fixture."""
    source_statement = 'return argc > 1 ? argv[1] : "";'
    sink_statement = "std::system(command.c_str());"
    source = re.search(r"std::string\s+(canary_source_[0-9a-f]+)\(int argc", content)
    sink = re.search(r"void\s+(canary_sink_[0-9a-f]+)\(const std::string& command\)", content)
    relation = re.search(r"void\s+(canary_relation_[0-9a-f]+)\(int argc", content)
    assert source and sink and relation
    source_name, sink_name, relation_name = source.group(1), sink.group(1), relation.group(1)
    source_line = _line_of(content, source_statement)
    sink_line = _line_of(content, sink_statement)
    relation_line = _line_of(content, f"void {relation_name}(")
    return {
        "sources": [{
            "type": "cli_arg",
            "entry": f"fixture.cpp:{source_line} {source_name} argv[1]",
            "trust_level": "attacker_controlled",
        }],
        "sinks": [{"type": "shell_exec", "location": f"fixture.cpp:{sink_line}"}],
        "trust_boundaries": [],
        "meta": {"app_type": "cli", "language": ["C++"]},
        "entry_points": [{
            "id": "EP-1", "type": "cli_arg", "file": "fixture.cpp",
            "line": relation_line, "name": relation_name,
        }],
        "sink_details": [{
            "id": "SINK-1", "type": "shell_exec", "operation": sink_statement,
            "file": "fixture.cpp", "line": sink_line, "name": sink_name,
        }],
        "boundary_details": [],
        "unchecked_flows": [{
            "entry_point": "EP-1", "sink": "SINK-1",
            "missing_boundary": "unvalidated command argument",
        }],
    }


def _gemini_response(text: str) -> bytes:
    return json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }, separators=(",", ":")).encode("utf-8")


def _gemini_error(status: int, code: str | None = None) -> bytes:
    status_name = code or {401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 429: "RESOURCE_EXHAUSTED"}.get(status, "INTERNAL")
    return json.dumps({"error": {
        "code": status, "status": status_name, "message": "private upstream body must never persist",
    }}, separators=(",", ":")).encode("utf-8")


class _StreamResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.headers = {
            "content-type": "application/json",
            "content-length": str(len(body)),
            "x-private-upstream": "secret",
        }
        self._body = body

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def iter_raw(self):
        yield self._body


class _Upstream:
    """In-process dispatcher upstream; no listener or TCP transport exists."""

    def __init__(self, outcomes: list[_StreamResponse | Exception]) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[dict[str, Any]] = []

    def client_type(self):
        upstream = self

        class _Client:
            def __init__(self, **_kwargs: object) -> None:
                self._kwargs = _kwargs

            def __enter__(self) -> _Client:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

            def stream(self, method: str, url: str, *, content: bytes, headers: Any):
                upstream.requests.append({
                    "method": method,
                    "url": url,
                    "body": bytes(content),
                    "headers": dict(headers),
                })
                outcome = upstream._outcomes.popleft()
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        return _Client


def _credentials() -> CredentialStore:
    creds = CredentialStore.__new__(CredentialStore)
    creds._keys = {"anthropic": None, "openai": None, "gemini": "AIza-real-key-never-sent"}
    return creds


def _wait_for_trace(path: Path, count: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if len(records) >= count:
            return records
        time.sleep(0.01)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _run_real_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcomes: list[_StreamResponse | Exception],
):
    from core.llm.dispatcher import client as dispatcher_client

    trace_path = tmp_path / "qualification-trace.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    upstream = _Upstream(outcomes)
    # Keep the SDK's real httpx module intact: replace only the server's
    # module binding so its in-process upstream client is mocked while the
    # provider still traverses the actual UDS HTTPTransport.
    import core.llm.dispatcher.server as dispatcher_server
    monkeypatch.setattr(
        dispatcher_server,
        "httpx",
        SimpleNamespace(
            Client=upstream.client_type(),
            Timeout=httpx.Timeout,
            HTTPError=httpx.HTTPError,
        ),
    )
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: _IDENTIFIER)
    monkeypatch.setattr(dispatcher_client, "_cached_token", None)
    dispatcher = LLMDispatcher(
        run_id="semantic-canary-transport",
        audit_path=audit_path,
        qualification_trace_path=trace_path,
        token_ttl_s=60,
        token_budget=2,
        creds=_credentials(),
    )
    _, token_fd = dispatcher.allocate_worker(label="semantic-canary-sdk")
    monkeypatch.setenv("RAPTOR_LLM_SOCKET", str(dispatcher.socket_path))
    monkeypatch.setenv("RAPTOR_LLM_TOKEN_FD", str(token_fd))
    model = ModelConfig(
        provider="gemini", model_name="gemini-2.5-flash", api_key="unused", timeout=5.0,
    )
    lifecycle: list[tuple[str, dict[str, Any]]] = []
    try:
        result = run_semantic_canary(
            model,
            provider_factory=GeminiProvider,
            lifecycle_events=lambda event, **metadata: lifecycle.append((event, metadata)),
        )
    finally:
        dispatcher.shutdown()
    records = _wait_for_trace(trace_path, len(upstream.requests))
    return result, upstream, records, lifecycle, trace_path, audit_path, dispatcher


def _assert_first_turn_progress(result: Any, lifecycle: list[tuple[str, dict[str, Any]]]) -> None:
    attestation = result.attestation
    assert attestation["provider_turns_started"] == 2
    assert attestation["provider_turns_completed"] == 1
    assert attestation["provider_turns_failed"] == 1
    assert attestation["tool_calls_dispatched"] == 1
    assert attestation["tool_calls_completed"] == 1
    assert attestation["fixture_read_calls_dispatched"] == 1
    assert attestation["fixture_read_calls_completed"] == 1
    assert attestation["fixture_read_verified"] is True
    assert any(
        event == "tool_call_completed"
        and metadata.get("fixture_read_call") is True
        and metadata.get("fixture_read_verified") is True
        for event, metadata in lifecycle
    )


def test_real_gemini_sdk_uds_reuses_cached_fd_token_and_carries_fixture_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: _IDENTIFIER)
    fixture = semantic_canary._fresh_fixture()
    terminal_map = _context_map_from_fixture(fixture.content)
    upstream = [
        _StreamResponse(200, _gemini_response(json.dumps({"tool": "read_file", "input": {"path": "fixture.cpp"}}))),
        _StreamResponse(200, _gemini_response(json.dumps({"tool": "submit_context_map", "input": {"context_map": terminal_map}}))),
    ]

    result, captured, trace, lifecycle, trace_path, audit_path, dispatcher = _run_real_stack(
        monkeypatch, tmp_path, upstream,
    )

    assert result.success is True
    attestation = result.attestation
    assert attestation["provider_turn_count"] == 2
    assert attestation["terminal_call_count"] == 1
    assert attestation["fixture_read_verified"] is True
    assert len(captured.requests) == 2
    second_request = captured.requests[1]
    second_payload = json.loads(second_request["body"])
    second_text = second_payload["contents"][0]["parts"][0]["text"]
    read_result_match = re.search(
        r"<untrusted-[^>]+>\s*(\{.*\})\s*</untrusted-[^>]+>",
        second_text,
        re.DOTALL,
    )
    assert read_result_match is not None
    assert json.loads(read_result_match.group(1)) == {
        "path": "fixture.cpp",
        "content": fixture.content,
        "truncated": False,
        "byte_cap": 262144,
    }
    assert [record["request_ordinal"] for record in trace] == [1, 2]
    assert [record["provider"] for record in trace] == ["gemini", "gemini"]
    assert all(record["token_accepted"] is True for record in trace)
    assert all(record["request_received"] is True for record in trace)
    assert all(record["upstream_status_code"] == 200 for record in trace)
    assert all(record["response_stream_completed"] is True for record in trace)
    assert all(record["exception_type"] is None for record in trace)
    trace_fields = {
        "request_ordinal", "provider", "token_accepted", "request_received",
        "upstream_started", "upstream_status_code", "response_headers_started",
        "response_stream_completed", "exception_type",
    }
    assert all(set(record) == trace_fields for record in trace)
    token_records = list(dispatcher._tokens.values())
    assert len(token_records) == 1
    assert token_records[0].requests_made == 2
    from core.llm.dispatcher import client as dispatcher_client
    assert dispatcher_client._cached_token == token_records[0].value
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
    dispatches = [event for event in audit if event["event"] == "request.dispatch"]
    assert len(dispatches) == 2
    assert dispatches[0]["token_id"] == dispatches[1]["token_id"]
    assert dispatches[0]["worker_label"] == dispatches[1]["worker_label"]
    durable = "\n".join((trace_path.read_text(), audit_path.read_text(), json.dumps(attestation, sort_keys=True)))
    for forbidden in (fixture.content, "private upstream body", "x-private-upstream", "secret"):
        assert forbidden not in durable
    assert [event for event, _ in lifecycle].count("provider_turn_started") == 2
    assert [event for event, _ in lifecycle].count("provider_turn_completed") == 2
    assert any(event == "terminal_call_dispatched" for event, _ in lifecycle)


@pytest.mark.parametrize(
    (
        "outcome",
        "expected_raw_category",
        "expected_raw_status",
        "expected_category",
        "expected_origin",
        "expected_exception_type",
        "expected_retryable",
        "expected_failure_class",
    ),
    [
        (_StreamResponse(400, _gemini_error(400, "INVALID_ARGUMENT")), "schema", 400, "schema", "upstream_http", "ClientError", False, "schema"),
        (_StreamResponse(401, _gemini_error(401)), "auth", 401, "auth", "upstream_http", "ClientError", False, "auth"),
        (_StreamResponse(403, _gemini_error(403)), "auth", 403, "auth", "upstream_http", "ClientError", False, "auth"),
        (_StreamResponse(429, _gemini_error(429)), "quota", 429, "quota", "upstream_http", "ClientError", True, "quota"),
        (_StreamResponse(500, _gemini_error(500)), "upstream_5xx", 500, "upstream_5xx", "upstream_http", "ServerError", True, "transport"),
        (_StreamResponse(502, _gemini_error(502)), "upstream_5xx", 502, "upstream_5xx", "upstream_http", "ServerError", True, "transport"),
        (_StreamResponse(503, _gemini_error(503)), "upstream_5xx", 503, "upstream_5xx", "upstream_http", "ServerError", True, "transport"),
        (ConnectionRefusedError("private upstream socket"), "upstream_5xx", 502, "connection_refused", "dispatcher_upstream_connect", "ConnectionRefusedError", True, "transport"),
        (ConnectionResetError("private upstream socket"), "upstream_5xx", 502, "connection_reset", "dispatcher_upstream_connect", "ConnectionResetError", True, "transport"),
        (httpx.ReadTimeout("private upstream timeout"), "upstream_5xx", 502, "timeout", "dispatcher_upstream_connect", "ReadTimeout", True, "transport"),
        (httpx.RemoteProtocolError("private upstream protocol"), "upstream_5xx", 502, "protocol", "dispatcher_upstream_connect", "RemoteProtocolError", False, "transport"),
        (_StreamResponse(200, b"{malformed"), "response_decode", None, "response_decode", "sdk_response_decode", "JSONDecodeError", False, "transport"),
        (_StreamResponse(200, _gemini_response("")), "empty_response", None, "empty_response", "provider_empty_response", "ProviderEmptyResponseError", False, "transport"),
        (RuntimeError("private unknown local exception"), "upstream_5xx", 502, "transport_unknown", "dispatcher_upstream_connect", "RuntimeError", True, "transport"),
    ],
)
def test_real_gemini_sdk_uds_preserves_progress_and_bounds_second_turn_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: _StreamResponse | Exception,
    expected_raw_category: str,
    expected_raw_status: int | None,
    expected_category: str,
    expected_origin: str,
    expected_exception_type: str,
    expected_retryable: bool,
    expected_failure_class: str,
):
    first = _StreamResponse(
        200, _gemini_response(json.dumps({"tool": "read_file", "input": {"path": "fixture.cpp"}})),
    )
    result, captured, trace, lifecycle, trace_path, audit_path, _dispatcher = _run_real_stack(
        monkeypatch, tmp_path, [first, outcome],
    )

    assert result.success is False
    _assert_first_turn_progress(result, lifecycle)
    assert result.attestation["failure_class"] == expected_failure_class
    failure = result.attestation["provider_failure"]
    assert failure is not None
    assert failure["turn_ordinal"] == 2
    assert failure["category"] == expected_raw_category
    assert failure["http_status_code"] == expected_raw_status

    assert len(captured.requests) == 2
    assert [request["method"] for request in captured.requests] == ["POST", "POST"]
    assert [record["request_ordinal"] for record in trace] == [1, 2]
    trace_fields = {
        "request_ordinal", "provider", "token_accepted", "request_received",
        "upstream_started", "upstream_status_code", "response_headers_started",
        "response_stream_completed", "exception_type",
    }
    assert all(set(record) == trace_fields for record in trace)
    assert all(record["provider"] == "gemini" for record in trace)
    assert all(record["token_accepted"] is True for record in trace)
    assert all(record["request_received"] is True for record in trace)
    assert all(record["upstream_started"] is True for record in trace)
    assert trace[0]["upstream_status_code"] == 200
    assert trace[0]["response_stream_completed"] is True
    assert trace[0]["exception_type"] is None

    second_trace = trace[1]
    if expected_origin == "upstream_http":
        assert second_trace["upstream_status_code"] == expected_raw_status
        assert second_trace["response_headers_started"] is True
        assert second_trace["response_stream_completed"] is True
        assert second_trace["exception_type"] is None
        reconciled_status = expected_raw_status
    elif expected_origin in {"sdk_response_decode", "provider_empty_response"}:
        assert second_trace["upstream_status_code"] == 200
        assert second_trace["response_headers_started"] is True
        assert second_trace["response_stream_completed"] is True
        assert second_trace["exception_type"] is None
        reconciled_status = None
    else:
        assert second_trace["upstream_status_code"] is None
        assert second_trace["response_headers_started"] is False
        assert second_trace["response_stream_completed"] is False
        assert second_trace["exception_type"] == expected_exception_type
        reconciled_status = None

    reconciled = _reconcile_provider_failure(failure, trace)
    assert reconciled is not None
    assert reconciled["origin"] == expected_origin
    assert reconciled["category"] == expected_category
    assert reconciled["turn_ordinal"] == 2
    assert reconciled["http_status_code"] == reconciled_status
    assert reconciled["exception_type"] == expected_exception_type
    assert reconciled["retryable"] is expected_retryable
    assert len(reconciled["failure_fingerprint_sha256"]) == 64

    fixture = semantic_canary._fresh_fixture()
    durable = "\n".join((trace_path.read_text(), audit_path.read_text(), json.dumps(result.attestation, sort_keys=True)))
    for forbidden in (fixture.content, "private", "x-private-upstream", "secret"):
        assert forbidden not in durable
