"""No-provider tests for the semantic map canary."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass

import pytest

from core.llm.config import ModelConfig
from core.llm.tool_use.types import Message, StopReason, TextBlock, ToolCall, TurnResponse
from packages.code_understanding import semantic_canary
from packages.code_understanding.semantic_canary import run_semantic_canary

_TOKEN = "a1b2c3d4e5f60718293a4b5c"
_SUCCESS_FIELDS = {
    "schema_version",
    "status",
    "fixture_sha256",
    "provider",
    "model",
    "sdk_version",
    "request_schema_sha256",
    "terminal_call_count",
    "provider_turn_count",
    "fixture_read",
    "language_verified",
    "semantic_relation_verified",
    "section_counts",
}


def _model() -> ModelConfig:
    return ModelConfig(provider="gemini", model_name="test-model")


def _context_map(token: str = _TOKEN, *, language: str = "C++") -> dict:
    source = f"canary_source_{token}"
    sink = f"canary_sink_{token}"
    relation = f"canary_relation_{token}"
    return {
        "sources": [{"name": source}],
        "sinks": [{"name": sink}],
        "trust_boundaries": [],
        "meta": {"language": language},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [{
            "source": source,
            "sink": sink,
            "relation": relation,
        }],
    }


@pytest.fixture(autouse=True)
def _fixed_canary_token(monkeypatch) -> None:
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: _TOKEN)


@dataclass
class _Turn:
    calls: list[ToolCall]
    reason: StopReason = StopReason.NEEDS_TOOL_CALL
    cost: float = 0.0


class _Provider:
    def __init__(self, turns: list[_Turn], *, first_error: Exception | None = None) -> None:
        self._turns = iter(turns)
        self._first_error = first_error
        self.calls = 0
        self.systems: list[str] = []
        self.messages: list[list[Message]] = []

    def supports_tool_use(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        return False

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def context_window(self) -> int:
        return 200_000

    def compute_cost(self, response: TurnResponse) -> float:
        return response.cost_usd

    def turn(self, messages, _tools, **kwargs) -> TurnResponse:
        self.calls += 1
        self.messages.append(list(messages))
        self.systems.append(kwargs["system"])
        if self._first_error is not None:
            error = self._first_error
            self._first_error = None
            raise error
        turn = next(self._turns)
        return TurnResponse(
            content=turn.calls,
            stop_reason=turn.reason,
            input_tokens=1,
            output_tokens=1,
            cost_usd=turn.cost,
        )


def _terminal_turns(
    count: int = 1,
    *,
    context_map: dict | None = None,
    read_path: str = "fixture.cpp",
) -> list[_Turn]:
    context_map = _context_map() if context_map is None else context_map
    return [
        _Turn([ToolCall(id="read", name="read_file", input={"path": read_path})]),
        _Turn([
            ToolCall(
                id=f"call-{index}",
                name="submit_context_map",
                input={"context_map": context_map},
            )
            for index in range(count)
        ]),
    ]


def _assert_failure(result, failure_class: str) -> None:
    assert result.success is False
    assert set(result.attestation) == _SUCCESS_FIELDS | {"failure_class"}
    assert result.attestation["status"] == "failed"
    assert result.attestation["failure_class"] == failure_class


def _first_user_text(provider: _Provider) -> str:
    for message in provider.messages[0]:
        for content in message.content:
            if isinstance(content, TextBlock):
                return content.text
    raise AssertionError("missing canary user message")


def test_canary_attests_only_bounded_non_sensitive_evidence():
    provider = _Provider(_terminal_turns())
    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    assert result.success is True
    assert set(result.attestation) == _SUCCESS_FIELDS
    assert result.attestation["schema_version"] == 1
    assert result.attestation["status"] == "passed"
    assert len(result.attestation["fixture_sha256"]) == 64
    assert len(result.attestation["request_schema_sha256"]) == 64
    assert result.attestation["provider"] == "gemini"
    assert result.attestation["model"] == "test-model"
    assert result.attestation["terminal_call_count"] == 1
    assert result.attestation["provider_turn_count"] == 2
    assert result.attestation["fixture_read"] is True
    assert result.attestation["language_verified"] is True
    assert result.attestation["semantic_relation_verified"] is True
    assert result.attestation["section_counts"] == {
        "sources": 1,
        "sinks": 1,
        "trust_boundaries": 0,
        "entry_points": 0,
        "sink_details": 0,
        "boundary_details": 0,
        "unchecked_flows": 1,
    }
    rendered = repr(result.attestation)
    assert "fixture.cpp" not in rendered
    assert _TOKEN not in rendered
    assert "canary_source_" not in rendered
    assert "canary_sink_" not in rendered
    assert "canary_relation_" not in rendered


def test_canary_does_not_disclose_expected_language_to_the_model():
    provider = _Provider(_terminal_turns())

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    assert result.success is True
    assert "Recognize the source as C++" not in provider.systems[0]
    assert '"language": "C++"' not in _first_user_text(provider)


def test_canary_rejects_terminal_map_without_fixture_read():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider([_Turn([ToolCall(
            id="terminal",
            name="submit_context_map",
            input={"context_map": _context_map()},
        )])]),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["fixture_read"] is False
    assert result.attestation["language_verified"] is False
    assert result.attestation["semantic_relation_verified"] is False


def test_canary_rejects_failed_fixture_read():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(read_path="missing.cpp")),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["fixture_read"] is False
    assert result.attestation["language_verified"] is False
    assert result.attestation["semantic_relation_verified"] is False


def test_canary_rejects_language_label_without_content_derived_relation():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider([_Turn([ToolCall(
            id="terminal",
            name="submit_context_map",
            input={"context_map": _context_map(language="cpp")},
        )])]),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["fixture_read"] is False
    assert result.attestation["language_verified"] is False
    assert result.attestation["semantic_relation_verified"] is False


def test_canary_rejects_generic_empty_map():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map={})),
    )

    _assert_failure(result, "map_validation")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sources", [{"name": "wrong-source"}]),
        ("sinks", [{"name": "wrong-sink"}]),
        ("unchecked_flows", [{
            "source": f"canary_source_{_TOKEN}",
            "sink": f"canary_sink_{_TOKEN}",
            "relation": "wrong-relation",
        }]),
    ],
)
def test_canary_rejects_wrong_hidden_semantic_identifier(field, value):
    context_map = _context_map()
    context_map[field] = value
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")


def test_canary_uses_fresh_fixture_identifiers_each_run(monkeypatch):
    first_token = "1" * 24
    second_token = "2" * 24
    tokens = iter((first_token, second_token))
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: next(tokens))

    first = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=_context_map(first_token))),
    )
    second = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=_context_map(second_token))),
    )

    assert first.success is True
    assert second.success is True
    assert first.attestation["fixture_sha256"] != second.attestation["fixture_sha256"]


def test_canary_rejects_undeclared_context_map_fields_without_attesting_them():
    context_map = _context_map()
    context_map["fixture_source_text"] = "do-not-emit"
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "map_validation")
    assert "fixture_source_text" not in repr(result.attestation)
    assert "do-not-emit" not in repr(result.attestation)


def test_canary_rejects_oversized_terminal_payload():
    context_map = _context_map()
    context_map["sources"] = [{"detail": "x" * 16_384}]
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "map_validation")


def test_canary_rejects_duplicate_terminal_calls():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(2)),
    )

    _assert_failure(result, "terminal_contract")


def test_canary_rejects_prose_only_completion():
    provider = _Provider([_Turn([], reason=StopReason.COMPLETE)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")


def test_canary_enforces_cost_quota():
    provider = _Provider([
        _Turn([ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})], cost=1.0),
    ])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "cost_limit")


def test_canary_enforces_iteration_limit():
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN) for _ in range(3)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")
    assert provider.calls == 3


def test_canary_enforces_wall_clock_limit(monkeypatch):
    moments = iter((0.0, 0.0, 31.0))
    loop_module = importlib.import_module("core.llm.tool_use.loop")
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: next(moments))
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")
    assert provider.calls == 1


@pytest.mark.parametrize(
    ("message", "failure_class"),
    [
        ("401 unauthorized", "auth"),
        ("429 quota exhausted", "quota"),
    ],
)
def test_canary_reports_auth_or_quota_failure_without_success(message, failure_class):
    provider = _Provider(_terminal_turns(), first_error=RuntimeError(message))

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, failure_class)
    assert provider.calls == 1


def test_canary_never_retries_transport_failure():
    provider = _Provider(_terminal_turns(), first_error=RuntimeError("503 transient"))

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "transport")
    assert provider.calls == 1


def test_semantic_canary_cli_rejects_unapproved_model_as_bounded_json(capsys):
    exit_code = semantic_canary.main(["--model", "other-model", "--format", "json"])

    assert exit_code == 1
    attestation = json.loads(capsys.readouterr().out)
    assert set(attestation) == _SUCCESS_FIELDS | {"failure_class"}
    assert attestation["failure_class"] == "unsupported_model"


def test_semantic_canary_cli_uses_resolved_model(monkeypatch, capsys):
    expected = semantic_canary.SemanticCanaryResult(True, {
        "schema_version": 1,
        "status": "passed",
        "fixture_sha256": "f" * 64,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "sdk_version": "test",
        "request_schema_sha256": "s" * 64,
        "terminal_call_count": 1,
        "provider_turn_count": 2,
        "fixture_read": True,
        "language_verified": True,
        "semantic_relation_verified": True,
        "section_counts": {key: 0 for key in semantic_canary._SECTION_KEYS},
    })
    monkeypatch.setattr(semantic_canary, "resolve_semantic_canary_model", lambda _: _model())
    monkeypatch.setattr(semantic_canary, "run_semantic_canary", lambda _: expected)

    exit_code = semantic_canary.main(["--model", "gemini-2.5-flash", "--format", "json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected.attestation
