"""No-provider tests for the semantic map canary."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from core.llm.config import ModelConfig
from core.llm.tool_use.types import StopReason, ToolCall, TurnResponse
from packages.code_understanding import semantic_canary
from packages.code_understanding.semantic_canary import run_semantic_canary

_TOKEN = "a1b2c3d4e5f60718293a4b5c"


def _model() -> ModelConfig:
    return ModelConfig(provider="test", model_name="test-model")


def _context_map(token: str = _TOKEN) -> dict:
    source = f"canary_source_{token}"
    sink = f"canary_sink_{token}"
    relation = f"canary_relation_{token}"
    return {
        "sources": [{"name": source}],
        "sinks": [{"name": sink}],
        "trust_boundaries": [],
        "meta": {"language": "C++"},
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
    def __init__(
        self,
        turns: list[_Turn],
        *,
        transient_once: bool = False,
        first_error: Exception | None = None,
    ) -> None:
        self._turns = iter(turns)
        self._transient_once = transient_once
        self._first_error = first_error
        self.calls = 0

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

    def turn(self, _messages, _tools, **_kwargs) -> TurnResponse:
        self.calls += 1
        if self._first_error is not None:
            error = self._first_error
            self._first_error = None
            raise error
        if self._transient_once:
            self._transient_once = False
            raise RuntimeError("503 transient")
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


def test_canary_attests_only_bounded_non_sensitive_evidence():
    result = run_semantic_canary(_model(), provider_factory=lambda _: _Provider(_terminal_turns()))

    assert result.success is True
    assert result.attestation == {
        "status": "passed",
        "fixture_sha256": "3d9774485c3eb5631e17e4cbed80847e1670629475373f6b81fc7b614c3cf4d2",
        "provider": "test",
        "model": "test-model",
        "terminal_calls": 1,
        "attempts": 2,
        "fixture_read": True,
        "language_verified": True,
        "semantic_relation_verified": True,
        "section_counts": {
            "sources": 1, "sinks": 1, "trust_boundaries": 0,
            "entry_points": 0, "sink_details": 0,
            "boundary_details": 0, "unchecked_flows": 1,
        },
    }
    rendered = repr(result.attestation)
    assert "fixture.cpp" not in rendered
    assert _TOKEN not in rendered
    assert "canary_source_" not in rendered
    assert "canary_sink_" not in rendered
    assert "canary_relation_" not in rendered
    assert len(result.attestation["provider"]) <= 128
    assert len(result.attestation["model"]) <= 128


def test_canary_rejects_terminal_map_without_fixture_read():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider([_Turn([ToolCall(
            id="terminal",
            name="submit_context_map",
            input={"context_map": _context_map()},
        )])]),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "semantic-evidence"}


def test_canary_requires_successful_read_of_the_actual_fixture():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(read_path="missing.cpp")),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "semantic-evidence"}


def test_canary_rejects_non_cpp_language_claim():
    context_map = _context_map()
    context_map["meta"] = {"language": "Python"}
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "semantic-evidence"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sources", [{"name": "static_source"}]),
        ("sinks", [{"name": "static_sink"}]),
        ("unchecked_flows", [{
            "source": "static_source",
            "sink": "static_sink",
            "relation": "static_relation",
        }]),
    ],
)
def test_canary_rejects_non_content_derived_semantic_claims(field, value):
    context_map = _context_map()
    context_map[field] = value
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "semantic-evidence"}


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


def test_canary_redacts_undeclared_context_map_fields():
    context_map = _context_map()
    context_map["fixture_source_text"] = "do-not-emit"
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "map-validation"}
    assert "fixture_source_text" not in repr(result.attestation)
    assert "do-not-emit" not in repr(result.attestation)


def test_canary_retries_once_without_a_provider_call():
    provider = _Provider(_terminal_turns(), transient_once=True)
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: provider,
        sleep=lambda _: None,
    )

    assert result.success is True
    assert result.attestation["attempts"] == 3
    assert provider.calls == 3


@pytest.mark.parametrize("message", ["daily quota exhausted", "429 rate limit exceeded"])
def test_canary_does_not_retry_quota_failure_or_report_success(message):
    provider = _Provider(_terminal_turns(), first_error=RuntimeError(message))
    sleeps: list[float] = []
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: provider,
        sleep=sleeps.append,
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "loop-failure"}
    assert provider.calls == 1
    assert sleeps == []


def test_canary_rejects_provider_initialization_failure():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "provider-init"}


def test_canary_rejects_duplicate_terminal_calls():
    result = run_semantic_canary(
        _model(), provider_factory=lambda _: _Provider(_terminal_turns(2)),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "terminal-contract"}


def test_canary_enforces_cost_quota():
    provider = _Provider([
        _Turn([ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})], cost=1.0),
    ])
    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "cost-limit"}


def test_canary_enforces_iteration_limit():
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN) for _ in range(3)])
    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "terminal-contract"}
    assert provider.calls == 3


def test_canary_enforces_wall_clock_limit(monkeypatch):
    moments = iter((0.0, 0.0, 31.0))
    loop_module = importlib.import_module("core.llm.tool_use.loop")
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: next(moments))
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": "terminal-contract"}
    assert provider.calls == 1
