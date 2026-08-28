"""No-provider tests for the semantic map canary."""

from __future__ import annotations
import importlib

from dataclasses import dataclass

from core.llm.config import ModelConfig
from core.llm.tool_use.types import StopReason, ToolCall, TurnResponse
from packages.code_understanding.semantic_canary import run_semantic_canary


def _model() -> ModelConfig:
    return ModelConfig(provider="test", model_name="test-model")


def _context_map() -> dict:
    return {
        "sources": [],
        "sinks": [],
        "trust_boundaries": [],
        "meta": {},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [],
    }


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


def _terminal_turns(count: int = 1) -> list[_Turn]:
    return [_Turn([
        ToolCall(
            id=f"call-{index}",
            name="submit_context_map",
            input={"context_map": _context_map()},
        )
        for index in range(count)
    ])]


def test_canary_attests_without_exposing_fixture_or_model_data():
    result = run_semantic_canary(_model(), provider_factory=lambda _: _Provider(_terminal_turns()))

    assert result.success is True
    assert result.attestation == {
        "status": "passed",
        "fixture": "isolated-cpp",
        "provider": "test",
        "model": "test-model",
        "terminal_calls": 1,
        "attempts": 1,
        "section_counts": {
            "sources": 0, "sinks": 0, "trust_boundaries": 0,
            "entry_points": 0, "sink_details": 0,
            "boundary_details": 0, "unchecked_flows": 0,
        },
    }
    rendered = repr(result.attestation)
    assert "fixture.cpp" not in rendered
    assert "test-model" in rendered


def test_canary_retries_once_without_a_provider_call():
    provider = _Provider(_terminal_turns(), transient_once=True)
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: provider,
        sleep=lambda _: None,
    )

    assert result.success is True
    assert result.attestation["attempts"] == 2
    assert provider.calls == 2


def test_canary_does_not_retry_daily_quota_or_report_success():
    provider = _Provider(
        _terminal_turns(),
        first_error=RuntimeError("daily quota exhausted"),
    )
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

def test_canary_does_not_retry_rate_limit_or_report_success():
    provider = _Provider(
        _terminal_turns(),
        first_error=RuntimeError("429 rate limit exceeded"),
    )
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
