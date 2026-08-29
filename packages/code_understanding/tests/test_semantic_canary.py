"""Offline contract tests for the isolated semantic canary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.llm.config import ModelConfig
from core.llm.tool_use.types import StopReason, ToolCall, TurnResponse
from packages.code_understanding import semantic_canary

_TOKEN = "a1b2c3d4e5f60718293a4b5c"


def _model() -> ModelConfig:
    return ModelConfig(provider="test", model_name="test-model")


def _context_map(token: str = _TOKEN) -> dict:
    source = f"canary_source_{token}"
    sink = f"canary_sink_{token}"
    relation = f"canary_relation_{token}"
    return {
        "meta": {"language": "C++"},
        "sources": [{"name": source}],
        "sinks": [{"name": sink}],
        "unchecked_flows": [{"source": source, "sink": sink, "relation": relation}],
    }


@pytest.fixture(autouse=True)
def _fixed_fixture(monkeypatch) -> None:
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: _TOKEN)


@dataclass
class _Turn:
    calls: list[ToolCall]


class _Provider:
    def __init__(self, turns: list[_Turn]) -> None:
        self._turns = iter(turns)

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
        return TurnResponse(
            content=next(self._turns).calls,
            stop_reason=StopReason.NEEDS_TOOL_CALL,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
        )


def _turns(context_map: dict | None = None, *, read_path: str = "fixture.cpp") -> list[_Turn]:
    return [
        _Turn([ToolCall(id="read", name="read_file", input={"path": read_path})]),
        _Turn([ToolCall(
            id="terminal",
            name="submit_context_map",
            input={"context_map": _context_map() if context_map is None else context_map},
        )]),
    ]


def test_canary_requires_read_and_exact_content_derived_relation() -> None:
    result = semantic_canary.run_semantic_canary(
        _model(), provider_factory=lambda _: _Provider(_turns()),
    )

    assert result.success is True
    assert result.attestation == {
        "status": "passed",
        "fixture_sha256": "3d9774485c3eb5631e17e4cbed80847e1670629475373f6b81fc7b614c3cf4d2",
        "provider": "test",
        "model": "test-model",
        "terminal_calls": 1,
        "fixture_read": True,
        "language_verified": True,
        "semantic_relation_verified": True,
    }


@pytest.mark.parametrize(
    ("turns", "reason"),
    [
        (_turns(read_path="other.cpp"), "semantic-evidence"),
        (_turns({**_context_map(), "unchecked_flows": [{
            "source": f"canary_source_{_TOKEN}",
            "sink": f"canary_sink_{_TOKEN}",
            "relation": "invented",
        }]}), "semantic-evidence"),
    ],
)
def test_canary_rejects_read_or_relation_that_is_not_exact(turns, reason) -> None:
    result = semantic_canary.run_semantic_canary(
        _model(), provider_factory=lambda _: _Provider(turns),
    )

    assert result.success is False
    assert result.attestation == {"status": "failed", "reason": reason}


def test_terminal_contract_is_closed_and_rejects_extra_fields(tmp_path) -> None:
    schema = semantic_canary._terminal_schema()
    context_schema = schema["properties"]["context_map"]

    assert schema["additionalProperties"] is False
    assert context_schema["additionalProperties"] is False
    invalid_payload = {"context_map": {**_context_map(), "unexpected": "value"}}
    assert semantic_canary._validate_terminal_context_map(invalid_payload) is None

    terminal = semantic_canary._build_tools(
        semantic_canary.SandboxedTools.for_repo(tmp_path),
    )[-1]
    with pytest.raises(ValueError, match="terminal contract"):
        terminal.handler(invalid_payload)
