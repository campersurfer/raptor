"""Tests for the read-only model-backed /understand map dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from unittest.mock import patch

import pytest

from core.llm.config import ModelConfig
from core.llm.tool_use.types import StopReason, TextBlock, ToolCall, TurnResponse


@dataclass
class FakeTurn:
    tool_calls: list[tuple[str, dict]] | None = None
    text: str = ""


class FakeProvider:
    def __init__(self, turns: list[FakeTurn]):
        self._turns: Iterator[FakeTurn] = iter(turns)
        self._calls = 0

    def supports_tool_use(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        return False

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def context_window(self) -> int:
        return 200_000

    def compute_cost(self, response: TurnResponse) -> float:
        return 0.0

    def turn(self, messages, tools, **kwargs) -> TurnResponse:
        del messages, tools, kwargs
        self._calls += 1
        turn = next(self._turns, FakeTurn(text="[end of script]"))
        content: list = [TextBlock(turn.text)] if turn.text else []
        for index, (name, payload) in enumerate(turn.tool_calls or []):
            content.append(ToolCall(
                id=f"call_{self._calls}_{index}",
                name=name,
                input=payload,
            ))
        return TurnResponse(
            content=content,
            stop_reason=(
                StopReason.NEEDS_TOOL_CALL if turn.tool_calls else StopReason.COMPLETE
            ),
            input_tokens=10,
            output_tokens=5,
        )


def _context_map() -> dict:
    return {
        "sources": [],
        "sinks": [],
        "trust_boundaries": [],
        "meta": {"app_type": "library"},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [],
    }


def _model() -> ModelConfig:
    return ModelConfig(provider="gemini", model_name="gemini-2.5-pro")


def _patch_provider(turns: list[FakeTurn]):
    return patch(
        "packages.code_understanding.dispatch.map_dispatch.create_provider",
        return_value=FakeProvider(turns),
    )


def test_map_dispatch_accepts_terminal_context_map(tmp_path):
    from packages.code_understanding.dispatch.map_dispatch import default_map_dispatch

    result_map = _context_map()
    with _patch_provider([
        FakeTurn(tool_calls=[("submit_context_map", {"context_map": result_map})]),
    ]):
        result = default_map_dispatch(
            _model(),
            str(tmp_path),
            checklist={"files": []},
        )

    assert result == result_map


def test_map_dispatch_can_read_source_but_has_no_write_or_shell_tool(tmp_path):
    from packages.code_understanding.dispatch.map_dispatch import (
        _build_tools,
        default_map_dispatch,
    )
    from packages.code_understanding.dispatch.tools import SandboxedTools

    source = tmp_path / "handler.py"
    source.write_text("def handle(value):\n    return value\n")
    result_map = _context_map()
    with _patch_provider([
        FakeTurn(tool_calls=[("read_file", {"path": "handler.py"})]),
        FakeTurn(tool_calls=[("submit_context_map", {"context_map": result_map})]),
    ]):
        result = default_map_dispatch(_model(), str(tmp_path))

    assert result == result_map
    tools = _build_tools(SandboxedTools.for_repo(tmp_path))
    names = sorted(tool.name for tool in tools)
    assert names == ["glob_files", "grep", "read_file", "submit_context_map"]
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    assert "handle" in read_tool.handler({"path": "handler.py"})


def test_map_dispatch_rejects_missing_context_map(tmp_path):
    from packages.code_understanding.dispatch.map_dispatch import default_map_dispatch

    with _patch_provider([
        FakeTurn(tool_calls=[("submit_context_map", {})]),
    ]):
        result = default_map_dispatch(_model(), str(tmp_path))

    assert result["error"] == "submit_context_map payload missing context_map object"

def test_map_submission_schema_requires_object_entries(tmp_path):
    from packages.code_understanding.dispatch.map_dispatch import _build_tools
    from packages.code_understanding.dispatch.tools import SandboxedTools

    tools = _build_tools(SandboxedTools.for_repo(tmp_path))
    submit_tool = next(tool for tool in tools if tool.name == "submit_context_map")
    fields = submit_tool.input_schema["properties"]["context_map"]["properties"]
    for name in (
        "sources",
        "sinks",
        "trust_boundaries",
        "entry_points",
        "sink_details",
        "boundary_details",
        "unchecked_flows",
    ):
        assert fields[name] == {"type": "array", "items": {"type": "object"}}


def test_model_context_map_rejects_scalar_entries():
    from core.orchestration.agentic_passes import _validate_model_context_map

    context_map = _context_map()
    context_map["entry_points"] = ["main"]

    assert (
        _validate_model_context_map(context_map)
        == "'entry_points' must contain only objects"
    )


def test_map_dispatch_retries_transient_provider_failure(tmp_path, monkeypatch):
    from packages.code_understanding.dispatch.map_dispatch import default_map_dispatch

    result_map = _context_map()
    provider = None

    class TransientProvider(FakeProvider):
        def turn(self, messages, tools, **kwargs):
            nonlocal provider
            provider = self
            if self._calls == 0:
                self._calls += 1
                raise RuntimeError("503 service unavailable")
            return super().turn(messages, tools, **kwargs)

    monkeypatch.setattr(
        "packages.code_understanding.dispatch.map_dispatch.create_provider",
        lambda _model: TransientProvider(
            [FakeTurn(tool_calls=[("submit_context_map", {"context_map": result_map})])]
        ),
    )
    monkeypatch.setattr(
        "packages.code_understanding.dispatch.map_dispatch.time.sleep",
        lambda _delay: None,
    )
    result = default_map_dispatch(_model(), str(tmp_path))

    assert result == result_map
    assert provider is not None
    assert provider._calls == 2


def test_map_dispatch_does_not_retry_nontransient_provider_failure(tmp_path, monkeypatch):
    from packages.code_understanding.dispatch.map_dispatch import default_map_dispatch

    class PermanentProvider(FakeProvider):
        def turn(self, messages, tools, **kwargs):
            raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(
        "packages.code_understanding.dispatch.map_dispatch.create_provider",
        lambda _model: PermanentProvider([]),
    )
    monkeypatch.setattr(
        "packages.code_understanding.dispatch.map_dispatch.time.sleep",
        lambda _delay: pytest.fail("nontransient failure must not retry"),
    )

    result = default_map_dispatch(_model(), str(tmp_path))

    assert result["error"].startswith("RuntimeError: 401 unauthorized")
