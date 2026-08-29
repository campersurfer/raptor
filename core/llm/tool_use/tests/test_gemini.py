"""Tests for ``GeminiProvider.turn`` through the native Google SDK."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# google-genai SDK gate — CI runs without it skip cleanly.
pytest.importorskip("google.genai")
from google.genai import types

from core.llm.config import ModelConfig
from core.llm.providers import GeminiProvider, LLMResponse
from core.llm.tool_use import (
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolDef,
)


def _config(model_name: str = "gemini-2.5-pro") -> ModelConfig:
    return ModelConfig(
        provider="gemini",
        model_name=model_name,
        api_key="test-key",
        timeout=1,
    )


def _echo_tool() -> ToolDef:
    return ToolDef(
        name="echo",
        description="echo input back",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=lambda inp: f"echoed:{inp.get('q', '')}",
    )


def _user(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


class _FakeModels:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response) -> None:
        self.models = _FakeModels(response)


def _provider_with_response(
    text: str,
    *,
    model_name: str = "gemini-2.5-pro",
    input_tokens: int = 4,
    output_tokens: int = 6,
    thinking_tokens: int = 0,
) -> tuple[GeminiProvider, _FakeClient]:
    response = SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thinking_tokens,
        ),
    )
    provider = GeminiProvider(_config(model_name))
    client = _FakeClient(response)
    provider._local.client = client
    return provider, client


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


def test_capabilities_advertise_native_json_path() -> None:
    p = GeminiProvider(_config())
    assert p.supports_tool_use() is True
    assert p.supports_parallel_tools() is False
    assert p.supports_prompt_caching() is False


# ---------------------------------------------------------------------------
# Native JSON turn()
# ---------------------------------------------------------------------------


def test_turn_json_text_response_returns_complete() -> None:
    p, client = _provider_with_response('{"text": "just a plain answer"}')
    out = p.turn(messages=[_user("hi")], tools=[_echo_tool()])

    assert out.stop_reason is StopReason.COMPLETE
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "just a plain answer"
    assert client.models.calls[0]["config"]["response_mime_type"] == "application/json"
    config = client.models.calls[0]["config"]
    assert "response_json_schema" not in config
    schema = config["response_schema"]
    assert schema["type"] == "OBJECT"
    assert {tuple(option["required"]) for option in schema["anyOf"]} == {
        ("tool", "input"),
        ("text",),
    }


def test_turn_without_tools_preserves_text_transport() -> None:
    p = GeminiProvider(_config())
    p.generate = lambda *_args, **_kwargs: LLMResponse(
        content="ordinary completion",
        model="gemini-2.5-pro",
        provider="gemini",
        tokens_used=10,
        cost=0.01,
        finish_reason="stop",
        input_tokens=4,
        output_tokens=6,
    )

    out = p.turn(messages=[_user("hi")], tools=[])

    assert out.stop_reason is StopReason.COMPLETE
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "ordinary completion"
    assert out.cost_usd == 0.01


def test_turn_native_json_response_emits_tool_call() -> None:
    p, client = _provider_with_response(
        '{"tool": "echo", "input": {"q": "x"}}',
        model_name="gemini-2.5-flash",
        thinking_tokens=2,
    )
    out = p.turn(messages=[_user("hi")], tools=[_echo_tool()])

    assert out.stop_reason is StopReason.NEEDS_TOOL_CALL
    assert isinstance(out.content[0], ToolCall)
    assert out.content[0].name == "echo"
    assert out.content[0].input == {"q": "x"}
    assert out.content[0].id.startswith("call_")
    assert out.input_tokens == 4
    assert out.output_tokens == 6
    assert client.models.calls[0]["config"]["response_mime_type"] == "application/json"
    assert client.models.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}
    config = client.models.calls[0]["config"]
    assert "response_json_schema" not in config
    schema = config["response_schema"]
    tool_option = next(
        option for option in schema["anyOf"]
        if option["required"] == ["tool", "input"]
    )
    assert tool_option["properties"]["tool"]["enum"] == ["echo"]
    assert tool_option["properties"]["input"] == {
        "type": "OBJECT",
        "properties": {"q": {"type": "STRING"}},
    }


def test_native_json_schema_binds_each_tool_input() -> None:
    other_tool = ToolDef(
        name="count",
        description="count input",
        input_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
        handler=lambda inp: str(inp["count"]),
    )

    schema = GeminiProvider._tool_response_schema([_echo_tool(), other_tool])
    tool_options = [
        option for option in schema["anyOf"]
        if option["required"] == ["tool", "input"]
    ]
    schemas_by_name = {
        option["properties"]["tool"]["enum"][0]: option["properties"]["input"]
        for option in tool_options
    }

    assert schemas_by_name == {
        "echo": _echo_tool().input_schema,
        "count": other_tool.input_schema,
    }


def test_turn_serializes_terminal_map_input_without_response_json_schema() -> None:
    terminal_map = ToolDef(
        name="submit_context_map",
        description="terminal map",
        input_schema={
            "type": "object",
            "properties": {
                "context_map": {
                    "type": "object",
                    "properties": {
                        "sources": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["sources"],
                    "additionalProperties": False,
                },
            },
            "required": ["context_map"],
            "additionalProperties": False,
        },
        handler=lambda _input: "received",
    )
    provider, client = _provider_with_response(
        '{"tool": "submit_context_map", "input": {"context_map": {"sources": []}}}',
    )

    result = provider.turn(messages=[_user("map")], tools=[terminal_map])

    assert isinstance(result.content[0], ToolCall)
    config = client.models.calls[0]["config"]
    wire_config = types.GenerateContentConfig.model_validate(config).model_dump(
        by_alias=True,
        exclude_none=True,
    )
    assert "responseJsonSchema" not in wire_config
    assert wire_config["responseSchema"] == config["response_schema"]
    assert "response_json_schema" not in config
    option = next(
        item for item in config["response_schema"]["anyOf"]
        if item["properties"]["tool"]["enum"] == ["submit_context_map"]
    )
    assert option["properties"]["input"] == {
        "type": "OBJECT",
        "properties": {
            "context_map": {
                "type": "OBJECT",
                "properties": {
                    "sources": {"type": "ARRAY", "items": {"type": "OBJECT"}},
                },
                "required": ["sources"],
                "additionalProperties": False,
            },
        },
        "required": ["context_map"],
        "additionalProperties": False,
    }
def test_turn_malformed_json_remains_an_ordinary_completion() -> None:
    p, _ = _provider_with_response("not JSON")
    out = p.turn(messages=[_user("hi")], tools=[_echo_tool()])

    assert out.stop_reason is StopReason.COMPLETE
    assert isinstance(out.content[0], TextBlock)
    assert out.content[0].text == "not JSON"


def test_turn_preserves_native_thinking_cost() -> None:
    p, _ = _provider_with_response(
        '{"tool": "echo", "input": {"q": "x"}}',
        input_tokens=100,
        output_tokens=50,
        thinking_tokens=25,
    )
    out = p.turn(messages=[_user("hi")], tools=[_echo_tool()])

    expected_cost = p._calculate_cost_split(100, 50, 25)
    assert out.cost_usd == expected_cost
    assert p.compute_cost(out) == expected_cost
    assert p.call_count == 1
    assert p.total_input_tokens == 100
    assert p.total_output_tokens == 50
