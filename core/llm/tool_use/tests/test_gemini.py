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
def _wire_config(config: dict) -> dict:
    return types.GenerateContentConfig.model_validate(config).model_dump(
        by_alias=True,
        exclude_none=True,
    )


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
    assert "response_schema" not in config
    assert "response_json_schema" in config
    schema = config["response_json_schema"]
    assert set(schema) == {"anyOf"}
    assert {tuple(option["required"]) for option in schema["anyOf"]} == {
        ("tool", "input"),
        ("text",),
    }
    wire_config = _wire_config(config)
    assert "responseSchema" not in wire_config
    assert wire_config["responseJsonSchema"] == schema


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
    assert "response_schema" not in config
    schema = config["response_json_schema"]
    tool_option = next(
        option for option in schema["anyOf"]
        if option["required"] == ["tool", "input"]
    )
    assert tool_option["properties"]["tool"]["enum"] == ["echo"]
    assert tool_option["properties"]["input"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
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

    echo_tool = _echo_tool()
    tools = [echo_tool, other_tool]
    schema = GeminiProvider._tool_response_schema(tools)
    assert set(schema) == {"anyOf"}
    tool_options = [
        option for option in schema["anyOf"]
        if option["required"] == ["tool", "input"]
    ]
    assert [option["properties"]["tool"]["enum"] for option in tool_options] == [
        ["echo"], ["count"],
    ]
    assert [option["properties"]["input"] for option in tool_options] == [
        echo_tool.input_schema, other_tool.input_schema,
    ]
    assert all(
        option["type"] == "object"
        and option["additionalProperties"] is False
        for option in tool_options
    )
    text_option = next(option for option in schema["anyOf"] if option["required"] == ["text"])
    assert text_option == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

def test_turn_serializes_terminal_map_input_with_response_json_schema() -> None:
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
    wire_config = _wire_config(config)
    assert "responseSchema" not in wire_config
    assert wire_config["responseJsonSchema"] == config["response_json_schema"]
    assert "response_schema" not in config
    option = next(
        item for item in config["response_json_schema"]["anyOf"]
        if item["properties"]["tool"]["enum"] == ["submit_context_map"]
    )
    assert option["properties"]["input"] == terminal_map.input_schema


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


def test_turn_emits_raw_response_json_schema_without_forbidden_root_type() -> None:
    provider, client = _provider_with_response('{"tool": "echo", "input": {"q": "x"}}')

    provider.turn(messages=[_user("hi")], tools=[_echo_tool()])

    config = client.models.calls[0]["config"]
    wire_config = _wire_config(config)
    assert "response_schema" not in config
    assert "response_json_schema" in config
    assert "responseSchema" not in wire_config
    schema = wire_config["responseJsonSchema"]
    assert set(schema) == {"anyOf"}


def test_duplicate_tool_names_fail_closed() -> None:
    duplicate = ToolDef(
        name="echo",
        description="duplicate",
        input_schema={"type": "object"},
        handler=lambda _input: "unused",
    )

    with pytest.raises(ValueError, match="duplicate tool name"):
        GeminiProvider._tool_response_schema([_echo_tool(), duplicate])


def test_unknown_tool_response_fails_closed() -> None:
    provider, _ = _provider_with_response('{"tool": "unknown", "input": {}}')

    response = provider.turn(messages=[_user("hi")], tools=[_echo_tool()])

    assert response.stop_reason is StopReason.COMPLETE
    assert isinstance(response.content[0], TextBlock)


def test_known_tool_with_malformed_input_fails_closed() -> None:
    provider, _ = _provider_with_response('{"tool": "echo", "input": {"q": 1}}')

    response = provider.turn(messages=[_user("hi")], tools=[_echo_tool()])

    assert response.stop_reason is StopReason.COMPLETE
    assert isinstance(response.content[0], TextBlock)


def test_terminal_tool_prose_cannot_masquerade_as_a_call() -> None:
    terminal_map = ToolDef(
        name="submit_context_map",
        description="terminal map",
        input_schema={"type": "object"},
        handler=lambda _input: "unused",
    )
    provider, _ = _provider_with_response("submit_context_map completed")

    response = provider.turn(messages=[_user("map")], tools=[terminal_map])

    assert response.stop_reason is StopReason.COMPLETE
    assert isinstance(response.content[0], TextBlock)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, float("nan"), "NaN", None],
)
def test_direct_client_rejects_invalid_timeout_values(monkeypatch, timeout) -> None:
    import core.llm.providers as providers_module

    monkeypatch.setattr(providers_module._genai_module, "Client", lambda **_kwargs: object())
    provider = GeminiProvider(
        ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="test-key",
            timeout=timeout,
        )
    )

    with pytest.raises(ValueError, match="positive finite"):
        _ = provider.client


def test_direct_client_uses_model_timeout_in_milliseconds(monkeypatch, caplog) -> None:
    import core.llm.providers as providers_module

    captured = {}
    secret = "gemini-direct-client-test-secret"

    def build_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(providers_module._genai_module, "Client", build_client)
    provider = GeminiProvider(
        ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key=secret,
            timeout=1.234,
        )
    )

    _ = provider.client

    assert captured["api_key"] == secret
    assert captured["http_options"].timeout == 1234
    assert secret not in caplog.text


def test_dispatcher_client_preserves_isolated_transport_and_timeout(monkeypatch) -> None:
    import httpx
    import core.llm.dispatcher.client as dispatcher_client
    import core.llm.providers as providers_module

    captured = {}
    client_kwargs = {}
    custom_client = httpx.Client()

    def make_gemini_base_url(**kwargs):
        captured.update(kwargs)
        return "http://_/gemini", custom_client

    def build_client(**kwargs):
        client_kwargs.update(kwargs)
        return object()

    monkeypatch.setenv("RAPTOR_LLM_SOCKET", "/tmp/raptor-test.sock")
    monkeypatch.setattr(dispatcher_client, "make_gemini_base_url", make_gemini_base_url)
    monkeypatch.setattr(providers_module._genai_module, "Client", build_client)
    provider = GeminiProvider(
        ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="must-not-reach-dispatcher-client",
            timeout=1.234,
        )
    )
    try:
        _ = provider.client
    finally:
        custom_client.close()

    assert captured == {"timeout": pytest.approx(1.234)}
    assert client_kwargs["api_key"] == "dummy-not-used"
    assert client_kwargs["http_options"].base_url == "http://_/gemini"
    assert client_kwargs["http_options"].httpx_client is custom_client
    assert client_kwargs["http_options"].timeout == 1234


def test_dispatcher_factory_passes_timeout_to_custom_httpx_client(monkeypatch) -> None:
    import core.llm.dispatcher.client as dispatcher_client

    captured = {}
    custom_client = object()

    monkeypatch.setattr(
        dispatcher_client,
        "_resolve_socket_and_token",
        lambda _socket, _token: ("/tmp/raptor-test.sock", "test-token"),
    )
    monkeypatch.setattr(
        dispatcher_client,
        "_make_httpx_client",
        lambda socket, token, *, timeout: captured.update(
            socket=socket,
            token=token,
            timeout=timeout,
        ) or custom_client,
    )

    base_url, returned_client = dispatcher_client.make_gemini_base_url(timeout=1.234)

    assert base_url == "http://_/gemini"
    assert returned_client is custom_client
    assert captured == {
        "socket": "/tmp/raptor-test.sock",
        "token": "test-token",
        "timeout": pytest.approx(1.234),
    }
def test_dispatcher_httpx_client_bounds_all_request_phases() -> None:
    from core.llm.dispatcher.client import _make_httpx_client

    client = _make_httpx_client(
        "/tmp/raptor-test.sock", "test-token", timeout=1.234
    )
    try:
        assert client.timeout.connect == pytest.approx(5.0)
        assert client.timeout.read == pytest.approx(1.234)
        assert client.timeout.write == pytest.approx(1.234)
        assert client.timeout.pool == pytest.approx(1.234)
    finally:
        client.close()
