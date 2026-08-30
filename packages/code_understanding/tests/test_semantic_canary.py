"""Offline coverage for the semantic map canary."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from core.llm.config import ModelConfig
from core.llm.tool_use.types import (
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    TurnResponse,
)
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
    "source_verified",
    "sink_verified",
    "semantic_relation_verified",
    "section_counts",
}


def _model() -> ModelConfig:
    return ModelConfig(provider="gemini", model_name="test-model")


def _canonical_empty_map() -> dict[str, Any]:
    """Return the canonical result when no attack path is proven."""
    return {
        "sources": [],
        "sinks": [],
        "trust_boundaries": [],
        "meta": {"language": ["C++"]},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [],
    }


def _line_of(content: str, fragment: str) -> int:
    return content[: content.index(fragment)].count("\n") + 1


def _canonical_map_from_fixture_content(content: str) -> dict[str, Any]:
    """Derive terminal evidence only from the returned fixture content."""
    source_statement = 'return argc > 1 ? argv[1] : "";'
    sink_statement = "std::system(command.c_str());"
    if source_statement not in content or sink_statement not in content:
        return _canonical_empty_map()

    source_match = re.search(
        r"std::string\s+(canary_source_[0-9a-f]+)\(int argc, char\*\* argv\)",
        content,
    )
    sink_match = re.search(
        r"void\s+(canary_sink_[0-9a-f]+)\(const std::string& command\)",
        content,
    )
    relation_match = re.search(
        r"void\s+(canary_relation_[0-9a-f]+)\(int argc, char\*\* argv\)",
        content,
    )
    assert source_match and sink_match and relation_match
    source = source_match.group(1)
    sink = sink_match.group(1)
    relation = relation_match.group(1)
    source_line = _line_of(content, source_statement)
    sink_line = _line_of(content, sink_statement)
    relation_line = _line_of(content, f"void {relation}(")
    return {
        "sources": [{
            "type": "cli_arg",
            "entry": f"fixture.cpp:{source_line} {source} argv[1]",
            "trust_level": "attacker_controlled",
        }],
        "sinks": [{
            "type": "shell_exec",
            "location": f"fixture.cpp:{sink_line}",
        }],
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
            "operation": sink_statement,
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


def _canonical_fixture_map() -> dict[str, Any]:
    return _canonical_map_from_fixture_content(semantic_canary._fresh_fixture().content)


def _private_canary_map() -> dict[str, Any]:
    """The retired private shape must not qualify the production path."""
    return {
        "sources": [{"name": f"canary_source_{_TOKEN}"}],
        "sinks": [{"name": f"canary_sink_{_TOKEN}"}],
        "trust_boundaries": [],
        "meta": {"language": "C++"},
        "entry_points": [],
        "sink_details": [],
        "boundary_details": [],
        "unchecked_flows": [{
            "source": f"canary_source_{_TOKEN}",
            "sink": f"canary_sink_{_TOKEN}",
            "relation": f"canary_relation_{_TOKEN}",
        }],
    }


def _legacy_benign_fixture() -> semantic_canary._Fixture:
    return semantic_canary._Fixture(
        source="legacy_source",
        sink="legacy_sink",
        relation="legacy_relation",
        source_line=4,
        sink_line=8,
        relation_line=11,
        content=(
            "#include <string>\n\n"
            "std::string legacy_source(int argc, char** argv) {\n"
            "    return \"\";\n"
            "}\n\n"
            "void legacy_sink(const std::string& command) {\n"
            "    (void)command;\n"
            "}\n\n"
            "void legacy_relation(int argc, char** argv) {\n"
            "    legacy_sink(legacy_source(argc, argv));\n"
            "}\n"
        ),
    )


@pytest.fixture(autouse=True)
def _fixed_canary_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: _TOKEN)


@dataclass
class _Turn:
    calls: list[ToolCall]
    reason: StopReason = StopReason.NEEDS_TOOL_CALL
    cost: float = 0.0


def _response(
    calls: list[ToolCall],
    *,
    reason: StopReason = StopReason.NEEDS_TOOL_CALL,
    cost: float = 0.0,
) -> TurnResponse:
    return TurnResponse(
        content=calls,
        stop_reason=reason,
        input_tokens=1,
        output_tokens=1,
        cost_usd=cost,
    )


class _Provider:
    def __init__(
        self,
        turns: list[_Turn],
        *,
        first_error: Exception | None = None,
    ) -> None:
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
        return response.cost_usd or 0.0

    def turn(self, messages, _tools, **kwargs) -> TurnResponse:
        self.calls += 1
        self.messages.append(list(messages))
        self.systems.append(kwargs["system"])
        if self._first_error is not None:
            error = self._first_error
            self._first_error = None
            raise error
        turn = next(self._turns)
        return _response(turn.calls, reason=turn.reason, cost=turn.cost)


class _CanonicalSimulator:
    """Offline model simulator that derives its map after the read result."""

    def __init__(self) -> None:
        self.calls = 0
        self.systems: list[str] = []
        self.messages: list[list[Message]] = []
        self.fixture_contents: list[str] = []

    def supports_tool_use(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        return False

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def context_window(self) -> int:
        return 200_000

    def compute_cost(self, response: TurnResponse) -> float:
        return response.cost_usd or 0.0

    def turn(self, messages, _tools, **kwargs) -> TurnResponse:
        self.calls += 1
        self.messages.append(list(messages))
        self.systems.append(kwargs["system"])
        if self.calls == 1:
            return _response([
                ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"}),
            ])
        content = self._read_fixture_content(messages)
        self.fixture_contents.append(content)
        return _response([
            ToolCall(
                id="terminal",
                name="submit_context_map",
                input={"context_map": _canonical_map_from_fixture_content(content)},
            ),
        ])

    @staticmethod
    def _read_fixture_content(messages: list[Message]) -> str:
        for message in messages:
            for block in message.content:
                if not isinstance(block, ToolResult) or block.tool_use_id != "read":
                    continue
                start = block.content.find("{")
                end = block.content.rfind("}")
                assert start >= 0 and end >= start
                payload = json.loads(block.content[start : end + 1])
                assert isinstance(payload, dict)
                content = payload.get("content")
                assert isinstance(content, str)
                return content
        raise AssertionError("canonical simulator did not receive fixture content")


def _terminal_turns(
    count: int = 1,
    *,
    context_map: dict[str, Any] | None = None,
    read_path: str = "fixture.cpp",
) -> list[_Turn]:
    context_map = _canonical_fixture_map() if context_map is None else context_map
    return [
        _Turn([ToolCall(id="read", name="read_file", input={"path": read_path})]),
        _Turn([
            ToolCall(
                id=f"terminal-{index}",
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


def _first_user_text(provider: _Provider | _CanonicalSimulator) -> str:
    for message in provider.messages[0]:
        for content in message.content:
            if isinstance(content, TextBlock):
                return content.text
    raise AssertionError("missing canary user message")


def test_fixture_contains_the_required_cli_to_shell_exec_flow():
    fixture = semantic_canary._fresh_fixture()

    assert 'argc > 1 ? argv[1] : ""' in fixture.content
    assert "std::system(command.c_str());" in fixture.content
    assert f"{fixture.sink}({fixture.source}(argc, argv));" in fixture.content


def test_canonical_simulator_returns_empty_map_for_benign_fixture():
    assert _canonical_map_from_fixture_content(_legacy_benign_fixture().content) == (
        _canonical_empty_map()
    )


def test_benign_fixture_cannot_qualify_the_canary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(semantic_canary, "_fresh_fixture", _legacy_benign_fixture)
    provider = _CanonicalSimulator()

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["fixture_read"] is True
    assert result.attestation["source_verified"] is False
    assert result.attestation["sink_verified"] is False


def test_private_canary_shape_is_not_the_canonical_production_map():
    import jsonschema

    from packages.code_understanding.dispatch.map_dispatch import build_context_map_schema

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance=_private_canary_map(),
            schema=build_context_map_schema(),
        )


def test_private_canary_map_cannot_qualify_the_production_path():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(
            _terminal_turns(context_map=_private_canary_map())
        ),
    )

    _assert_failure(result, "map_validation")


def test_canary_attests_only_bounded_non_sensitive_evidence():
    provider = _CanonicalSimulator()
    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)
    fixture = semantic_canary._fresh_fixture()

    assert result.success is True
    assert set(result.attestation) == _SUCCESS_FIELDS
    assert result.attestation["schema_version"] == 2
    assert result.attestation["status"] == "passed"
    assert len(result.attestation["fixture_sha256"]) == 64
    assert len(result.attestation["request_schema_sha256"]) == 64
    assert result.attestation["provider"] == "gemini"
    assert result.attestation["model"] == "test-model"
    assert result.attestation["terminal_call_count"] == 1
    assert result.attestation["provider_turn_count"] == 2
    assert result.attestation["fixture_read"] is True
    assert result.attestation["language_verified"] is True
    assert result.attestation["source_verified"] is True
    assert result.attestation["sink_verified"] is True
    assert result.attestation["semantic_relation_verified"] is True
    assert result.attestation["section_counts"] == {
        "sources": 1,
        "sinks": 1,
        "trust_boundaries": 0,
        "entry_points": 1,
        "sink_details": 1,
        "boundary_details": 0,
        "unchecked_flows": 1,
    }
    assert provider.fixture_contents == [fixture.content]
    rendered = repr(result.attestation)
    for identifier in (fixture.source, fixture.sink, fixture.relation):
        assert identifier not in rendered
    assert "fixture.cpp" not in rendered


def test_canary_prompt_never_discloses_hidden_fixture_identifiers():
    provider = _CanonicalSimulator()
    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)
    fixture = semantic_canary._fresh_fixture()

    assert result.success is True
    initial_system = provider.systems[0]
    initial_user = _first_user_text(provider)
    for identifier in (fixture.source, fixture.sink, fixture.relation):
        assert identifier not in initial_system
        assert identifier not in initial_user
    assert fixture.content not in initial_system
    assert fixture.content not in initial_user


def test_canary_has_no_compiler_or_execution_capability(tmp_path):
    from packages.code_understanding.dispatch.map_dispatch import _build_tools
    from packages.code_understanding.dispatch.tools import SandboxedTools

    tree = ast.parse(inspect.getsource(semantic_canary))
    calls = {
        _dotted_call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    tools = _build_tools(SandboxedTools.for_repo(tmp_path))

    assert {"subprocess.run", "subprocess.Popen", "os.system", "os.popen", "compile"}.isdisjoint(calls)
    assert {tool.name for tool in tools} == {
        "read_file",
        "grep",
        "glob_files",
        "submit_context_map",
    }


def _dotted_call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def test_canary_rejects_terminal_map_without_fixture_read():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider([_Turn([
            ToolCall(
                id="terminal",
                name="submit_context_map",
                input={"context_map": _canonical_fixture_map()},
            ),
        ])]),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["fixture_read"] is False
    assert result.attestation["language_verified"] is False
    assert result.attestation["source_verified"] is False
    assert result.attestation["sink_verified"] is False
    assert result.attestation["semantic_relation_verified"] is False


def test_canary_rejects_failed_fixture_read():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(read_path="missing.cpp")),
    )

    _assert_failure(result, "terminal_contract")
    assert result.attestation["fixture_read"] is False


def test_canary_requires_exact_fixture_read_result():
    fixture = semantic_canary._fresh_fixture()
    read_result = SimpleNamespace(
        is_error=False,
        content=json.dumps({
            "path": "fixture.cpp",
            "content": fixture.content + "tampered",
            "truncated": False,
        }),
    )

    assert not semantic_canary._matches_fixture_read_result(
        read_result,
        "fixture.cpp",
        fixture.content,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda context_map: context_map["sources"][0].update(
            entry="fixture.cpp:5 wrong-source argv[1]"
        ),
        lambda context_map: context_map["sink_details"][0].update(name="wrong-sink"),
        lambda context_map: context_map["entry_points"][0].update(name="wrong-relation"),
    ],
)
def test_canary_rejects_wrong_hidden_semantic_identifier(mutate):
    context_map = _canonical_fixture_map()
    mutate(context_map)

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"].replace(" ", "0 ", 1)
        ),
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"].replace(
                context_map["sources"][0]["entry"].split()[1],
                "prefix_" + context_map["sources"][0]["entry"].split()[1],
            )
        ),
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"].replace(
                context_map["sources"][0]["entry"].split()[1],
                context_map["sources"][0]["entry"].split()[1] + "é",
            )
        ),
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"].replace(
                context_map["sources"][0]["entry"].split()[1],
                context_map["sources"][0]["entry"].split()[1] + chr(0x0301),
            )
        ),
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"].replace(
                context_map["sources"][0]["entry"].split()[1],
                context_map["sources"][0]["entry"].split()[1] + "\\u0301",
            )
        ),
        lambda context_map: context_map["sources"][0].update(
            entry=context_map["sources"][0]["entry"] + " fabricated"
        ),
        lambda context_map: context_map["sinks"][0].update(
            location=context_map["sinks"][0]["location"] + "0"
        ),
        lambda context_map: context_map["entry_points"][0].update(
            name="prefix_" + context_map["entry_points"][0]["name"]
        ),
        lambda context_map: context_map["entry_points"][0].update(
            name="different_relation",
            notes=context_map["entry_points"][0]["name"],
        ),
        lambda context_map: context_map["sink_details"][0].update(
            name="prefix_" + context_map["sink_details"][0]["name"]
        ),
        lambda context_map: context_map["sink_details"][0].update(
            name="different_sink",
            notes=context_map["sink_details"][0]["name"],
        ),
    ],
)
def test_canary_rejects_noncanonical_evidence_anchors(mutate):
    context_map = _canonical_fixture_map()
    mutate(context_map)

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")


@pytest.mark.parametrize(
    "field",
    ["entry_point", "sink"],
)
def test_canary_rejects_unresolved_flow_reference(field: str):
    context_map = _canonical_fixture_map()
    context_map["unchecked_flows"][0][field] = "UNKNOWN-ID"

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")


def test_canary_rejects_non_attacker_controlled_source():
    context_map = _canonical_fixture_map()
    context_map["sources"][0]["trust_level"] = "internal_value"

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["source_verified"] is False


def test_canary_rejects_non_dangerous_sink():
    context_map = _canonical_fixture_map()
    context_map["sinks"][0]["type"] = "logging"
    context_map["sink_details"][0]["type"] = "logging"

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["sink_verified"] is False


def test_canary_rejects_declared_validation_boundary():
    context_map = _canonical_fixture_map()
    context_map["trust_boundaries"] = [{
        "boundary": "command validation",
        "check": "allowlist",
    }]

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "semantic_evidence")


def test_canary_rejects_scalar_language_label():
    context_map = _canonical_fixture_map()
    context_map["meta"]["language"] = "C++"

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "map_validation")


def test_canary_rejects_empty_vulnerable_map():
    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(
            _terminal_turns(context_map=_canonical_empty_map())
        ),
    )

    _assert_failure(result, "semantic_evidence")
    assert result.attestation["source_verified"] is False
    assert result.attestation["sink_verified"] is False


def test_canary_rejects_undeclared_context_map_fields_without_attesting_them():
    context_map = _canonical_fixture_map()
    context_map["fixture_source_text"] = "do-not-emit"

    result = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _Provider(_terminal_turns(context_map=context_map)),
    )

    _assert_failure(result, "map_validation")
    assert "fixture_source_text" not in repr(result.attestation)
    assert "do-not-emit" not in repr(result.attestation)


def test_canary_rejects_oversized_terminal_payload():
    context_map = _canonical_fixture_map()
    context_map["sources"][0]["notes"] = "x" * 16_384

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


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(id="unknown", name="unknown_tool", input={}),
        ToolCall(id="bad-read", name="read_file", input={"path": 1}),
    ],
)
def test_canary_rejects_unknown_or_malformed_tool_calls(call: ToolCall):
    provider = _Provider([
        _Turn([call]),
        _Turn([
            ToolCall(
                id="terminal",
                name="submit_context_map",
                input={"context_map": _canonical_fixture_map()},
            ),
        ]),
    ])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")


def test_canary_rejects_prose_only_completion():
    provider = _Provider([_Turn([], reason=StopReason.COMPLETE)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")


def test_canary_uses_fresh_fixture_identifiers_each_run(monkeypatch: pytest.MonkeyPatch):
    tokens = iter(("1" * 24, "2" * 24))
    monkeypatch.setattr(semantic_canary, "_fresh_identifier", lambda: next(tokens))

    first = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _CanonicalSimulator(),
    )
    second = run_semantic_canary(
        _model(),
        provider_factory=lambda _: _CanonicalSimulator(),
    )

    assert first.success is True
    assert second.success is True
    assert first.attestation["fixture_sha256"] != second.attestation["fixture_sha256"]


def test_canary_enforces_cost_quota():
    provider = _Provider([
        _Turn(
            [ToolCall(id="read", name="read_file", input={"path": "fixture.cpp"})],
            cost=1.0,
        ),
    ])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "cost_limit")


def test_canary_enforces_iteration_limit():
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN) for _ in range(3)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")
    assert provider.calls == 3


def test_canary_enforces_wall_clock_limit(monkeypatch: pytest.MonkeyPatch):
    moments = iter((0.0, 0.0, 31.0))
    loop_module = importlib.import_module("core.llm.tool_use.loop")
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: next(moments))
    provider = _Provider([_Turn([], reason=StopReason.PAUSE_TURN)])

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "terminal_contract")
    assert provider.calls == 1


@pytest.mark.parametrize(
    ("error", "failure_class"),
    [
        (RuntimeError("401 unauthorized"), "auth"),
        (RuntimeError("429 quota exhausted"), "quota"),
        (TimeoutError("request timeout"), "timeout"),
        (RuntimeError("400 schema malformed"), "schema"),
        (RuntimeError("503 transient"), "transport"),
    ],
)
def test_provider_failures_never_qualify(error: Exception, failure_class: str):
    provider = _Provider(_terminal_turns(), first_error=error)

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, failure_class)
    assert provider.calls == 1


def test_canary_never_retries_transport_failure():
    provider = _Provider(
        _terminal_turns(),
        first_error=RuntimeError("503 transient"),
    )

    result = run_semantic_canary(_model(), provider_factory=lambda _: provider)

    _assert_failure(result, "transport")
    assert provider.calls == 1


def test_semantic_canary_cli_rejects_unapproved_model_as_bounded_json(capsys):
    exit_code = semantic_canary.main(["--model", "other-model", "--format", "json"])

    assert exit_code == 1
    attestation = json.loads(capsys.readouterr().out)
    assert set(attestation) == _SUCCESS_FIELDS | {"failure_class"}
    assert attestation["failure_class"] == "unsupported_model"


def test_semantic_canary_cli_uses_resolved_model(monkeypatch: pytest.MonkeyPatch, capsys):
    expected = semantic_canary.SemanticCanaryResult(True, {
        "schema_version": 2,
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
        "source_verified": True,
        "sink_verified": True,
        "semantic_relation_verified": True,
        "section_counts": {key: 0 for key in semantic_canary._SECTION_KEYS},
    })
    monkeypatch.setattr(semantic_canary, "resolve_semantic_canary_model", lambda _: _model())
    monkeypatch.setattr(semantic_canary, "run_semantic_canary", lambda _: expected)

    exit_code = semantic_canary.main(["--model", "gemini-2.5-flash", "--format", "json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected.attestation
