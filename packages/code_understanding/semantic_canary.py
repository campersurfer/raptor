"""Isolated, offline-testable semantic canary for constrained terminal maps.

Fresh main has no context-map dispatch path, so this module deliberately has no
lifecycle wiring. It preserves the bounded terminal contract without claiming
runtime qualification for the removed architecture.
"""

from __future__ import annotations

import hashlib
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.llm.config import ModelConfig
from core.llm.providers import create_provider
from core.llm.tool_use import CacheControl, ContextPolicy, ToolDef, ToolUseLoop
from core.llm.tool_use.types import ToolCallDispatched, ToolCallReturned
from packages.code_understanding.dispatch.tools import SandboxedTools

_MAX_ITERATIONS = 3
_MAX_COST_USD = 0.05
_MAX_SECONDS = 30.0
_TOOL_TIMEOUT_S = 5.0
_ATTESTATION_ID_MAX_CHARS = 128


@dataclass(frozen=True)
class _Fixture:
    source: str
    sink: str
    relation: str
    content: str


@dataclass(frozen=True)
class SemanticCanaryResult:
    success: bool
    attestation: dict[str, Any]


def _fresh_identifier() -> str:
    return secrets.token_hex(12)


def _fresh_fixture() -> _Fixture:
    """Build opaque per-run identifiers; attest only a content digest."""
    identifier = _fresh_identifier()
    source = f"canary_source_{identifier}"
    sink = f"canary_sink_{identifier}"
    relation = f"canary_relation_{identifier}"
    return _Fixture(
        source=source,
        sink=sink,
        relation=relation,
        content=(
            "#include <string>\n\n"
            f"std::string {source}(const std::string& input) {{\n"
            "    return input;\n"
            "}\n\n"
            f"void {sink}(const std::string& value) {{\n"
            "    (void)value;\n"
            "}\n\n"
            f"void {relation}(const std::string& input) {{\n"
            f"    {sink}({source}(input));\n"
            "}\n"
        ),
    )


def _terminal_schema() -> dict[str, Any]:
    """Return the closed terminal input contract used by the canary."""
    named_entry = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    flow_entry = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "sink": {"type": "string"},
            "relation": {"type": "string"},
        },
        "required": ["source", "sink", "relation"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "context_map": {
                "type": "object",
                "properties": {
                    "meta": {
                        "type": "object",
                        "properties": {"language": {"type": "string"}},
                        "required": ["language"],
                        "additionalProperties": False,
                    },
                    "sources": {"type": "array", "items": named_entry},
                    "sinks": {"type": "array", "items": named_entry},
                    "unchecked_flows": {"type": "array", "items": flow_entry},
                },
                "required": ["meta", "sources", "sinks", "unchecked_flows"],
                "additionalProperties": False,
            },
        },
        "required": ["context_map"],
        "additionalProperties": False,
    }


def _validate_terminal_context_map(payload: Any) -> dict[str, Any] | None:
    """Reject anything outside the exact closed canary terminal shape."""
    if not isinstance(payload, dict) or set(payload) != {"context_map"}:
        return None
    context_map = payload["context_map"]
    if not isinstance(context_map, dict) or set(context_map) != {
        "meta", "sources", "sinks", "unchecked_flows",
    }:
        return None
    meta = context_map["meta"]
    if not isinstance(meta, dict) or set(meta) != {"language"}:
        return None
    if not isinstance(meta["language"], str) or not meta["language"].strip():
        return None
    for key in ("sources", "sinks"):
        entries = context_map[key]
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict)
            or set(entry) != {"name"}
            or not isinstance(entry["name"], str)
            or not entry["name"].strip()
            for entry in entries
        ):
            return None
    flows = context_map["unchecked_flows"]
    if not isinstance(flows, list) or any(
        not isinstance(flow, dict)
        or set(flow) != {"source", "sink", "relation"}
        or any(not isinstance(flow[field], str) or not flow[field].strip()
               for field in ("source", "sink", "relation"))
        for flow in flows
    ):
        return None
    return context_map


def _build_tools(sandbox: SandboxedTools) -> list[ToolDef]:
    """Expose only a closed read contract and the closed terminal map."""
    def submit(payload: dict[str, Any]) -> str:
        if _validate_terminal_context_map(payload) is None:
            raise ValueError("invalid semantic canary terminal contract")
        return '{"received":true}'

    return [
        ToolDef(
            name="read_file",
            description="Read the isolated fixture by repo-relative path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda args: sandbox.read_file(
                args["path"], max_lines=args.get("max_lines"),
            ),
        ),
        ToolDef(
            name="submit_context_map",
            description="TERMINAL — submit the exact derived semantic map once.",
            input_schema=_terminal_schema(),
            handler=submit,
        ),
    ]


def _has_exact_semantic_evidence(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    return (
        any(entry["name"] == fixture.source for entry in context_map["sources"])
        and any(entry["name"] == fixture.sink for entry in context_map["sinks"])
        and any(
            flow["source"] == fixture.source
            and flow["sink"] == fixture.sink
            and flow["relation"] == fixture.relation
            for flow in context_map["unchecked_flows"]
        )
    )


def _bounded_identity(value: str) -> str:
    return value[:_ATTESTATION_ID_MAX_CHARS]


def run_semantic_canary(
    model: ModelConfig,
    *,
    provider_factory: Callable[[ModelConfig], Any] = create_provider,
) -> SemanticCanaryResult:
    """Exercise the isolated terminal contract without lifecycle integration."""
    if not isinstance(model, ModelConfig):
        return SemanticCanaryResult(False, {"status": "failed", "reason": "invalid-model"})
    try:
        provider = provider_factory(model)
    except Exception:
        return SemanticCanaryResult(False, {"status": "failed", "reason": "provider-init"})

    with tempfile.TemporaryDirectory(prefix="raptor-semantic-canary-") as directory:
        fixture_root = Path(directory)
        fixture = _fresh_fixture()
        fixture_path = "fixture.cpp"
        (fixture_root / fixture_path).write_text(fixture.content, encoding="utf-8")
        terminal_calls = 0
        read_call_ids: set[str] = set()
        fixture_read = False

        def events(event: Any) -> None:
            nonlocal terminal_calls, fixture_read
            if isinstance(event, ToolCallDispatched):
                if event.call.name == "submit_context_map":
                    terminal_calls += 1
                elif event.call.name == "read_file" and event.call.input.get("path") == fixture_path:
                    read_call_ids.add(event.call.id)
            elif isinstance(event, ToolCallReturned) and event.call_id in read_call_ids:
                fixture_read = fixture_read or not event.result.is_error

        try:
            result = ToolUseLoop(
                provider=provider,
                tools=_build_tools(SandboxedTools.for_repo(fixture_root)),
                system=(
                    "Read fixture.cpp before submitting the terminal map. Derive the exact "
                    "source, sink, and relation identifiers from its contents."
                ),
                terminal_tool="submit_context_map",
                max_iterations=_MAX_ITERATIONS,
                max_cost_usd=_MAX_COST_USD,
                max_seconds=_MAX_SECONDS,
                tool_timeout_s=_TOOL_TIMEOUT_S,
                context_policy=ContextPolicy.RAISE,
                cache_control=CacheControl(system=True, tools=True),
                terminate_on_handler_error=False,
                events=events,
            ).run(f"Inspect {fixture_path}; it is C++.")
        except Exception:
            return SemanticCanaryResult(False, {"status": "failed", "reason": "loop-failure"})

    context_map = _validate_terminal_context_map(result.terminal_tool_input)
    if terminal_calls != 1 or result.terminated_by != "terminal_tool":
        return SemanticCanaryResult(False, {"status": "failed", "reason": "terminal-contract"})
    if (
        context_map is None
        or not fixture_read
        or context_map["meta"]["language"].strip().casefold() not in {"c++", "cpp", "cxx"}
        or not _has_exact_semantic_evidence(context_map, fixture)
    ):
        return SemanticCanaryResult(False, {"status": "failed", "reason": "semantic-evidence"})
    return SemanticCanaryResult(
        True,
        {
            "status": "passed",
            "fixture_sha256": hashlib.sha256(fixture.content.encode("utf-8")).hexdigest(),
            "provider": _bounded_identity(model.provider),
            "model": _bounded_identity(model.model_name),
            "terminal_calls": terminal_calls,
            "fixture_read": True,
            "language_verified": True,
            "semantic_relation_verified": True,
        },
    )
