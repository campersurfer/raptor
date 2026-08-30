"""Bounded semantic preflight for the model-backed context-map path."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from core.llm.config import ModelConfig
from core.llm.providers import (
    GeminiProvider,
    LocalToolSchemaDependencyMissing,
    LocalToolSchemaInvalid,
    create_provider,
)
from core.llm.tool_use import (
    CacheControl,
    ContextPolicy,
    CostBudgetExceeded,
    ToolUseLoop,
)
from core.llm.tool_use.types import ToolCallDispatched, ToolCallReturned
from packages.code_understanding.dispatch.map_dispatch import (
    DEFAULT_MAX_TOKENS_PER_TURN,
    _build_tools,
    _format_user_message,
    _validate_terminal_context_map,
)
from packages.code_understanding.dispatch.tools import SandboxedTools
from packages.code_understanding.prompts import MAP_SYSTEM_PROMPT

_MAX_ITERATIONS = 2
_MAX_COST_USD = 0.05
_MAX_SECONDS = 30.0
_TOOL_TIMEOUT_S = 5.0
_ATTESTATION_ID_MAX_CHARS = 128
_ATTESTATION_SCHEMA_VERSION = 2
_MAX_ATTESTATION_SECTION_COUNT = 1_024
_QUALIFICATION_MODEL = "gemini-2.5-flash"
_SECTION_KEYS = (
    "sources",
    "sinks",
    "trust_boundaries",
    "entry_points",
    "sink_details",
    "boundary_details",
    "unchecked_flows",
)


@dataclass(frozen=True)
class _Fixture:
    source: str
    sink: str
    relation: str
    source_line: int
    sink_line: int
    relation_line: int
    content: str


@dataclass(frozen=True)
class SemanticCanaryResult:
    success: bool
    attestation: dict[str, Any]


class _CountingCanaryProvider:
    """Count turns without replaying a transport request."""

    def __init__(
        self,
        provider: Any,
        *,
        lifecycle_events: Callable[..., None] | None = None,
        request_schema_sha256: str = "",
    ) -> None:
        self._provider = provider
        self.turn_count = 0
        self._lifecycle_events = lifecycle_events
        self._request_schema_sha256 = request_schema_sha256

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def _validate_turn_contract(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self.turn_count >= _MAX_ITERATIONS:
            raise RuntimeError("semantic canary exceeded two provider turns")
        if "response_schema" in kwargs:
            raise RuntimeError("semantic canary tool turns must not use response_schema")
        tool_definitions = kwargs.get("tools")
        if tool_definitions is None and len(args) > 1:
            tool_definitions = args[1]
        if (
            not isinstance(tool_definitions, list)
            or _request_schema_sha256(tool_definitions) != self._request_schema_sha256
        ):
            raise RuntimeError("semantic canary tool response_json_schema changed")

    def turn(self, *args: Any, **kwargs: Any) -> Any:
        # GeminiProvider derives response_json_schema from these exact ToolDefs.
        self._validate_turn_contract(args, kwargs)
        self.turn_count += 1
        if self._lifecycle_events is not None:
            metadata = (
                {"request_schema_sha256": self._request_schema_sha256}
                if self.turn_count == 1
                else {}
            )
            self._lifecycle_events("provider_turn_started", **metadata)
        result = self._provider.turn(*args, **kwargs)
        if self._lifecycle_events is not None:
            self._lifecycle_events("provider_turn_completed")
        return result


def _fresh_identifier() -> str:
    return secrets.token_hex(12)


def _fresh_fixture() -> _Fixture:
    """Build an inert, opaque C++ CLI-input-to-shell-exec fixture."""
    identifier = _fresh_identifier()
    source = f"canary_source_{identifier}"
    sink = f"canary_sink_{identifier}"
    relation = f"canary_relation_{identifier}"
    return _Fixture(
        source=source,
        sink=sink,
        relation=relation,
        source_line=5,
        sink_line=9,
        relation_line=12,
        content=(
            "#include <cstdlib>\n"
            "#include <string>\n\n"
            f"std::string {source}(int argc, char** argv) {{\n"
            "    return argc > 1 ? argv[1] : \"\";\n"
            "}\n\n"
            f"void {sink}(const std::string& command) {{\n"
            "    std::system(command.c_str());\n"
            "}\n\n"
            f"void {relation}(int argc, char** argv) {{\n"
            f"    {sink}({source}(argc, argv));\n"
            "}\n\n"
            "int main(int argc, char** argv) {\n"
            f"    {relation}(argc, argv);\n"
            "    return 0;\n"
            "}\n"
        ),
    )


_CPP_LABELS = frozenset({"c++", "cpp", "cxx"})
_CLI_SOURCE_TYPES = frozenset({"cli_arg", "argv", "command_line", "command_line_argument"})
_SHELL_EXEC_TYPES = frozenset({"shell_exec", "command_exec", "command_execution", "system_call"})


def _is_cpp_label(value: Any) -> bool:
    return (
        isinstance(value, list)
        and any(
            isinstance(label, str) and label.strip().casefold() in _CPP_LABELS
            for label in value
        )
    )


def _is_fixture_location(value: Any, line: int) -> bool:
    return value == f"fixture.cpp:{line}"


def _details_match_fixture_line(entry: dict[str, Any], line: int) -> bool:
    return entry.get("file") == "fixture.cpp" and entry.get("line") == line


def _matches_source_entry(value: Any, fixture: _Fixture) -> bool:
    return (
        isinstance(value, str)
        and value.split()
        == [f"fixture.cpp:{fixture.source_line}", fixture.source, "argv[1]"]
    )


def _reference_ids(value: Any) -> set[str] | None:
    if isinstance(value, str) and value:
        return {value}
    if (
        isinstance(value, list)
        and 1 <= len(value) <= 32
        and all(isinstance(item, str) and item for item in value)
    ):
        return set(value)
    return None


def _matching_entry_point_ids(
    context_map: dict[str, Any], fixture: _Fixture
) -> set[str]:
    return {
        entry["id"]
        for entry in context_map["entry_points"]
        if isinstance(entry.get("id"), str)
        and _details_match_fixture_line(entry, fixture.relation_line)
        and entry.get("name") == fixture.relation
    }


def _matching_sink_detail_ids(
    context_map: dict[str, Any], fixture: _Fixture
) -> set[str]:
    return {
        entry["id"]
        for entry in context_map["sink_details"]
        if isinstance(entry.get("id"), str)
        and isinstance(entry.get("type"), str)
        and entry["type"].casefold() in _SHELL_EXEC_TYPES
        and _details_match_fixture_line(entry, fixture.sink_line)
        and entry.get("name") == fixture.sink
    }


def _source_is_verified(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    return any(
        isinstance(entry.get("type"), str)
        and entry["type"].casefold() in _CLI_SOURCE_TYPES
        and entry.get("trust_level") == "attacker_controlled"
        and _matches_source_entry(entry.get("entry"), fixture)
        for entry in context_map["sources"]
    )


def _sink_is_verified(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    top_level_sink = any(
        isinstance(entry.get("type"), str)
        and entry["type"].casefold() in _SHELL_EXEC_TYPES
        and _is_fixture_location(entry.get("location"), fixture.sink_line)
        for entry in context_map["sinks"]
    )
    return top_level_sink and bool(_matching_sink_detail_ids(context_map, fixture))


def _flow_references_are_known(context_map: dict[str, Any]) -> bool:
    entry_ids = {
        entry["id"]
        for entry in context_map["entry_points"]
        if isinstance(entry.get("id"), str) and entry["id"]
    }
    sink_ids = {
        entry["id"]
        for entry in context_map["sink_details"]
        if isinstance(entry.get("id"), str) and entry["id"]
    }
    for flow in context_map["unchecked_flows"]:
        entry_refs = _reference_ids(flow.get("entry_point"))
        sink_refs = _reference_ids(flow.get("sink"))
        if (
            not entry_refs
            or not sink_refs
            or not entry_refs.issubset(entry_ids)
            or not sink_refs.issubset(sink_ids)
        ):
            return False
    return True


def _flow_has_no_declared_boundary(
    context_map: dict[str, Any], entry_ids: set[str], sink_ids: set[str]
) -> bool:
    if context_map["trust_boundaries"] or context_map["boundary_details"]:
        return False
    for entry in context_map["entry_points"]:
        if entry.get("id") in entry_ids and entry.get("auth_required") is True:
            return False
    for entry in context_map["sink_details"]:
        if entry.get("id") in sink_ids and entry.get("trust_boundaries_crossed"):
            return False
    return True


def _has_canonical_semantic_evidence(
    context_map: dict[str, Any], fixture: _Fixture
) -> tuple[bool, bool, bool]:
    source_verified = _source_is_verified(context_map, fixture)
    sink_verified = _sink_is_verified(context_map, fixture)
    entry_ids = _matching_entry_point_ids(context_map, fixture)
    sink_ids = _matching_sink_detail_ids(context_map, fixture)
    linked_flow = any(
        (entry_refs := _reference_ids(flow.get("entry_point")))
        and (sink_refs := _reference_ids(flow.get("sink")))
        and bool(entry_refs & entry_ids)
        and bool(sink_refs & sink_ids)
        for flow in context_map["unchecked_flows"]
    )
    semantic_relation_verified = (
        source_verified
        and sink_verified
        and bool(entry_ids)
        and bool(sink_ids)
        and linked_flow
        and _flow_references_are_known(context_map)
        and _flow_has_no_declared_boundary(context_map, entry_ids, sink_ids)
    )
    return source_verified, sink_verified, semantic_relation_verified


def _matches_fixture_read_result(
    result: Any, fixture_path: str, fixture_content: str
) -> bool:
    if result.is_error:
        return False
    try:
        payload = json.loads(result.content)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("path") == fixture_path
        and payload.get("content") == fixture_content
        and payload.get("truncated") is False
    )
def _bounded_attestation_identity(value: Any) -> str:
    return str(value)[:_ATTESTATION_ID_MAX_CHARS]


def _google_genai_sdk_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "unavailable"


def _request_schema_sha256(tools: list[Any]) -> str:
    schema = GeminiProvider._tool_response_schema(tools)
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _section_counts(context_map: dict[str, Any] | None) -> dict[str, int]:
    if context_map is None:
        return {key: 0 for key in _SECTION_KEYS}
    return {
        key: min(
            len(value) if isinstance(value := context_map.get(key), list) else 0,
            _MAX_ATTESTATION_SECTION_COUNT,
        )
        for key in _SECTION_KEYS
    }


def _attestation(
    *,
    model: ModelConfig,
    fixture_sha256: str,
    request_schema_sha256: str,
    terminal_call_count: int,
    provider_turn_count: int,
    fixture_read: bool,
    language_verified: bool,
    source_verified: bool,
    sink_verified: bool,
    semantic_relation_verified: bool,
    section_counts: dict[str, int],
    failure_class: str | None = None,
) -> dict[str, Any]:
    attestation: dict[str, Any] = {
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
        "status": "failed" if failure_class else "passed",
        "fixture_sha256": fixture_sha256,
        "provider": _bounded_attestation_identity(model.provider),
        "model": _bounded_attestation_identity(model.model_name),
        "sdk_version": _bounded_attestation_identity(_google_genai_sdk_version()),
        "request_schema_sha256": request_schema_sha256,
        "terminal_call_count": terminal_call_count,
        "provider_turn_count": provider_turn_count,
        "fixture_read": fixture_read,
        "language_verified": language_verified,
        "source_verified": source_verified,
        "sink_verified": sink_verified,
        "semantic_relation_verified": semantic_relation_verified,
        "section_counts": section_counts,
    }
    if failure_class:
        attestation["failure_class"] = failure_class
    return attestation


def _failed_result(
    *,
    model: ModelConfig,
    fixture_sha256: str,
    request_schema_sha256: str,
    terminal_call_count: int = 0,
    provider_turn_count: int = 0,
    fixture_read: bool = False,
    language_verified: bool = False,
    source_verified: bool = False,
    sink_verified: bool = False,
    semantic_relation_verified: bool = False,
    section_counts: dict[str, int] | None = None,
    failure_class: str,
) -> SemanticCanaryResult:
    return SemanticCanaryResult(
        False,
        _attestation(
            model=model,
            fixture_sha256=fixture_sha256,
            request_schema_sha256=request_schema_sha256,
            terminal_call_count=terminal_call_count,
            provider_turn_count=provider_turn_count,
            fixture_read=fixture_read,
            language_verified=language_verified,
            source_verified=source_verified,
            sink_verified=sink_verified,
            semantic_relation_verified=semantic_relation_verified,
            section_counts=_section_counts(None) if section_counts is None else section_counts,
            failure_class=failure_class,
        ),
    )


def _normalise_provider_failure(exc: Exception) -> str:
    """Classify failures without retaining provider content or headers."""
    if isinstance(exc, LocalToolSchemaDependencyMissing):
        return "local_dependency_missing"
    if isinstance(exc, LocalToolSchemaInvalid):
        return "local_schema_invalid"
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()[:512]
    if "429" in text or "quota" in text or "rate limit" in text:
        return "quota"
    if "401" in text or "403" in text or "auth" in text or "permission" in text:
        return "auth"
    if "400" in text and ("schema" in text or "json" in text or "field" in text):
        return "schema"
    if "schema" in name or "schema" in text:
        return "schema"
    if "timeout" in name or "timeout" in text:
        return "timeout"
    return "transport"


def resolve_semantic_canary_model(model_name: str) -> ModelConfig:
    """Resolve the exact requested model through RAPTOR's pinned-model route."""
    from core.llm.client import _pinned_llm_config

    config = _pinned_llm_config(model_name)
    model = config.primary_model
    if (
        not isinstance(model, ModelConfig)
        or model.provider != "gemini"
        or model.model_name != model_name
    ):
        raise ValueError("semantic canary model resolution did not produce Gemini")
    return model


def run_semantic_canary(
    model: ModelConfig,
    *,
    provider_factory: Any = create_provider,
    fixture_root: Path | None = None,
    lifecycle_events: Callable[..., None] | None = None,
) -> SemanticCanaryResult:
    """Exercise one isolated map turn without automatic transport retries."""
    if not isinstance(model, ModelConfig):
        invalid_model = ModelConfig(provider="", model_name="")
        return _failed_result(
            model=invalid_model,
            fixture_sha256="",
            request_schema_sha256="",
            failure_class="invalid_model",
        )

    directory_context = (
        tempfile.TemporaryDirectory(prefix="raptor-semantic-canary-")
        if fixture_root is None
        else nullcontext(fixture_root)
    )
    with directory_context as directory:
        active_fixture_root = Path(directory)
        fixture = _fresh_fixture()
        fixture_sha256 = hashlib.sha256(fixture.content.encode("utf-8")).hexdigest()
        fixture_path = "fixture.cpp"
        (active_fixture_root / fixture_path).write_text(fixture.content, encoding="utf-8")
        tools = _build_tools(SandboxedTools.for_repo(active_fixture_root))
        request_schema_sha256 = _request_schema_sha256(tools)
        terminal_calls = 0
        fixture_read_call_ids: list[str] = []
        successful_fixture_read_call_ids: set[str] = set()
        tool_handler_error = False
        invalid_fixture_read = False

        def events(event: Any) -> None:
            nonlocal terminal_calls, tool_handler_error, invalid_fixture_read
            if isinstance(event, ToolCallDispatched):
                if lifecycle_events is not None:
                    lifecycle_events("tool_call_dispatched")
                if event.call.name == "submit_context_map":
                    terminal_calls += 1
                    if lifecycle_events is not None:
                        lifecycle_events("terminal_call_dispatched")
                elif event.call.name == "read_file":
                    if event.call.input.get("path") == fixture_path:
                        fixture_read_call_ids.append(event.call.id)
                    else:
                        invalid_fixture_read = True
            elif isinstance(event, ToolCallReturned):
                fixture_read_completed = False
                if event.result.is_error:
                    tool_handler_error = True
                if (
                    event.call_id in fixture_read_call_ids
                    and _matches_fixture_read_result(
                        event.result,
                        fixture_path,
                        fixture.content,
                    )
                ):
                    successful_fixture_read_call_ids.add(event.call_id)
                    fixture_read_completed = True
                if lifecycle_events is not None:
                    lifecycle_events(
                        "tool_call_completed",
                        fixture_read=fixture_read_completed,
                    )

        try:
            provider = _CountingCanaryProvider(
                provider_factory(model),
                lifecycle_events=lifecycle_events,
                request_schema_sha256=request_schema_sha256,
            )
        except Exception as exc:
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                failure_class="provider_init" if _normalise_provider_failure(exc) == "transport" else _normalise_provider_failure(exc),
            )

        try:
            loop = ToolUseLoop(
                provider=provider,
                tools=tools,
                system=MAP_SYSTEM_PROMPT + (
                    "\n\nFor this isolated semantic canary, successfully read fixture.cpp "
                    "before submitting the terminal map. Emit only canonical evidence: an "
                    "attacker-controlled cli_arg source, a relation entry point, a shell_exec "
                    "sink, and an ID-linked unchecked flow. Include exact identifiers discovered "
                    "after the read in sources.entry, entry_points.name, and sink_details.name. "
                    "Set sources.entry to `fixture.cpp:<line> <identifier> argv[1]`. Use "
                    "repository-relative fixture locations and real source lines. Set "
                    "context_map.meta.language to an array inferred from source. Do not include "
                    "source text in the context map."
                ),
                terminal_tool="submit_context_map",
                max_iterations=_MAX_ITERATIONS,
                max_cost_usd=_MAX_COST_USD,
                max_seconds=_MAX_SECONDS,
                tool_timeout_s=_TOOL_TIMEOUT_S,
                max_tokens_per_turn=DEFAULT_MAX_TOKENS_PER_TURN,
                context_policy=ContextPolicy.RAISE,
                cache_control=CacheControl(system=True, tools=True),
                terminate_on_handler_error=False,
                events=events,
            )
            result = loop.run(_format_user_message({
                "files": [{
                    "path": fixture_path,
                    "lines": fixture.content.count("\n"),
                }],
            }))
        except CostBudgetExceeded:
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                failure_class="cost_limit",
            )
        except Exception as exc:
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                failure_class=_normalise_provider_failure(exc),
            )

        fixture_read = (
            len(fixture_read_call_ids) == 1
            and len(successful_fixture_read_call_ids) == 1
            and successful_fixture_read_call_ids == set(fixture_read_call_ids)
        )
        if (
            provider.turn_count != _MAX_ITERATIONS
            or terminal_calls != 1
            or result.terminated_by != "terminal_tool"
            or tool_handler_error
            or invalid_fixture_read
        ):
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                fixture_read=fixture_read,
                failure_class="terminal_contract",
            )
        context_map, error = _validate_terminal_context_map(result.terminal_tool_input)
        if error is not None or context_map is None:
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                fixture_read=fixture_read,
                failure_class="map_validation",
            )
        language_verified = fixture_read and _is_cpp_label(context_map["meta"].get("language"))
        source_verified = False
        sink_verified = False
        semantic_relation_verified = False
        if fixture_read:
            (
                source_verified,
                sink_verified,
                semantic_relation_verified,
            ) = _has_canonical_semantic_evidence(context_map, fixture)
        section_counts = _section_counts(context_map)
        if not (
            language_verified
            and source_verified
            and sink_verified
            and semantic_relation_verified
        ):
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                fixture_read=fixture_read,
                language_verified=language_verified,
                source_verified=source_verified,
                sink_verified=sink_verified,
                semantic_relation_verified=semantic_relation_verified,
                section_counts=section_counts,
                failure_class="semantic_evidence",
            )
        return SemanticCanaryResult(
            True,
            _attestation(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                fixture_read=fixture_read,
                language_verified=language_verified,
                source_verified=source_verified,
                sink_verified=sink_verified,
                semantic_relation_verified=semantic_relation_verified,
                section_counts=section_counts,
            ),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAPTOR semantic map preflight")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)
    from packages.code_understanding.semantic_canary_controller import (
        failed_semantic_canary_cli_result,
        run_semantic_canary_controller,
    )

    if args.model != _QUALIFICATION_MODEL:
        result = failed_semantic_canary_cli_result(args.model, "unsupported_model")
    else:
        try:
            model = resolve_semantic_canary_model(args.model)
        except Exception:
            result = failed_semantic_canary_cli_result(args.model, "model_resolution")
        else:
            result = run_semantic_canary_controller(model)
    print(json.dumps(result.attestation, sort_keys=True, separators=(",", ":")))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
