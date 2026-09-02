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
    build_context_map_schema,
)
from packages.code_understanding.dispatch.tools import SandboxedTools
from packages.code_understanding.prompts import MAP_SYSTEM_PROMPT

_MAX_ITERATIONS = 2
_MAX_COST_USD = 0.05
_MAX_SECONDS = 30.0
_TOOL_TIMEOUT_S = 5.0
_ATTESTATION_ID_MAX_CHARS = 128
_ATTESTATION_SCHEMA_VERSION = 3
_MAX_ATTESTATION_SECTION_COUNT = 1_024
_MAX_SEMANTIC_FAILURE_REASONS = 16
_QUALIFICATION_MODEL = "gemini-2.5-flash"
SECTION_COUNT_KEYS = (
    "sources",
    "sinks",
    "trust_boundaries",
    "entry_points",
    "sink_details",
    "boundary_details",
    "unchecked_flows",
)
SEMANTIC_CHECK_KEYS = (
    "source_type_verified",
    "source_entry_verified",
    "source_trust_verified",
    "top_level_sink_type_verified",
    "top_level_sink_callsite_verified",
    "sink_detail_type_verified",
    "sink_detail_callsite_verified",
    "sink_detail_wrapper_name_verified",
    "sink_detail_operation_verified",
    "entry_point_callsite_verified",
    "entry_point_relation_name_verified",
    "flow_entry_reference_known",
    "flow_sink_reference_known",
    "flow_links_expected_entry_and_sink",
    "no_declared_boundary_verified",
)
_SEMANTIC_FAILURE_REASON_BY_CHECK = {
    "source_type_verified": "source_type_mismatch",
    "source_entry_verified": "source_entry_mismatch",
    "source_trust_verified": "source_trust_mismatch",
    "top_level_sink_type_verified": "top_level_sink_type_mismatch",
    "top_level_sink_callsite_verified": "top_level_sink_callsite_mismatch",
    "sink_detail_type_verified": "sink_detail_type_mismatch",
    "sink_detail_callsite_verified": "sink_detail_callsite_mismatch",
    "sink_detail_wrapper_name_verified": "sink_detail_wrapper_name_mismatch",
    "sink_detail_operation_verified": "sink_detail_operation_mismatch",
    "entry_point_callsite_verified": "entry_point_callsite_mismatch",
    "entry_point_relation_name_verified": "entry_point_relation_name_mismatch",
    "flow_entry_reference_known": "flow_entry_reference_unknown",
    "flow_sink_reference_known": "flow_sink_reference_unknown",
    "flow_links_expected_entry_and_sink": "flow_not_linked",
    "no_declared_boundary_verified": "declared_boundary_present",
}
SEMANTIC_FAILURE_REASONS = frozenset(
    (*_SEMANTIC_FAILURE_REASON_BY_CHECK.values(), "semantic_evidence_mismatch")
)
_DANGEROUS_OPERATION = "std::system(command.c_str());"
_CANARY_TRUSTED_ADDENDUM = (
    "\n\nFor this isolated semantic canary, read fixture.cpp first. Identify the "
    "line containing the actual std::system call. Use that call line for "
    "sinks.location and sink_details.file and sink_details.line. Use the "
    "enclosing user-defined wrapper function's exact identifier for "
    "sink_details.name, not std::system. Use the exact relation function "
    "identifier for entry_points.name. Link unchecked_flows to the exact "
    "entry-point and sink-detail IDs. Invoke submit_context_map exactly once."
)


@dataclass(frozen=True)
class _Fixture:
    source: str
    sink: str
    relation: str
    source_line: int
    sink_wrapper_line: int
    sink_call_line: int
    relation_line: int
    operation: str
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
        sink_wrapper_line=8,
        sink_call_line=9,
        relation_line=12,
        operation=_DANGEROUS_OPERATION,
        content=(
            "#include <cstdlib>\n"
            "#include <string>\n\n"
            f"std::string {source}(int argc, char** argv) {{\n"
            "    return argc > 1 ? argv[1] : \"\";\n"
            "}\n\n"
            f"void {sink}(const std::string& command) {{\n"
            f"    {_DANGEROUS_OPERATION}\n"
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


@dataclass(frozen=True)
class _SemanticEvidence:
    source_verified: bool
    sink_verified: bool
    semantic_relation_verified: bool
    checks: dict[str, bool]
    failure_reasons: list[str]


def _is_cli_source_type(value: Any) -> bool:
    return isinstance(value, str) and value.casefold() in _CLI_SOURCE_TYPES


def _is_shell_exec_type(value: Any) -> bool:
    return isinstance(value, str) and value.casefold() in _SHELL_EXEC_TYPES


def _matching_entry_point_ids(
    context_map: dict[str, Any], fixture: _Fixture
) -> set[str]:
    return {
        entry["id"]
        for entry in context_map["entry_points"]
        if isinstance(entry.get("id"), str)
        and entry["id"]
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
        and entry["id"]
        and _is_shell_exec_type(entry.get("type"))
        and _details_match_fixture_line(entry, fixture.sink_call_line)
        and entry.get("name") == fixture.sink
        and entry.get("operation") == fixture.operation
    }


def _source_is_verified(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    return any(
        _is_cli_source_type(entry.get("type"))
        and entry.get("trust_level") == "attacker_controlled"
        and _matches_source_entry(entry.get("entry"), fixture)
        for entry in context_map["sources"]
    )


def _sink_is_verified(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    top_level_sink = any(
        _is_shell_exec_type(entry.get("type"))
        and _is_fixture_location(entry.get("location"), fixture.sink_call_line)
        for entry in context_map["sinks"]
    )
    return top_level_sink and bool(_matching_sink_detail_ids(context_map, fixture))


def _all_entry_ids(context_map: dict[str, Any]) -> set[str]:
    return {
        entry["id"]
        for entry in context_map["entry_points"]
        if isinstance(entry.get("id"), str) and entry["id"]
    }


def _all_sink_detail_ids(context_map: dict[str, Any]) -> set[str]:
    return {
        entry["id"]
        for entry in context_map["sink_details"]
        if isinstance(entry.get("id"), str) and entry["id"]
    }


def _flow_reference_checks(context_map: dict[str, Any]) -> tuple[bool, bool]:
    entry_ids = _all_entry_ids(context_map)
    sink_ids = _all_sink_detail_ids(context_map)
    entry_known = True
    sink_known = True
    for flow in context_map["unchecked_flows"]:
        entry_refs = _reference_ids(flow.get("entry_point"))
        sink_refs = _reference_ids(flow.get("sink"))
        entry_known = entry_known and bool(entry_refs and entry_refs.issubset(entry_ids))
        sink_known = sink_known and bool(sink_refs and sink_refs.issubset(sink_ids))
    return entry_known, sink_known


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


def _evaluate_canonical_semantic_evidence(
    context_map: dict[str, Any], fixture: _Fixture
) -> _SemanticEvidence:
    entry_ids = _matching_entry_point_ids(context_map, fixture)
    sink_ids = _matching_sink_detail_ids(context_map, fixture)
    all_entry_ids = _all_entry_ids(context_map)
    all_sink_ids = _all_sink_detail_ids(context_map)
    expected_flow_entry_ids = all_entry_ids if len(all_entry_ids) == 1 else set()
    expected_flow_sink_ids = all_sink_ids if len(all_sink_ids) == 1 else set()
    flow_entry_reference_known, flow_sink_reference_known = _flow_reference_checks(
        context_map
    )
    checks = {
        "source_type_verified": any(
            _is_cli_source_type(entry.get("type"))
            for entry in context_map["sources"]
        ),
        "source_entry_verified": any(
            _matches_source_entry(entry.get("entry"), fixture)
            for entry in context_map["sources"]
        ),
        "source_trust_verified": any(
            entry.get("trust_level") == "attacker_controlled"
            for entry in context_map["sources"]
        ),
        "top_level_sink_type_verified": any(
            _is_shell_exec_type(entry.get("type"))
            for entry in context_map["sinks"]
        ),
        "top_level_sink_callsite_verified": any(
            _is_fixture_location(entry.get("location"), fixture.sink_call_line)
            for entry in context_map["sinks"]
        ),
        "sink_detail_type_verified": any(
            _is_shell_exec_type(entry.get("type"))
            for entry in context_map["sink_details"]
        ),
        "sink_detail_callsite_verified": any(
            _details_match_fixture_line(entry, fixture.sink_call_line)
            for entry in context_map["sink_details"]
        ),
        "sink_detail_wrapper_name_verified": any(
            entry.get("name") == fixture.sink
            for entry in context_map["sink_details"]
        ),
        "sink_detail_operation_verified": any(
            entry.get("operation") == fixture.operation
            for entry in context_map["sink_details"]
        ),
        "entry_point_callsite_verified": any(
            _details_match_fixture_line(entry, fixture.relation_line)
            for entry in context_map["entry_points"]
        ),
        "entry_point_relation_name_verified": any(
            entry.get("name") == fixture.relation
            for entry in context_map["entry_points"]
        ),
        "flow_entry_reference_known": flow_entry_reference_known,
        "flow_sink_reference_known": flow_sink_reference_known,
        "flow_links_expected_entry_and_sink": (
            not flow_entry_reference_known
            or not flow_sink_reference_known
            or any(
                (entry_refs := _reference_ids(flow.get("entry_point")))
                and (sink_refs := _reference_ids(flow.get("sink")))
                and entry_refs == expected_flow_entry_ids
                and sink_refs == expected_flow_sink_ids
                for flow in context_map["unchecked_flows"]
            )
        ),
        "no_declared_boundary_verified": _flow_has_no_declared_boundary(
            context_map, entry_ids, sink_ids
        ),
    }
    source_verified = _source_is_verified(context_map, fixture)
    sink_verified = _sink_is_verified(context_map, fixture)
    semantic_relation_verified = (
        source_verified
        and sink_verified
        and bool(entry_ids)
        and bool(sink_ids)
        and checks["flow_links_expected_entry_and_sink"]
        and flow_entry_reference_known
        and flow_sink_reference_known
        and checks["no_declared_boundary_verified"]
    )
    failure_reasons = sorted(
        _SEMANTIC_FAILURE_REASON_BY_CHECK[key]
        for key in SEMANTIC_CHECK_KEYS
        if not checks[key]
    )[:_MAX_SEMANTIC_FAILURE_REASONS]
    if not semantic_relation_verified and not failure_reasons:
        failure_reasons = ["semantic_evidence_mismatch"]
    return _SemanticEvidence(
        source_verified=source_verified,
        sink_verified=sink_verified,
        semantic_relation_verified=semantic_relation_verified,
        checks=checks,
        failure_reasons=failure_reasons,
    )


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


def _trusted_system_instruction() -> str:
    return MAP_SYSTEM_PROMPT + _CANARY_TRUSTED_ADDENDUM


def _system_instruction_sha256() -> str:
    return hashlib.sha256(
        _trusted_system_instruction().encode("utf-8")
    ).hexdigest()


def _request_schema_sha256(tools: list[Any]) -> str:
    schema = GeminiProvider._tool_response_schema(tools)
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _section_counts(context_map: dict[str, Any] | None) -> dict[str, int]:
    if context_map is None:
        return {key: 0 for key in SECTION_COUNT_KEYS}
    return {
        key: min(
            len(value) if isinstance(value := context_map.get(key), list) else 0,
            _MAX_ATTESTATION_SECTION_COUNT,
        )
        for key in SECTION_COUNT_KEYS
    }


def _default_semantic_checks() -> dict[str, bool]:
    return {key: False for key in SEMANTIC_CHECK_KEYS}


def _bounded_semantic_checks(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return _default_semantic_checks()
    return {key: value.get(key) is True for key in SEMANTIC_CHECK_KEYS}


def _bounded_semantic_failure_reasons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            reason
            for reason in value
            if isinstance(reason, str) and reason in SEMANTIC_FAILURE_REASONS
        }
    )[:_MAX_SEMANTIC_FAILURE_REASONS]


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
    semantic_checks: dict[str, bool],
    semantic_failure_reasons: list[str],
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
        "system_instruction_sha256": _system_instruction_sha256(),
        "request_schema_sha256": request_schema_sha256,
        "terminal_call_count": terminal_call_count,
        "provider_turn_count": provider_turn_count,
        "fixture_read": fixture_read,
        "language_verified": language_verified,
        "source_verified": source_verified,
        "sink_verified": sink_verified,
        "semantic_relation_verified": semantic_relation_verified,
        "semantic_checks": _bounded_semantic_checks(semantic_checks),
        "semantic_failure_reasons": _bounded_semantic_failure_reasons(
            semantic_failure_reasons
        ),
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
    semantic_checks: dict[str, bool] | None = None,
    semantic_failure_reasons: list[str] | None = None,
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
            semantic_checks=(
                _default_semantic_checks()
                if semantic_checks is None
                else semantic_checks
            ),
            semantic_failure_reasons=(
                [] if semantic_failure_reasons is None else semantic_failure_reasons
            ),
            section_counts=(
                _section_counts(None) if section_counts is None else section_counts
            ),
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
        canary_schema = build_context_map_schema(require_canary_names=True)
        tools = _build_tools(
            SandboxedTools.for_repo(active_fixture_root),
            context_map_schema=canary_schema,
        )
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
                system=_trusted_system_instruction(),
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
        context_map, error = _validate_terminal_context_map(
            result.terminal_tool_input,
            context_map_schema=canary_schema,
        )
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
        language_verified = fixture_read and _is_cpp_label(
            context_map["meta"].get("language")
        )
        evidence = _SemanticEvidence(
            source_verified=False,
            sink_verified=False,
            semantic_relation_verified=False,
            checks=_default_semantic_checks(),
            failure_reasons=[],
        )
        if fixture_read:
            evidence = _evaluate_canonical_semantic_evidence(context_map, fixture)
        source_verified = evidence.source_verified
        sink_verified = evidence.sink_verified
        semantic_relation_verified = evidence.semantic_relation_verified
        section_counts = _section_counts(context_map)
        if not (
            language_verified
            and source_verified
            and sink_verified
            and semantic_relation_verified
            and all(evidence.checks.values())
            and not evidence.failure_reasons
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
                semantic_checks=evidence.checks,
                semantic_failure_reasons=evidence.failure_reasons,
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
                semantic_checks=evidence.checks,
                semantic_failure_reasons=evidence.failure_reasons,
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
