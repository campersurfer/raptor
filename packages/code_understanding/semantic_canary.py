"""Bounded semantic preflight for the model-backed context-map path."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from core.llm.config import ModelConfig
from core.llm.providers import GeminiProvider, create_provider
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

_MAX_ITERATIONS = 3
_MAX_COST_USD = 0.05
_MAX_SECONDS = 30.0
_TOOL_TIMEOUT_S = 5.0
_ATTESTATION_ID_MAX_CHARS = 128
_ATTESTATION_SCHEMA_VERSION = 1
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
    content: str


@dataclass(frozen=True)
class SemanticCanaryResult:
    success: bool
    attestation: dict[str, Any]


class _CountingCanaryProvider:
    """Count turns without replaying a transport request."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.turn_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def turn(self, *args: Any, **kwargs: Any) -> Any:
        self.turn_count += 1
        return self._provider.turn(*args, **kwargs)


def _fresh_identifier() -> str:
    return secrets.token_hex(12)


def _fresh_fixture() -> _Fixture:
    """Build opaque per-run C++ markers that are never attested."""
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


def _has_exact_semantic_evidence(context_map: dict[str, Any], fixture: _Fixture) -> bool:
    """Require map claims to use the exact opaque identifiers in source."""
    return (
        any(entry.get("name") == fixture.source for entry in context_map["sources"])
        and any(entry.get("name") == fixture.sink for entry in context_map["sinks"])
        and any(
            entry.get("source") == fixture.source
            and entry.get("sink") == fixture.sink
            and entry.get("relation") == fixture.relation
            for entry in context_map["unchecked_flows"]
        )
    )


def _is_cpp_label(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in {"c++", "cpp", "cxx"}


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
        key: len(value) if isinstance(value := context_map.get(key), list) else 0
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
            semantic_relation_verified=semantic_relation_verified,
            section_counts=_section_counts(None) if section_counts is None else section_counts,
            failure_class=failure_class,
        ),
    )


def _normalise_provider_failure(exc: Exception) -> str:
    """Classify failures without retaining provider content or headers."""
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

    with tempfile.TemporaryDirectory(prefix="raptor-semantic-canary-") as directory:
        fixture_root = Path(directory)
        fixture = _fresh_fixture()
        fixture_sha256 = hashlib.sha256(fixture.content.encode("utf-8")).hexdigest()
        fixture_path = "fixture.cpp"
        (fixture_root / fixture_path).write_text(fixture.content, encoding="utf-8")
        tools = _build_tools(SandboxedTools.for_repo(fixture_root))
        request_schema_sha256 = _request_schema_sha256(tools)
        terminal_calls = 0
        fixture_read_call_ids: set[str] = set()
        successful_fixture_read_call_ids: set[str] = set()

        def events(event: Any) -> None:
            nonlocal terminal_calls
            if isinstance(event, ToolCallDispatched):
                if event.call.name == "submit_context_map":
                    terminal_calls += 1
                elif (
                    event.call.name == "read_file"
                    and event.call.input.get("path") == fixture_path
                ):
                    fixture_read_call_ids.add(event.call.id)
            elif (
                isinstance(event, ToolCallReturned)
                and event.call_id in fixture_read_call_ids
                and not event.result.is_error
            ):
                successful_fixture_read_call_ids.add(event.call_id)

        try:
            provider = _CountingCanaryProvider(provider_factory(model))
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
                    "before submitting the terminal map. Derive one source name, one "
                    "sink name, and their relation name exactly from the fixture source; "
                    "place them in sources, sinks, and unchecked_flows. Infer the language "
                    "from the file before setting context_map.meta.language. Do not include "
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
            and successful_fixture_read_call_ids == fixture_read_call_ids
        )
        if terminal_calls != 1 or result.terminated_by != "terminal_tool":
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
        semantic_relation_verified = fixture_read and _has_exact_semantic_evidence(context_map, fixture)
        section_counts = _section_counts(context_map)
        if not language_verified or not semantic_relation_verified:
            return _failed_result(
                model=model,
                fixture_sha256=fixture_sha256,
                request_schema_sha256=request_schema_sha256,
                terminal_call_count=terminal_calls,
                provider_turn_count=provider.turn_count,
                fixture_read=fixture_read,
                language_verified=language_verified,
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
                semantic_relation_verified=semantic_relation_verified,
                section_counts=section_counts,
            ),
        )


def _cli_failure(model_name: str, failure_class: str) -> SemanticCanaryResult:
    model = ModelConfig(provider="gemini", model_name=model_name)
    return _failed_result(
        model=model,
        fixture_sha256="",
        request_schema_sha256="",
        failure_class=failure_class,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAPTOR semantic map preflight")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)
    if args.model != _QUALIFICATION_MODEL:
        result = _cli_failure(args.model, "unsupported_model")
    else:
        try:
            model = resolve_semantic_canary_model(args.model)
        except Exception:
            result = _cli_failure(args.model, "model_resolution")
        else:
            result = run_semantic_canary(model)
    print(json.dumps(result.attestation, sort_keys=True, separators=(",", ":")))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
