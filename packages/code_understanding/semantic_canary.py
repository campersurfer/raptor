"""Bounded semantic preflight for the model-backed context-map path."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.llm.client import _is_quota_error, _is_retryable_error
from core.llm.config import ModelConfig
from core.llm.providers import create_provider
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
_RETRY_DELAYS_SECONDS = (1.0,)
_ATTESTATION_ID_MAX_CHARS = 128


@dataclass(frozen=True)
class _Fixture:
    source: str
    sink: str
    relation: str
    content: str


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


def _bounded_attestation_identity(value: str) -> str:
    return value[:_ATTESTATION_ID_MAX_CHARS]

@dataclass(frozen=True)
class SemanticCanaryResult:
    success: bool
    attestation: dict[str, Any]


class _RetryingCanaryProvider:
    def __init__(
        self,
        provider: Any,
        *,
        sleep: Callable[[float], None],
        retry_delays: tuple[float, ...],
    ) -> None:
        self._provider = provider
        self._sleep = sleep
        self._retry_delays = retry_delays
        self.attempts = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def turn(self, *args: Any, **kwargs: Any) -> Any:
        for delay in (*self._retry_delays, None):
            self.attempts += 1
            try:
                return self._provider.turn(*args, **kwargs)
            except Exception as exc:  # provider boundary
                if delay is None or _is_quota_error(exc) or not _is_retryable_error(exc):
                    raise
                self._sleep(delay)
        raise AssertionError("canary retry loop exhausted without a result")


def run_semantic_canary(
    model: ModelConfig,
    *,
    provider_factory: Callable[[ModelConfig], Any] = create_provider,
    sleep: Callable[[float], None] = time.sleep,
    retry_delays: tuple[float, ...] = _RETRY_DELAYS_SECONDS,
) -> SemanticCanaryResult:
    """Exercise one isolated map turn and return a bounded attestation."""
    if not isinstance(model, ModelConfig):
        return SemanticCanaryResult(False, {"status": "failed", "reason": "invalid-model"})

    try:
        provider = _RetryingCanaryProvider(
            provider_factory(model), sleep=sleep, retry_delays=retry_delays,
        )
    except Exception:
        return SemanticCanaryResult(False, {"status": "failed", "reason": "provider-init"})

    with tempfile.TemporaryDirectory(prefix="raptor-semantic-canary-") as directory:
        fixture_root = Path(directory)
        fixture = _fresh_fixture()
        fixture_path = "fixture.cpp"
        (fixture_root / fixture_path).write_text(fixture.content, encoding="utf-8")
        terminal_calls = 0
        fixture_read_call_ids: set[str] = set()
        fixture_read_succeeded = False

        def events(event: Any) -> None:
            nonlocal terminal_calls, fixture_read_succeeded
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
                fixture_read_succeeded = True

        try:
            loop = ToolUseLoop(
                provider=provider,
                tools=_build_tools(SandboxedTools.for_repo(fixture_root)),
                system=MAP_SYSTEM_PROMPT + (
                    "\n\nFor this isolated semantic canary, successfully read fixture.cpp "
                    "before submitting the terminal map. Recognize the source as C++. "
                    "Derive one source name, one sink name, and their relation name exactly "
                    "from the fixture source; place them in sources, sinks, and unchecked_flows. "
                    "Set context_map.meta.language to C++. Do not include source text in the "
                    "context map."
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
                    "language": "C++",
                    "lines": fixture.content.count("\n"),
                }],
            }))
        except CostBudgetExceeded:
            return SemanticCanaryResult(False, {"status": "failed", "reason": "cost-limit"})
        except Exception:
            return SemanticCanaryResult(False, {"status": "failed", "reason": "loop-failure"})

    if terminal_calls != 1 or result.terminated_by != "terminal_tool":
        return SemanticCanaryResult(False, {"status": "failed", "reason": "terminal-contract"})
    context_map, error = _validate_terminal_context_map(result.terminal_tool_input)
    if error is not None or context_map is None:
        return SemanticCanaryResult(False, {"status": "failed", "reason": "map-validation"})
    language = context_map["meta"].get("language")
    if (
        not fixture_read_succeeded
        or not isinstance(language, str)
        or language.strip().casefold() not in {"c++", "cpp", "cxx"}
        or not _has_exact_semantic_evidence(context_map, fixture)
    ):
        return SemanticCanaryResult(False, {"status": "failed", "reason": "semantic-evidence"})
    return SemanticCanaryResult(
        True,
        {
            "status": "passed",
            "fixture_sha256": hashlib.sha256(fixture.content.encode("utf-8")).hexdigest(),
            "provider": _bounded_attestation_identity(model.provider),
            "model": _bounded_attestation_identity(model.model_name),
            "terminal_calls": terminal_calls,
            "attempts": provider.attempts,
            "fixture_read": True,
            "language_verified": True,
            "semantic_relation_verified": True,
            "section_counts": {
                key: len(context_map[key])
                for key in (
                    "sources", "sinks", "trust_boundaries", "entry_points",
                    "sink_details", "boundary_details", "unchecked_flows",
                )
            },
        },
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAPTOR semantic map preflight")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    result = run_semantic_canary(ModelConfig(provider=args.provider, model_name=args.model))
    print(json.dumps(result.attestation, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
