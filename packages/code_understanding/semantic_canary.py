"""Bounded semantic preflight for the model-backed context-map path."""

from __future__ import annotations

import argparse
import json
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
from core.llm.tool_use.types import ToolCallDispatched
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

_FIXTURE = """#include <string>\n\nstd::string handle_request(const std::string& input) {\n    return input;\n}\n"""


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
    """Exercise one isolated map turn and return a redacted attestation."""
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
        (fixture_root / "fixture.cpp").write_text(_FIXTURE, encoding="utf-8")
        terminal_calls = 0

        def events(event: Any) -> None:
            nonlocal terminal_calls
            if isinstance(event, ToolCallDispatched) and event.call.name == "submit_context_map":
                terminal_calls += 1

        try:
            loop = ToolUseLoop(
                provider=provider,
                tools=_build_tools(SandboxedTools.for_repo(fixture_root)),
                system=MAP_SYSTEM_PROMPT,
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
            result = loop.run(_format_user_message({"files": []}))
        except CostBudgetExceeded:
            return SemanticCanaryResult(False, {"status": "failed", "reason": "cost-limit"})
        except Exception:
            return SemanticCanaryResult(False, {"status": "failed", "reason": "loop-failure"})

    if terminal_calls != 1 or result.terminated_by != "terminal_tool":
        return SemanticCanaryResult(False, {"status": "failed", "reason": "terminal-contract"})
    context_map, error = _validate_terminal_context_map(result.terminal_tool_input)
    if error is not None or context_map is None:
        return SemanticCanaryResult(False, {"status": "failed", "reason": "map-validation"})
    return SemanticCanaryResult(
        True,
        {
            "status": "passed",
            "fixture": "isolated-cpp",
            "provider": model.provider,
            "model": model.model_name,
            "terminal_calls": terminal_calls,
            "attempts": provider.attempts,
            "section_counts": {
                key: len(value) if isinstance(value, list) else 0
                for key, value in context_map.items()
                if key != "meta"
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
