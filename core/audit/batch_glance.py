"""Batch GLANCE review — send multiple low-complexity functions in one LLM call.

GLANCE reviews use a minimal prompt (~500 tokens per function).  Batching
N functions into a single API call eliminates (N-1) system-prompt
repetitions and (N-1) round trips.  For 600 GLANCE items at batch size
10, that's 540 fewer API calls and ~1M fewer input tokens.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from collections.abc import Callable

from .context import _format_glance_prompt
from .orchestrator import ReviewOutcome, OrchestratorConfig

logger = logging.getLogger(__name__)

BATCH_SIZE = 10

_BATCH_SYSTEM_PROMPT = (
    "You are a security auditor doing a quick triage pass over multiple "
    "functions. For each function, determine whether it is security-relevant "
    "and could contain a vulnerability.\n\n"
    "Respond with a JSON array, one object per function, in the same order "
    "as presented. Each object must have:\n"
    '  {"file": "<file>", "function": "<name>", "status": "clean"|"suspicious", '
    '"body": "<one sentence>"}\n\n'
    'Use "suspicious" only when there is a concrete reason (e.g. unchecked '
    "input, missing bounds check, unsafe pattern). Default to \"clean\"."
)


def format_batch_prompt(contexts: list[dict[str, Any]]) -> str:
    """Build a single prompt containing N GLANCE reviews."""
    parts = [f"Review {len(contexts)} functions:\n"]
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"### Function {i}")
        parts.append(_format_glance_prompt(ctx))
        parts.append("")
    parts.append(
        f"Respond with a JSON array of exactly {len(contexts)} objects."
    )
    return "\n".join(parts)


def parse_batch_response(
    raw: str,
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse the LLM batch response into per-function result dicts.

    Falls back to empty results if the response is malformed —
    those functions will be individually re-reviewed.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(lines)
    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                results = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(results, list):
        return []
    return results


def make_batch_review_fn(
    llm_client: Any,
    *,
    model_name: str | None = None,
    task_type: str = "audit",
) -> Callable[[list[dict[str, Any]], OrchestratorConfig], list[ReviewOutcome]]:
    """Build a batch review function for GLANCE-bucket items.

    Returns a callable that accepts a list of contexts and returns
    a list of ReviewOutcomes — one per context, in order.
    Functions that can't be parsed from the batch response get
    status='error' so the caller can fall back to individual review.
    """
    import contextlib

    model_config_override = None
    if model_name:
        with contextlib.suppress(AttributeError, KeyError, TypeError):
            model_config_override = llm_client.config.find_model_config(model_name)

    def batch_review_fn(
        contexts: list[dict[str, Any]],
        config: OrchestratorConfig,
    ) -> list[ReviewOutcome]:
        if not contexts:
            return []

        prompt = format_batch_prompt(contexts)
        t0 = time.monotonic()

        kwargs: dict[str, Any] = {}
        if model_config_override is not None:
            kwargs["model_config"] = model_config_override
        else:
            kwargs["task_type"] = task_type

        try:
            response = llm_client.generate(
                prompt,
                system_prompt=_BATCH_SYSTEM_PROMPT,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("batch glance review failed: %s", exc)
            return [
                ReviewOutcome(
                    file=ctx["file"],
                    function=ctx["function"],
                    status="error",
                    body=f"batch review failed: {exc}",
                )
                for ctx in contexts
            ]

        raw_text = response.text if hasattr(response, "text") else str(response)
        duration = time.monotonic() - t0
        model = getattr(response, "model", "")
        cost = getattr(response, "cost", 0.0)
        per_cost = cost / len(contexts) if contexts else 0.0

        results = parse_batch_response(raw_text, contexts)

        outcomes: list[ReviewOutcome] = []
        for i, ctx in enumerate(contexts):
            if i < len(results) and isinstance(results[i], dict):
                r = results[i]
                status = r.get("status", "suspicious")
                if status not in ("clean", "suspicious"):
                    status = "suspicious"
                outcomes.append(ReviewOutcome(
                    file=ctx["file"],
                    function=ctx["function"],
                    status=status,
                    body=r.get("body", ""),
                    cost_usd=per_cost,
                    model=model,
                    duration_s=duration / len(contexts),
                    evidence_tool="triage:batch",
                ))
            else:
                outcomes.append(ReviewOutcome(
                    file=ctx["file"],
                    function=ctx["function"],
                    status="error",
                    body="batch response missing or malformed for this function",
                ))

        logger.info(
            "batch glance: %d functions in %.1fs, $%.4f",
            len(contexts), duration, cost,
        )
        return outcomes

    return batch_review_fn
