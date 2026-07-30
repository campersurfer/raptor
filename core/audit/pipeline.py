"""Shared audit pipeline entry point.

Both ``libexec/raptor-audit run`` and the corpus runner call
``run_audit_pipeline()`` so setup logic lives in one place and the
corpus runner can run in-process (no subprocess, no cold-start).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditPipelineOpts:
    """Plain-data options for ``run_audit_pipeline``.

    Every field has a sensible default so callers only set what they need.
    """

    target_path: Path = field(default_factory=lambda: Path("."))
    out_dir: Path = field(default_factory=lambda: Path("out"))
    scope: Optional[list[str]] = None
    functions: Optional[list[str]] = None
    models: Optional[list[str]] = None
    max_cost_usd: Optional[float] = None
    max_seconds: Optional[float] = None
    budget: Optional[int] = None
    strategy_filter: Optional[str] = None
    review_passes: int = 1
    adversarial: bool = False
    max_propagation_depth: Optional[int] = None
    subsystem_depth: int = 0
    validate: bool = True
    no_binary_oracle: bool = False
    binary_verdicts: Optional[Dict[str, str]] = None
    inventory: Optional[Dict[str, Any]] = None
    annotations_dir: Optional[Path] = None
    codeql_db_path: Optional[str] = None
    threat_model: Optional[Dict[str, Any]] = None
    joern_overrides: Optional[Dict[str, Any]] = None
    joern_server: Optional[Any] = None
    on_progress: Optional[Callable] = None


def run_audit_pipeline(opts: AuditPipelineOpts):
    """Run the audit orchestrator and return its result.

    Sets up the LLM client, builds the OrchestratorConfig, and calls
    ``run_orchestrator``.  Returns the ``OrchestratorResult``.
    """
    from core.llm.log_quiet import quiet_noisy_loggers

    quiet_noisy_loggers()

    from core.llm.client import LLMClient
    from core.audit.llm_review import make_review_fn
    from core.audit.orchestrator import OrchestratorConfig, run_orchestrator

    llm_cfg = None
    if opts.max_cost_usd is not None:
        from core.llm.config import LLMConfig

        llm_cfg = LLMConfig(max_cost_per_scan=opts.max_cost_usd)
    client = LLMClient(config=llm_cfg)

    review_fn = make_review_fn(client, task_type="audit")

    models = opts.models or ["default"]
    config = OrchestratorConfig(
        target_path=opts.target_path,
        out_dir=opts.out_dir,
        budget=opts.budget,
        scope=opts.scope,
        strategy_filter=opts.strategy_filter,
        models=models,
        multi_model=len(models) > 1,
        adversarial=opts.adversarial,
        max_cost_usd=opts.max_cost_usd,
        max_seconds=opts.max_seconds,
        review_passes=opts.review_passes,
        subsystem_depth=opts.subsystem_depth,
        max_propagation_depth=opts.max_propagation_depth,
        validate=opts.validate,
        no_binary_oracle=opts.no_binary_oracle,
        binary_verdicts=opts.binary_verdicts,
        inventory=opts.inventory,
        codeql_db_path=opts.codeql_db_path,
        threat_model=opts.threat_model,
        annotations_dir=opts.annotations_dir,
        functions=opts.functions,
        joern_overrides=opts.joern_overrides,
        joern_server=opts.joern_server,
    )

    return run_orchestrator(config, review_fn, on_progress=opts.on_progress)