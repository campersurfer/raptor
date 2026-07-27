"""Mechanical reachability check for /audit finding gate.

Two-signal design: a function is classified as dead (and forced to
``dormant`` instead of ``finding``) only when BOTH signals agree:

  1. **Call graph**: ``callers_of()`` returns zero callers (definitive +
     uncertain + method-match-overinclusive all empty)
  2. **Binary oracle**: classification is ``absent`` with full-DWARF tier

If the binary oracle is unavailable (no binary, non-native language,
stripped binary), the single call-graph signal is not enough for a hard
gate — indirect calls, callbacks, cross-language dispatch, and dynamic
routing all produce zero-static-caller functions that are fully
reachable. In that case the gate requires the operator to provide a
``--reach-via`` explanation instead.

Entry points (from context-map) bypass the check entirely — they are
reachable by definition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReachabilityVerdict:
    reachable: bool
    reason: str
    is_entry_point: bool = False
    has_callers: bool = False
    binary_absent: bool = False
    binary_tier: Optional[str] = None


def check_reachability(
    *,
    file_path: str,
    function_name: str,
    checklist: Optional[Dict[str, Any]] = None,
    context_map: Optional[Dict[str, Any]] = None,
    inventory: Optional[Dict[str, Any]] = None,
) -> ReachabilityVerdict:
    """Check whether a function is reachable from any entry point.

    Returns a verdict with enough detail for the caller to decide
    whether to hard-gate (both signals agree = dead) or soft-gate
    (call graph only, require --reach-via).
    """
    func_key = f"{file_path}:{function_name}"

    if _is_entry_point(func_key, context_map):
        return ReachabilityVerdict(
            reachable=True,
            reason="entry point in context-map",
            is_entry_point=True,
            has_callers=True,
        )

    has_callers = _has_any_callers(
        file_path, function_name, inventory,
    )

    if has_callers:
        return ReachabilityVerdict(
            reachable=True,
            reason="has static callers",
            has_callers=True,
        )

    # None means inventory was unavailable -- we cannot confirm zero
    # callers, so treat the function as reachable (unknown).
    if has_callers is None:
        return ReachabilityVerdict(
            reachable=True,
            reason="caller information unavailable, assuming reachable",
            has_callers=False,
        )

    binary_verdict = _check_binary_oracle(
        file_path, function_name, checklist,
    )

    if binary_verdict is None:
        return ReachabilityVerdict(
            reachable=False,
            reason="zero static callers, no binary oracle data",
            has_callers=False,
            binary_absent=False,
            binary_tier=None,
        )

    binary_absent, binary_tier = binary_verdict

    if binary_absent and binary_tier == "full":
        return ReachabilityVerdict(
            reachable=False,
            reason="zero static callers AND binary oracle absent (full DWARF)",
            has_callers=False,
            binary_absent=True,
            binary_tier="full",
        )

    if binary_absent and binary_tier == "symbol_only":
        return ReachabilityVerdict(
            reachable=True,
            reason="zero static callers, binary oracle absent at symbol-only tier (not suppression-grade per policy)",
            has_callers=False,
            binary_absent=True,
            binary_tier="symbol_only",
        )

    return ReachabilityVerdict(
        reachable=True,
        reason=f"zero static callers but binary oracle says present ({binary_tier})",
        has_callers=False,
        binary_absent=False,
        binary_tier=binary_tier,
    )


def _is_entry_point(
    func_key: str,
    context_map: Optional[Dict[str, Any]],
) -> bool:
    if not context_map:
        return False
    for ep in context_map.get("entry_points", []):
        key = f"{ep.get('file', '')}:{ep.get('name', '')}"
        if key == func_key:
            return True
    return False


def _has_any_callers(
    file_path: str,
    function_name: str,
    inventory: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """Check whether a function has any callers.

    Returns True/False when the inventory is available, or None when
    inventory is unavailable (meaning "unknown" -- callers should not
    treat this as "confirmed zero callers").
    """
    if not inventory:
        return None
    try:
        from core.analysis.reachability import (
            InternalFunction,
            callers_of,
        )
        target = InternalFunction(
            file_path=file_path, name=function_name, line=0,
        )
        result = callers_of(inventory, target, exclude_test_files=True)
        return len(result.all_callers) > 0
    except (ImportError, KeyError, TypeError, AttributeError, ValueError):
        logger.debug(
            "callers_of failed for %s:%s", file_path, function_name,
            exc_info=True,
        )
        return None


def _check_binary_oracle(
    file_path: str,
    function_name: str,
    checklist: Optional[Dict[str, Any]],
) -> Optional[tuple]:
    """Check binary oracle verdict from checklist metadata.

    Returns (is_absent: bool, tier: str) or None if no binary data.
    """
    if not checklist:
        return None

    for file_entry in checklist.get("files", []):
        if file_entry.get("path") != file_path:
            continue
        for item in file_entry.get("items", []):
            if item.get("name") != function_name:
                continue
            bo = (item.get("metadata") or {}).get("binary_oracle")
            if not bo:
                return None
            classification = bo.get("classification", "")
            tier = "full"
            for b in bo.get("binaries", []):
                if b.get("tier") == "symbol_only":
                    tier = "symbol_only"
                    break
            return (classification == "absent", tier)

    return None
