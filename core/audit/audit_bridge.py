"""Bridge between /audit output and /validate input.

Discovers constraints.json from sibling audit run directories and
enriches attack-paths with parameter preconditions so the validation
pipeline can score paths by how constrained they are.

Usage (from Stage 0 in /validate):

    from core.audit.audit_bridge import find_audit_output, load_audit_constraints

    audit_dir = find_audit_output(validate_dir, target_path=target)
    if audit_dir:
        constraints = load_audit_constraints(audit_dir)
        if constraints:
            enrich_attack_paths(attack_paths_file, constraints)

The two-tier search in find_audit_output() covers:
  1. Project sibling directories (same project, different run)
  2. Global out/ scan (match by checklist target_path)

Co-located (tier 1) is not searched because /audit and /validate
always produce separate output directories.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONSTRAINTS_FILENAME = "constraints.json"


def find_audit_output(
    validate_dir: Path,
    target_path: str = None,
) -> Optional[Path]:
    """Find the best sibling audit run directory with constraints.json.

    Two-tier search:
      1. Sibling directories under the same project parent
      2. Global out/ for audit runs matching the target path

    Returns the path to the audit run directory, or None.
    """
    validate_dir = Path(validate_dir)
    candidates = _collect_candidates(validate_dir, target_path)
    if not candidates:
        return None

    best = _pick_best(candidates)
    if best:
        logger.info("audit_bridge: selected %s", best)
    return best


def load_audit_constraints(audit_dir: Path) -> List[Dict[str, Any]]:
    """Load constraints from an audit run directory.

    Returns the list of constraint dicts, filtering out refuted ones.
    Returns empty list on any error.
    """
    path = Path(audit_dir) / CONSTRAINTS_FILENAME
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.debug("audit_bridge: failed to read %s", path, exc_info=True)
        return []

    if not isinstance(data, list):
        return []

    return [
        c for c in data
        if isinstance(c, dict) and c.get("status") != "refuted"
    ]


def enrich_attack_paths(
    attack_paths_file: Path,
    constraints: List[Dict[str, Any]],
) -> int:
    """Enrich attack-paths.json with audit constraints.

    For each attack path step that references a constrained function,
    adds a 'parameter_constraints' field with the relevant preconditions.
    This lets Stage B's hypothesis formation and SMT pre-flight use
    concrete parameter bounds.

    Returns the number of steps enriched.
    """
    attack_paths_file = Path(attack_paths_file)
    if not attack_paths_file.exists():
        return 0

    try:
        data = json.loads(attack_paths_file.read_text())
    except (json.JSONDecodeError, OSError):
        return 0

    if not isinstance(data, list):
        return 0

    constraint_index = _build_constraint_index(constraints)
    if not constraint_index:
        return 0

    enriched = 0
    for path_entry in data:
        if not isinstance(path_entry, dict):
            continue
        for step in path_entry.get("steps", []):
            if not isinstance(step, dict):
                continue
            func = step.get("function", "")
            file_path = step.get("file", "")
            if not func:
                continue

            matched = constraint_index.get(f"{file_path}:{func}", [])
            if not matched:
                matched = constraint_index.get(f":{func}", [])
            if matched:
                step["parameter_constraints"] = matched
                enriched += 1

    if enriched:
        try:
            tmp = attack_paths_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(attack_paths_file)
            logger.info(
                "audit_bridge: enriched %d attack-path steps with constraints",
                enriched,
            )
        except OSError:
            logger.debug(
                "audit_bridge: failed to write enriched attack-paths",
                exc_info=True,
            )
            return 0

    return enriched


def _build_constraint_index(
    constraints: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """Index constraints by file:function and bare :function.

    Returns a dict mapping keys to lists of simplified constraint
    records (kind, target, rule, violation).
    """
    index: Dict[str, List[Dict[str, str]]] = {}
    for c in constraints:
        func = c.get("function", "")
        file_path = c.get("file", "")
        if not func:
            continue

        entry = {
            "kind": c.get("kind", ""),
            "target": c.get("target", ""),
            "rule": c.get("rule", ""),
            "violation": c.get("violation", ""),
        }

        key = f"{file_path}:{func}"
        index.setdefault(key, []).append(entry)
        bare_key = f":{func}"
        index.setdefault(bare_key, []).append(entry)

    return index


def _collect_candidates(
    validate_dir: Path,
    target_path: Optional[str],
) -> List[Path]:
    """Gather audit run directories from sibling and global searches."""
    seen: set = set()
    results: List[Path] = []

    parent = validate_dir.parent
    for d in _search_audit_dirs(parent, exclude=validate_dir):
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            results.append(d)

    if target_path:
        try:
            from core.config import RaptorConfig
            out_root = RaptorConfig.get_out_dir()
        except Exception:
            out_root = None
        if out_root and out_root != parent:
            for d in _search_audit_dirs(out_root, exclude=validate_dir):
                resolved = d.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(d)

    return results


def _search_audit_dirs(
    parent: Path,
    exclude: Optional[Path] = None,
) -> List[Path]:
    """Find subdirectories containing constraints.json."""
    if not parent.is_dir():
        return []
    results = []
    try:
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            if exclude and child == exclude:
                continue
            if (child / CONSTRAINTS_FILENAME).exists():
                results.append(child)
    except OSError:
        pass
    return results


def _pick_best(candidates: List[Path]) -> Optional[Path]:
    """Pick the most recent candidate by mtime."""
    if not candidates:
        return None

    def _safe_mtime(p: Path) -> float:
        try:
            return (p / CONSTRAINTS_FILENAME).stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=_safe_mtime, reverse=True)
    return candidates[0]


ATTACK_CHAINS_FILENAME = "attack-chains.json"
SUMMARIES_FILENAME = "summaries.json"


def load_attack_chains(audit_dir: Path) -> List[Dict[str, Any]]:
    """Load attack-chains.json from an audit run directory.

    Attack chains describe multi-finding exploit paths synthesised by
    /audit's attacker_synthesis pass. /validate Stage B consumes these
    as pre-formed hypotheses — each chain becomes a candidate attack path
    to verify rather than discover from scratch.

    Returns list of chain dicts, empty on any error.
    """
    path = Path(audit_dir) / ATTACK_CHAINS_FILENAME
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.debug("audit_bridge: failed to read %s", path, exc_info=True)
        return []

    if not isinstance(data, list):
        return []

    return [c for c in data if isinstance(c, dict)]


def load_summaries(audit_dir: Path) -> Dict[str, Any]:
    """Load summaries.json from an audit run directory.

    Summaries capture per-function security behavior (taint rules,
    preconditions, postconditions). Consumed by:
    - /validate: callee contracts enrich attack-path step annotations
    - /understand --map: augments entry-point and sink annotations

    Returns dict mapping "file:function" → summary dict, empty on error.
    """
    path = Path(audit_dir) / SUMMARIES_FILENAME
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.debug("audit_bridge: failed to read %s", path, exc_info=True)
        return {}

    if isinstance(data, list):
        return {
            f"{s.get('file', '')}:{s.get('function', '')}": s
            for s in data if isinstance(s, dict)
        }
    elif isinstance(data, dict):
        return data

    return {}


def inject_chains_as_hypotheses(
    chains: List[Dict[str, Any]],
    attack_surface_file: Path,
) -> int:
    """Inject attack chains into attack-surface.json as pre-formed hypotheses.

    Each chain becomes an entry in attack_surface.hypotheses[] with
    status="imported" and source="audit_chain". Stage B sees these and
    can verify/disprove them rather than discovering from scratch.

    Returns number of hypotheses injected.
    """
    if not chains:
        return 0

    attack_surface_file = Path(attack_surface_file)
    if not attack_surface_file.exists():
        return 0

    try:
        surface = json.loads(attack_surface_file.read_text())
    except (json.JSONDecodeError, OSError):
        return 0

    if not isinstance(surface, dict):
        return 0

    hypotheses = surface.setdefault("hypotheses", [])
    injected = 0

    for chain in chains:
        hypothesis = {
            "id": f"audit_chain_{injected}",
            "source": "audit_chain",
            "status": "imported",
            "description": chain.get("description", ""),
            "primitives": chain.get("primitives", []),
            "findings": [f.get("finding_id", "") for f in chain.get("findings", [])],
            "goal": chain.get("goal", ""),
        }
        hypotheses.append(hypothesis)
        injected += 1

    if injected:
        try:
            tmp = attack_surface_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(surface, indent=2))
            tmp.replace(attack_surface_file)
            logger.info(
                "audit_bridge: injected %d attack-chain hypotheses", injected,
            )
        except OSError:
            logger.debug(
                "audit_bridge: failed to write hypotheses", exc_info=True,
            )
            return 0

    return injected


def enrich_with_summaries(
    attack_paths_file: Path,
    summaries: Dict[str, Any],
) -> int:
    """Enrich attack-path steps with callee contract information from summaries.

    For each step in an attack path, if a summary exists for that function,
    annotates the step with preconditions and taint rules. This gives
    /validate concrete contract data without re-running /audit.

    Returns number of steps enriched.
    """
    if not summaries:
        return 0

    attack_paths_file = Path(attack_paths_file)
    if not attack_paths_file.exists():
        return 0

    try:
        data = json.loads(attack_paths_file.read_text())
    except (json.JSONDecodeError, OSError):
        return 0

    if not isinstance(data, list):
        return 0

    enriched = 0
    for path_entry in data:
        if not isinstance(path_entry, dict):
            continue
        for step in path_entry.get("steps", []):
            if not isinstance(step, dict):
                continue
            func = step.get("function", "")
            file_path = step.get("file", "")
            if not func:
                continue

            key = f"{file_path}:{func}"
            summary = summaries.get(key)
            if not summary:
                continue

            step["callee_contract"] = {
                "preconditions": summary.get("preconditions", []),
                "taint_rules": summary.get("taint_rules", []),
                "postconditions": summary.get("returns", []),
            }
            enriched += 1

    if enriched:
        try:
            tmp = attack_paths_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(attack_paths_file)
            logger.info(
                "audit_bridge: enriched %d steps with callee contracts", enriched,
            )
        except OSError:
            return 0

    return enriched
