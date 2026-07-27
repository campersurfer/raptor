"""Data loaders for the /audit orchestrator.

Pure I/O helpers that load optional side-channel data (variants, fuzz
coverage, exploit feedback, taint approximations) from the run
directory.  Separated from orchestrator.py for readability — no
orchestrator state is mutated.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def load_variants(out_dir: Path) -> Set[str]:
    """Load variants.json from /understand --hunt if present.

    Returns a set of "file:function" keys that match known variant
    patterns, used to boost priority.
    """
    path = out_dir / "variants.json"
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        targets: Set[str] = set()
        for variant in data if isinstance(data, list) else data.get("variants", []):
            fp = variant.get("file", "")
            fn = variant.get("function", "")
            if fp and fn:
                targets.add(f"{fp}:{fn}")
        if targets:
            logger.info("variants: %d pattern-match targets loaded", len(targets))
        return targets
    except Exception:
        logger.debug("variants.json load failed", exc_info=True)
        return set()


def load_coverage_records(out_dir: Path) -> List[Dict[str, Any]]:
    """Load coverage-record.json if present."""
    path = out_dir / "coverage-record.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]
    except (json.JSONDecodeError, OSError):
        return []


def load_exploit_feedback(out_dir: Path, load_feedback_state, FeedbackState):
    """Load exploit feedback state from the project or run directory."""
    candidates = [
        out_dir / "exploit-feedback.json",
        out_dir.parent / "exploit-feedback.json",
    ]
    for path in candidates:
        if path.is_file():
            state = load_feedback_state(path)
            if state.outcomes:
                return state
    return FeedbackState()


def load_fuzz_coverage(out_dir: Path) -> Optional[Dict[str, Any]]:
    """Load fuzz coverage data if present."""
    path = out_dir / "coverage-fuzz.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fuzz_coverage_for(
    fuzz_data: Dict[str, Any],
    file_path: str,
    function_name: str,
) -> Optional[Dict[str, Any]]:
    """Extract fuzz coverage for a specific function."""
    flat_key = f"{file_path}:{function_name}"
    if flat_key in fuzz_data:
        return fuzz_data[flat_key]
    file_data = fuzz_data.get("files", {}).get(file_path, {})
    func_data = file_data.get("functions", {}).get(function_name)
    if func_data:
        return func_data
    return None


def load_or_build_taint_approx(
    target_path: Optional[Path],
    out_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Load cached taint approximations or build and persist them."""
    if out_dir:
        cache_path = out_dir / "taint-approx.json"
        if cache_path.exists():
            try:
                from core.json import load_json
                cached = load_json(cache_path)
                if cached:
                    logger.info(
                        "taint_approx: loaded %d cached results", len(cached),
                    )
                    return cached
            except Exception:
                pass
    results = _build_taint_approx(target_path)
    if results and out_dir:
        try:
            from core.json import save_json
            save_json(out_dir / "taint-approx.json", results)
        except Exception:
            pass
    return results


def _build_taint_approx(
    target_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Build tree-sitter taint approximations for all C/C++ files."""
    if not target_path or not target_path.is_dir():
        return None

    try:
        from core.analysis.taint_approx import (
            extract_taint_approx_c,
            extract_taint_approx_cpp,
        )
    except ImportError:
        return None

    results: Dict[str, Any] = {}
    c_exts = {".c", ".h"}
    cpp_exts = {".cc", ".cpp", ".cxx", ".hpp"}

    for path in target_path.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in c_exts and suffix not in cpp_exts:
            continue

        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue

        rel = str(path.relative_to(target_path))

        if suffix in c_exts:
            approxes = extract_taint_approx_c(content)
        else:
            approxes = extract_taint_approx_cpp(content)

        for func_name, approx in approxes.items():
            results[f"{rel}:{func_name}"] = approx

    if results:
        logger.info("taint_approx: %d functions analysed", len(results))
    return results or None


def recreate_coverage_from_annotations(
    coverage_records: List[Dict[str, Any]],
    annotations_dir: Path,
    checklist: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Re-create coverage entries from annotations when runs are cleaned.

    Annotations survive /project clean; coverage records don't. When an
    annotation exists for a function but no coverage record covers it,
    synthesize a coverage record so compute_gaps doesn't re-review it.
    """
    if not annotations_dir.is_dir():
        return coverage_records

    covered = set()
    for rec in coverage_records:
        for fa in rec.get("functions_analysed", []):
            covered.add((fa.get("file", ""), fa.get("function", "")))

    _META_STATUS_RE = re.compile(r"status=(\S+)")

    synthetic = []
    for ann_path in annotations_dir.rglob("*.md"):
        rel = ann_path.relative_to(annotations_dir)
        source_file = str(rel).removesuffix(".md")
        try:
            text = ann_path.read_text(errors="replace")
        except OSError:
            continue
        current_func = None
        current_status = "clean"
        for line in text.splitlines():
            if line.startswith("## ") and not line.startswith("## _"):
                if current_func and (source_file, current_func) not in covered:
                    synthetic.append({
                        "file": source_file,
                        "function": current_func,
                        "status": current_status,
                        "hash": None,
                    })
                    covered.add((source_file, current_func))
                current_func = line[3:].strip()
                current_status = "clean"
            elif current_func and line.startswith("<!-- meta:"):
                m = _META_STATUS_RE.search(line)
                if m:
                    current_status = m.group(1)
        if current_func and (source_file, current_func) not in covered:
            synthetic.append({
                "file": source_file,
                "function": current_func,
                "status": current_status,
                "hash": None,
            })
            covered.add((source_file, current_func))

    if synthetic:
        logger.info(
            "coverage: re-created %d records from surviving annotations",
            len(synthetic),
        )
        coverage_records = list(coverage_records)
        coverage_records.append({
            "tool": "audit",
            "functions_analysed": synthetic,
            "files_examined": sorted({s["file"] for s in synthetic}),
        })

    return coverage_records
