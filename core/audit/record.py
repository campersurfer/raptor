"""Coverage-audit record emission.

Updates ``coverage-audit.json`` via ``core.coverage.record`` when
the audit workflow reviews a function.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({"clean", "suspicious", "finding", "error", "dormant"})


def _resolve_annotations_dir(out_dir: Path) -> Path:
    """Resolve annotations directory to project level when possible.

    Project runs have out_dir = project_dir/<run_name>/, so
    out_dir.parent is the project directory. Annotations at the project
    level survive /project clean (which deletes run dirs).

    Detection: a run dir contains .raptor-run.json (written by
    raptor-run-lifecycle start). If present, the parent is the
    project directory.
    """
    run_marker = out_dir / ".raptor-run.json"
    if run_marker.exists():
        project_dir = out_dir.parent
        if project_dir and project_dir != out_dir:
            return project_dir / "annotations"
    return out_dir / "annotations"


def record_review(
    *,
    out_dir: Path,
    target_path: Path,
    file_path: str,
    function_name: str,
    status: str,
    body: str,
    line_start: int = 0,
    line_end: Optional[int] = None,
    cwe: Optional[str] = None,
    strategies: Optional[List[str]] = None,
    checked_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record a single function review.

    Updates the coverage-audit tracking data. Returns the record dict.

    Args:
        out_dir: Run output directory.
        target_path: Root of the target codebase.
        file_path: Relative path to the source file.
        function_name: Name of the reviewed function.
        status: One of clean/suspicious/finding/error.
        body: Annotation prose (hypotheses tested, tool evidence).
        line_start: Function start line.
        line_end: Function end line.
        cwe: CWE identifier if applicable.
        strategies: Strategies applied during review.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}; must be one of {sorted(VALID_STATUSES)}"
        )

    source_hash = _compute_hash(target_path, file_path, line_start, line_end)

    record = {
        "file": file_path,
        "function": function_name,
        "status": status,
        "hash": source_hash,
        "strategies": strategies or [],
    }
    if cwe:
        record["cwe"] = cwe
    if checked_by:
        record["checked_by"] = checked_by

    _update_coverage_audit(out_dir, file_path, function_name, record)

    return record


def load_audit_log(out_dir: Path) -> List[Dict[str, Any]]:
    """Load the audit log (one JSON record per line)."""
    log_path = out_dir / ".audit-log.jsonl"
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_audit_log(out_dir: Path, entry: Dict[str, Any]) -> None:
    """Append an entry to the audit log."""
    log_path = out_dir / ".audit-log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _compute_hash(
    target_path: Path,
    file_path: str,
    line_start: int,
    line_end: Optional[int],
) -> Optional[str]:
    """Compute source hash for staleness detection."""
    full_path = target_path / file_path
    if not full_path.exists():
        return None

    try:
        from core.annotations.storage import compute_function_hash
        end = line_end if line_end is not None else line_start
        return compute_function_hash(full_path, line_start, end)
    except Exception:
        logger.debug("hash computation failed for %s:%d", file_path, line_start)
        return None



def _update_coverage_audit(
    out_dir: Path,
    file_path: str,
    function_name: str,
    record: Dict[str, Any],
) -> None:
    """Update coverage-audit.json with the new record."""
    audit_path = out_dir / "coverage-audit.json"

    if audit_path.exists():
        try:
            with open(audit_path) as f:
                audit_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("corrupt coverage-audit.json, reinitialising")
            audit_data = None
        if not isinstance(audit_data, dict):
            audit_data = None
    else:
        audit_data = None
    if audit_data is None:
        audit_data = {
            "tool": "audit",
            "files_examined": [],
            "functions_analysed": [],
        }

    functions = audit_data.setdefault("functions_analysed", [])
    existing = None
    for entry in functions:
        if entry.get("file") == file_path and entry.get("function") == function_name:
            existing = entry
            break
    func_record = {
        "file": file_path,
        "function": function_name,
        "status": record["status"],
        "hash": record.get("hash"),
    }
    if record.get("checked_by"):
        func_record["checked_by"] = record["checked_by"]
    if existing is not None:
        existing.update(func_record)
    else:
        functions.append(func_record)

    if file_path not in audit_data.get("files_examined", []):
        audit_data.setdefault("files_examined", []).append(file_path)

    fd, tmp = tempfile.mkstemp(dir=str(audit_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(audit_data, f, indent=2)
        os.replace(tmp, str(audit_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
