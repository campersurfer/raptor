"""Run the /audit calibration corpus and score results.

Usage:
    python3 -m core.audit.corpus.run_corpus [options]

Steps:
    1. Load labels from core/audit/corpus/labels/
    2. Fetch pinned sources if missing (--fetch)
    3. Build checklist + context map for each target
    4. Run /audit's orchestrator against the labeled functions
    5. Score each outcome against ground truth
    6. Emit JSON + detailed summary with cost, duration, per-function verdicts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent
LABELS_DIR = CORPUS_DIR / "labels"
FIXTURES_DIR = Path("out/audit-corpus-fixtures")


def _fetch_source(repo_key: str, sha: str) -> Path:
    """Fetch a pinned source tree.  Returns the local path."""
    dest = FIXTURES_DIR / repo_key
    if dest.is_dir():
        result = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        current = result.stdout.strip()
        if current == sha:
            return dest
        logger.info("SHA mismatch for %s: %s != %s, re-fetching",
                     repo_key, current[:12], sha[:12])
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", sha],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", sha],
            check=True, capture_output=True, timeout=30,
        )
        return dest

    logger.warning(
        "Source %s not found at %s. Run with --fetch or clone manually. "
        "See SOURCES.md for instructions.",
        repo_key, dest,
    )
    return dest


def _resolve_source_dirs(
    labels: List[Any],
    *,
    do_fetch: bool = False,
) -> Dict[str, Path]:
    """Resolve and optionally fetch source directories for all labels."""
    repos: Dict[str, str] = {}
    for label in labels:
        key = label.source.repo
        if key not in repos:
            repos[key] = label.source.sha

    resolved = {}
    for key, sha in repos.items():
        if do_fetch:
            resolved[key] = _fetch_source(key, sha)
        else:
            dest = FIXTURES_DIR / key
            if not dest.is_dir():
                logger.warning("Source %s not found at %s", key, dest)
            resolved[key] = dest

    return resolved


def _verify_labels(
    labels: List[Any],
    source_dirs: Dict[str, Path],
) -> List[str]:
    """Verify that labeled functions exist in fetched sources."""
    errors = []
    for label in labels:
        src_dir = source_dirs.get(label.source.repo)
        if src_dir is None or not src_dir.is_dir():
            errors.append(f"{label.function_id}: source dir missing")
            continue
        src_file = src_dir / label.source.file
        if not src_file.is_file():
            errors.append(f"{label.function_id}: file not found: {src_file}")
    return errors


def _build_checklist(
    target_dir: Path,
    out_dir: Path,
) -> bool:
    """Build checklist for a target (mechanical, no LLM).

    context-map.json requires an LLM pass (/understand --map) and is
    not built here.  raptor-audit run will search for it via the bridge.
    """
    raptor_dir = Path(os.environ["RAPTOR_DIR"])
    env = {**os.environ, "CLAUDECODE": "1", "_RAPTOR_TRUSTED": "1"}

    checklist_path = out_dir / "checklist.json"
    if not checklist_path.exists():
        print(f"  Building checklist for {target_dir.name}...", flush=True)
        cp = subprocess.run(
            [sys.executable,
             str(raptor_dir / "libexec" / "raptor-build-checklist"),
             str(target_dir), "--out", str(out_dir)],
            env=env, capture_output=True, text=True,
        )
        if cp.returncode != 0:
            print(f"  checklist build failed: {cp.stderr.strip()[:200]}",
                  file=sys.stderr)
            return False

    return True


def _poll_progress(
    log_path: Path,
    seen: int,
    labeled_ids: set,
) -> int:
    """Print new audit-log entries since last poll.  Returns new seen count."""
    if not log_path.exists():
        return seen
    with open(log_path) as f:
        lines = f.readlines()
    new_count = len(lines)
    if new_count <= seen:
        return seen
    for line in lines[seen:]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("action") != "orchestrator_review":
            continue
        key = entry.get("key", "")
        status = entry.get("status", "?")
        marker = " *" if key in labeled_ids else ""
        char = {"clean": ".", "suspicious": "?", "finding": "!",
                "dormant": "~", "error": "x"}.get(status, ".")
        print(f"  [{new_count}] {key} -> {status} {char}{marker}",
              flush=True)
    return new_count


def _run_audit(
    labels: List[Any],
    source_dirs: Dict[str, Path],
    *,
    model: str = "",
    out_dir: Optional[Path] = None,
    two_pass: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Run /audit's orchestrator against labeled functions.

    Returns (results, run_dirs) — results is a list of per-function
    outcome dicts; run_dirs lists the output directories used (for
    --debug journal retrieval).
    """
    from .label import FunctionLabel

    by_repo: Dict[str, List[FunctionLabel]] = {}
    for label in labels:
        by_repo.setdefault(label.source.repo, []).append(label)

    results = []
    run_dirs: List[Path] = []
    for repo_key, repo_labels in by_repo.items():
        src_dir = source_dirs.get(repo_key)
        if src_dir is None or not src_dir.is_dir():
            for label in repo_labels:
                results.append({
                    "function_id": label.function_id,
                    "bug_class": label.bug_class,
                    "expected": label.expected_status,
                    "actual": "error",
                    "match": False,
                    "hypothesis": "",
                    "evidence_tool": "",
                    "model": model,
                    "cost_usd": 0.0,
                    "duration_s": 0.0,
                    "error": f"source dir missing: {repo_key}",
                })
            continue

        outcomes, audit_dir = _run_audit_on_target(
            src_dir, repo_labels, model=model, out_dir=out_dir,
            two_pass=two_pass,
        )
        if audit_dir:
            run_dirs.append(audit_dir)
        for label in repo_labels:
            outcome = outcomes.get(label.function_id)
            if outcome is None:
                actual = "error"
                hypothesis = ""
                evidence_tool = ""
                cost = 0.0
                dur = 0.0
            else:
                actual = outcome["status"]
                hypothesis = outcome.get("hypothesis", "")
                evidence_tool = outcome.get("evidence_tool", "")
                cost = outcome.get("cost_usd", 0.0)
                dur = outcome.get("duration_s", 0.0)

            expected = label.expected_status
            match = _status_matches(expected, actual)

            results.append({
                "function_id": label.function_id,
                "bug_class": label.bug_class,
                "expected": expected,
                "actual": actual,
                "match": match,
                "hypothesis": hypothesis,
                "evidence_tool": evidence_tool,
                "model": model,
                "cost_usd": cost,
                "duration_s": dur,
            })

    return results, run_dirs


def _status_matches(expected: str, actual: str) -> bool:
    """Check if actual status satisfies the expected ground truth."""
    if expected == "finding":
        return actual == "finding"
    if expected == "clean":
        return actual in ("clean", "dormant")
    if expected == "dormant":
        return actual in ("dormant", "clean")
    return False


def _run_audit_on_target(
    target_dir: Path,
    labels: List[Any],
    *,
    model: str = "",
    out_dir: Optional[Path] = None,
    two_pass: bool = False,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run /audit orchestrator on a target.

    Returns (outcomes_by_function_id, audit_output_dir).
    """
    if out_dir is None:
        out_dir = Path(f"out/audit-corpus-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _build_checklist(target_dir, out_dir)

    raptor_dir = Path(os.environ["RAPTOR_DIR"])
    cmd = [
        sys.executable,
        str(raptor_dir / "libexec" / "raptor-audit"),
        "run",
        str(target_dir),
        "--out", str(out_dir),
        "--max-cost", "50",
    ]
    if model:
        cmd.extend(["--model", model])
    if two_pass:
        cmd.append("--two-pass")

    env = {**os.environ, "CLAUDECODE": "1", "_RAPTOR_TRUSTED": "1"}
    labeled_ids = {label.function_id for label in labels}
    log_path = out_dir / ".audit-log.jsonl"

    # 7200s = 2h hard deadline for a single audit target.
    _AUDIT_DEADLINE_S = 7200

    print(f"  Audit started: {target_dir}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    seen = 0
    deadline = t0 + _AUDIT_DEADLINE_S
    while proc.poll() is None:
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            logger.error("Audit timed out after %ds", _AUDIT_DEADLINE_S)
            break
        time.sleep(3.0)
        seen = _poll_progress(log_path, seen, labeled_ids)

    _poll_progress(log_path, seen, labeled_ids)

    wall_s = time.monotonic() - t0
    rc = proc.returncode
    if rc != 0:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode(
            errors="replace")
        logger.error("Audit run failed (rc=%d):\n%s", rc, stderr[-2000:])

    print(f"  Audit finished in {wall_s:.0f}s (rc={rc})", flush=True)

    outcomes_by_id: Dict[str, Dict[str, Any]] = {}
    if log_path.exists():
        with open(log_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if entry.get("action") != "orchestrator_review":
                    continue
                key = entry.get("key", "")
                if key:
                    outcomes_by_id[key] = entry

    return outcomes_by_id, out_dir


def _write_results(
    results: List[Dict[str, Any]],
    output: Path,
) -> None:
    """Write results to a JSON file."""
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def _format_detail_table(results: List[Dict[str, Any]]) -> str:
    """Format per-function detail table."""
    lines = []
    lines.append(f"{'Function':<45} {'Expected':<10} {'Actual':<12} "
                 f"{'Match':<6} {'Evidence':<25} {'Cost':>7}")
    lines.append("-" * 110)
    for r in results:
        fid = r["function_id"]
        if len(fid) > 44:
            fid = "..." + fid[-41:]
        match_str = "yes" if r["match"] else "NO"
        evidence = r.get("evidence_tool", "")
        if len(evidence) > 24:
            evidence = evidence[:21] + "..."
        cost = r.get("cost_usd", 0.0)
        lines.append(
            f"{fid:<45} {r['expected']:<10} {r['actual']:<12} "
            f"{match_str:<6} {evidence:<25} ${cost:>6.4f}"
        )
    return "\n".join(lines)


def _format_summary(
    results: List[Dict[str, Any]],
    wall_s: float,
    model: str,
) -> str:
    """Format the full summary block."""
    from .corpus_metrics import check_gate, compute_metrics, format_report

    aggregate, per_class = compute_metrics(results)
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    total_llm_s = sum(r.get("duration_s", 0.0) for r in results)
    matched = sum(1 for r in results if r.get("match"))
    mismatched = [r for r in results if not r.get("match")]

    lines = []
    lines.append("=" * 70)
    lines.append("Corpus run complete")
    lines.append(f"  Model: {model or 'default'}")
    lines.append(f"  Labels: {len(results)}")
    lines.append(f"  Matched: {matched}/{len(results)}")
    lines.append(f"  Cost: ${total_cost:.4f}")
    lines.append(f"  Wall clock: {wall_s:.0f}s ({wall_s/60:.1f}m)")
    lines.append(f"  LLM time: {total_llm_s:.0f}s ({total_llm_s/60:.1f}m)")
    lines.append("")
    lines.append(format_report(aggregate, per_class, model=model))
    lines.append("")
    lines.append(_format_detail_table(results))

    if mismatched:
        lines.append("")
        lines.append("Mismatches:")
        for r in mismatched:
            hyp = r.get("hypothesis", "")
            if len(hyp) > 80:
                hyp = hyp[:77] + "..."
            lines.append(f"  {r['function_id']}: "
                         f"expected={r['expected']} got={r['actual']} "
                         f"evidence={r.get('evidence_tool', '')}")
            if hyp:
                lines.append(f"    hypothesis: {hyp}")

    gates = check_gate(aggregate, per_class, results)
    if gates:
        lines.append("")
        for g in gates:
            lines.append(f"GATE FAIL: {g}")
    else:
        lines.append("")
        lines.append("All gates passed.")

    return "\n".join(lines)


def _save_debug(
    results: List[Dict[str, Any]],
    run_dirs: List[Path],
    output_path: Path,
) -> None:
    """Save LLM reasoning alongside results for diagnosis.

    Collects review-journal.jsonl entries from each run directory and
    writes a per-function debug JSONL next to the results file.  Each
    line has the function_id, verdict, hypotheses, and verdict_rationale.
    """
    debug_path = output_path.with_suffix(".debug.jsonl")

    journal_entries: Dict[str, Dict[str, Any]] = {}
    for d in run_dirs:
        jpath = d / "review-journal.jsonl"
        if not jpath.exists():
            continue
        with open(jpath) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                fid = entry.get("file", "") + ":" + entry.get("function", "")
                if fid != ":":
                    journal_entries[fid] = entry

    labeled_ids = {r["function_id"] for r in results}
    with open(debug_path, "w") as f:
        for fid in sorted(labeled_ids):
            je = journal_entries.get(fid, {})
            hypotheses = je.get("hypotheses", [])
            record = {
                "function_id": fid,
                "verdict": je.get("verdict", ""),
                "hypotheses": hypotheses,
                "cwe": je.get("cwe", ""),
                "verdict_rationale": je.get("verdict_rationale", ""),
                "counter_hypothesis": je.get("counter_hypothesis", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Debug reasoning written to {debug_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run /audit calibration corpus",
    )
    parser.add_argument(
        "--class", dest="bug_class", default=None,
        help="Run only one bug class (e.g. aliasing, honeyslop)",
    )
    parser.add_argument(
        "--label", dest="label_id", default=None,
        help="Run only one label by function_id (e.g. c/heartbeat.c:read_u16_be)",
    )
    parser.add_argument(
        "--model", default="",
        help="LLM model to use (default: orchestrator default)",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch/update pinned sources before running",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for the audit run",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("corpus-results.json"),
        help="Path for the results JSON (default: corpus-results.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and verify labels without running audit",
    )
    parser.add_argument(
        "--two-pass", action="store_true",
        help="Split reasoning from classification into two LLM calls",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save LLM reasoning alongside CSV for diagnosis",
    )
    args = parser.parse_args(argv)

    from .label import load_all_labels

    labels = load_all_labels(bug_class=args.bug_class)

    if args.label_id:
        labels = [lb for lb in labels if lb.function_id == args.label_id]

    if not labels:
        print("No labels found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(labels)} label(s)", end="")
    if args.bug_class:
        print(f" (class: {args.bug_class})", end="")
    if args.label_id:
        print(f" (id: {args.label_id})", end="")
    print()

    source_dirs = _resolve_source_dirs(labels, do_fetch=args.fetch)
    errors = _verify_labels(labels, source_dirs)
    if errors:
        print(f"{len(errors)} label verification error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        if not args.dry_run:
            return 1

    if args.dry_run:
        print("Dry run — labels verified, not running audit.")
        for label in labels:
            print(f"  {label.function_id} ({label.bug_class}) "
                  f"expected={label.expected_status}")
        return 0

    print(f"Running audit (model: {args.model or 'default'})...", flush=True)
    t0 = time.monotonic()
    results, run_dirs = _run_audit(
        labels, source_dirs,
        model=args.model, out_dir=args.out,
        two_pass=args.two_pass,
    )
    wall_s = time.monotonic() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_results(results, args.output)
    print(f"\nResults written to {args.output}")

    if args.debug:
        _save_debug(results, run_dirs, args.output)

    print()
    print(_format_summary(results, wall_s, args.model))

    return 0


if __name__ == "__main__":
    sys.exit(main())
