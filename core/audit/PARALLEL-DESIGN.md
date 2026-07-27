# /audit parallel task-graph executor — design

## Problem

The orchestrator reviews functions serially. The LLM round trip (~2-10s per
function) dominates wall time. A 500-function codebase takes 30-60 minutes.
With 8 concurrent LLM calls it could take 4-8 minutes.

## Architecture

### Three layers

```
OrchestratorConfig + prep
        │
        ▼
  ┌─────────────┐
  │  TaskGraph   │  ← DAG of ReviewTask nodes, edges = callee dependencies
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Executor    │  ← bounded-concurrency scheduler, budget-gated
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Collector   │  ← merges outcomes, flushes writes, builds OrchestratorResult
  └─────────────┘
```

### TaskGraph

Built from the workqueue + call_edges after Phase 1 prep completes.

```python
@dataclass
class ReviewTask:
    key: str                          # "file.py:function_name"
    gap: dict[str, Any]               # checklist gap record
    depends_on: frozenset[str]        # keys of callees in the workqueue
    priority: float                   # from score_functions()
    triage_bucket: str                # from classify_all()

@dataclass
class TaskGraph:
    tasks: dict[str, ReviewTask]
    ready: set[str]                   # tasks with all deps satisfied (or no deps)

    def mark_complete(self, key: str) -> list[str]:
        """Mark task done, return newly-ready task keys."""

    def pop_ready(self, n: int) -> list[ReviewTask]:
        """Pop up to n ready tasks, highest priority first."""
```

Dependency rule: task A depends on task B iff there is a call edge from A's
function to B's function AND B is in the workqueue. Functions not in the
workqueue (already reviewed, out of scope, prefilter-skipped) are not
dependencies — their mechanical summaries are already available.

### Executor

```python
@dataclass
class ExecutorConfig:
    max_workers: int = 4              # concurrent LLM calls
    budget_check_interval: int = 1    # check budget before each dispatch

async def run_executor(
    graph: TaskGraph,
    review_fn: Callable,
    shared: SharedState,
    config: OrchestratorConfig,
    executor_config: ExecutorConfig,
) -> list[ReviewOutcome]:
```

The executor is an async loop:

1. Pop up to `max_workers` ready tasks from the graph
2. For each, check budget — if exceeded, stop
3. Dispatch each as a coroutine: build_context → review_fn → post_process
4. On completion: publish taint summary to SharedState, mark task complete
   in graph (unlocking dependents), submit outcome to Collector
5. Repeat until graph is empty or budget exhausted

When `max_workers=1`, this is identical to the current serial loop.
No behavioural change unless the operator opts in.

### SharedState

Append-only shared context that workers read and write concurrently.

```python
class SharedState:
    # Written by workers after each review
    taint_summaries: dict[str, FunctionSummary]   # thread-safe dict
    observations: list[dict[str, str]]            # append-only log
    discovered_evidence: dict[str, Any]            # append-only
    checker_stats: dict[str, tuple[int, int]]     # rule_id → (tp, total)

    # Read-only after prep (no lock needed)
    context_map: dict[str, Any]
    checklist: dict[str, Any]
    evidence_index: dict[str, EvidenceRecord]
    provenance_map: dict[str, list[dict]]
    security_decision_keys: frozenset[str]
    feeds_security_keys: frozenset[str]
    threat_model: dict[str, Any] | None
    # ... all other prep outputs
```

Taint summaries are the critical write path — a callee's summary must be
visible before its caller starts. The task graph guarantees this: a caller
task doesn't become ready until its callee tasks complete, and completion
publishes the summary before unlocking dependents.

Observations/evidence/checker_stats are eventual-consistency: a worker
reads whatever's accumulated so far. No barrier needed. Workers reviewing
higher-priority functions (which run first) generate observations that
lower-priority workers (which start later) benefit from naturally.

### Collector

Receives outcomes from workers and handles all I/O:

```python
class Collector:
    def submit(self, outcome: ReviewOutcome, gap: dict) -> None:
        """Thread-safe. Buffers outcome."""

    def flush(self) -> None:
        """Write buffered outcomes to disk."""
        # - append_audit_log: batch JSONL append
        # - record_review: one file per function (no conflict)
        # - mark_checked: accumulate keys, single checklist rewrite at flush

    def build_result(self) -> OrchestratorResult:
        """Aggregate counters from all outcomes."""
```

`mark_checked` is the only contended write — it currently rewrites the
whole checklist.json on every function. The Collector accumulates checked
keys in memory and writes once at the end (or periodically).

### Worker body (one per task)

This is the current per-function logic extracted verbatim:

```python
async def review_task(
    task: ReviewTask,
    shared: SharedState,
    review_fn: Callable,
    config: OrchestratorConfig,
    collector: Collector,
) -> ReviewOutcome:
    gap = task.gap
    ctx = build_context(gap, shared, config)       # ~lines 987-1300
    ctx = enrich_with_gates(ctx, gap, shared)      # mechanical gates A-E

    if config.prefilter:
        pf = run_prefilter(ctx, config)
        if pf.skip_llm:
            outcome = clean_outcome(gap, pf)
            collector.submit(outcome, gap)
            return outcome

    outcome = review_fn(ctx, config)               # LLM call
    outcome = post_process(outcome, ctx, config,   # gates, validation
                           shared, gap)             # ~lines 1300-1600

    # Publish summary for dependents
    if outcome.review_result:
        summary = summary_from_review_result(...)
        shared.taint_summaries[task.key] = summary

    collector.submit(outcome, gap)
    return outcome
```

## Phase mapping

### Phase 1: Prep (serial, unchanged)

Lines 263-780 of current orchestrator. Produces all read-only shared state.
No change except output goes into SharedState instead of local variables.

### Phase 2: Main review (parallel via executor)

The current serial loop (lines 780-1645) becomes `run_executor()`.
The per-function body becomes `review_task()`.

### Phase 3: Post-loop passes (serial or pipelined)

- Joern re-review: tasks that were reviewed before Joern finished.
  Could be a second executor run with the enriched evidence index.
- Deepen suspicious: re-review suspicious outcomes with deeper context.
  These are independent of each other — another executor run.
- Iterative re-review: serial (reads prior pass outcomes).
- Promote/resolve: serial (reads all outcomes).

Phases 3a (Joern re-review) and 3b (deepen) are themselves parallelisable
via the same executor — they're just new task graphs over subsets of outcomes.

### Phase 4: Post-loop analysis (parallel, no LLM)

All independent: attack chain synthesis, IRIS refinement, post-loop pattern
checks, postcondition verification. Can run as concurrent tasks or
asyncio.gather.

### Phase 5: Output (serial, trivial)

Findings export, SARIF, measurement, cost tracker. No change.

## Refactoring plan

### Step 1: Extract SharedState ✅

`core/audit/shared_state.py` — 30 read-only + 6 mutable fields, `from_prep()`
factory. SharedState constructed in `_run_audit_body` before the main loop.

### Step 2: Extract per-function body ✅

`review_one_function()` at `orchestrator.py:266` — 752-line standalone function.
Local aliases for SharedState fields. `continue` → `return`, budget `break` → `raise`.
Serial loop calls it; 2661 existing tests pass unchanged.

### Step 3: Extract Collector ✅

`core/audit/collector.py` — batches `mark_checked` (single checklist.json rewrite)
and `append_audit_log` (single JSONL write). `record_review` stays per-call (one
file per function, no contention). `review_one_function` accepts optional `collector`
param; when provided, uses it instead of `_commit_outcome`. 8 tests.

### Step 4: Build TaskGraph ✅

`core/audit/task_graph.py` — DAG of ReviewTask nodes. `from_workqueue()` builds
edges from call_edges. `pop_ready(n)` returns highest-priority ready tasks.
`mark_complete()` unlocks dependents. `serial_order()` for deterministic replay.
13 tests.

### Step 5: Build Executor ✅

`core/audit/executor.py` — `run_executor_sync()` drives review through TaskGraph.
Serial path (max_workers=1): identical to current loop. Async path (max_workers>1):
bounded concurrency via semaphore + run_in_executor. `review_one_fn` injectable
for testing. `budget_check` callable for budget gating. `on_tick` callback for
per-iteration side effects (Joern drain). `derive_max_workers(model)` computes
safe concurrency from RPM (rpm // 2, capped at 32). 13 tests.

`OrchestratorConfig.max_workers`: 1 = serial (default), 0 = auto (derive from
model RPM), N = explicit override (operator's responsibility).

### Step 6: Thread-safety audit ✅

**Fixed:**

- `constraints`: `shared._constraints_lock` (threading.Lock) guards the
  read-modify-write in `_extract_and_propagate()`.
- `expansion_budget.try_expand()`: `threading.Lock` inside `ExpansionBudget`
  protects the check-then-act sequence.
- `checker_library.add()` / `.record_match()` / `.retire_low_precision()`:
  `threading.Lock` inside `CheckerLibrary` guards all mutations + disk writes.
- `workqueue.append()` (synthesis): replaced with `shared.synthesis_queue`
  (list.append is GIL-atomic). After the main executor pass, a second
  executor pass drains the queue if budget permits.

**Already safe (no change needed):**

- `taint_summary_results` (dict): TaskGraph guarantees callees complete
  before callers; independent tasks write different keys.
- `session_observations` (list): list.append is GIL-atomic.
- `discovered_evidence` (dict): each task writes its own key.
- `result` counters (+=): GIL-atomic in CPython for simple int operations.

### Step 7: Wire executor into orchestrator ✅

Replaced the manual `for gap in workqueue` loop in `_run_audit_body` with
`run_executor_sync()`. Joern-drain logic moved to an `on_tick` closure that
the executor calls before each dispatch. `TaskGraph.from_workqueue()` builds
the DAG from the existing workqueue + call_edges.

`max_workers=0` (auto) resolves via `derive_max_workers(config.models[0])`:
`rpm_for(model) // 2`, clamped to [1, 32]. This is now the default.
Explicit values pass through. 2695 tests pass.

### Step 8: Validate

Run /audit on core/audit/ with max_workers=1 (identical to current).
Compare outcomes. Then run with max_workers > 1 and compare again.
Differences indicate ordering-dependent behaviour that needs fixing.

## Config surface

```python
@dataclass
class OrchestratorConfig:
    # ... existing fields ...
    max_workers: int = 0  # 0 = auto (derive from model RPM), 1 = serial
```

CLI: `/audit ./target --workers 4` or `/audit ./target --workers 1` (serial)

## Risks

1. **Non-determinism**: parallel execution means review order varies between
   runs. Session observations arrive in different order. Outcomes should be
   deterministic given the same taint summaries (guaranteed by task graph)
   but observations add noise. Mitigation: log the execution order for
   reproducibility.

2. **LLM rate limits**: too many concurrent calls → 429 errors. Mitigation:
   exponential backoff per worker, or lower max_workers.

3. **Budget overshoot**: N workers dispatch simultaneously before any
   completes, spending N × cost before the budget check fires. Mitigation:
   pre-deduct estimated cost before dispatch, refund on completion.

4. **Debugging**: parallel failures harder to trace. Mitigation: per-task
   structured logging with task key prefix.
