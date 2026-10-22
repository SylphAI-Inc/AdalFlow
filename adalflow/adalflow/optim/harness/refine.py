"""Task-level harness refinement: propose -> probe -> compose -> blind select.

Follows the task-level recursion of Harness² (Xu et al., 2026, arXiv:2610.14224),
implemented independently. No labels are used anywhere: edits are justified by the
agent's own trajectories, and the final pick is made by a selector that sees only the
anonymized deliverables.

Parallel mode (budget k): k base runs; 2k candidates (k procedure + k domain edits of
the base harness), each probed once; k compositions over all candidates, each probed
once. Sequential mode: k rounds; each round edits the current checkpoint once per
dimension, probes both, and composes them into the next checkpoint. Both spend
R = 4k rollouts and P = 3k editor sessions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import string
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from adalflow.optim.harness.harness import (
    DEFAULT_ALWAYS_ON_CAP,
    Harness,
    HarnessValidationError,
)
from adalflow.optim.harness.ledger import HarnessBudgetLedger
from adalflow.optim.harness.probe import HarnessExecutor, ProbeResult
from adalflow.optim.harness.roles import (
    DOMAIN,
    PROCEDURE,
    BlindSelector,
    EditProposal,
    HarnessEditor,
    RoleOutputError,
)

__all__ = [
    "PARALLEL",
    "SEQUENTIAL",
    "StageRecord",
    "SelectionRecord",
    "RefinementResult",
    "refine_harness",
]

log = logging.getLogger(__name__)

PARALLEL = "parallel"
SEQUENTIAL = "sequential"
_DIMENSIONS: Tuple[Tuple[str, str], ...] = ((PROCEDURE, "inv"), (DOMAIN, "dom"))


@dataclass
class StageRecord:
    """One editor session: a proposal or a composition."""

    label: str
    role: str  # "procedure" | "domain" | "compose"
    attempts: int
    fell_back: bool = False
    kept_base: bool = False
    notes: str = ""
    error: str = ""
    changes: str = ""


@dataclass
class SelectionRecord:
    """How the final harness was picked."""

    id_map: dict = field(default_factory=dict)  # anonymous id -> label
    choice: Optional[str] = None
    why: str = ""
    fallback: bool = False
    trivial: bool = False


@dataclass
class RefinementResult:
    task_id: str
    mode: str
    k: int
    base: Harness
    base_runs: List[ProbeResult]
    candidates: List[ProbeResult]
    finals: List[ProbeResult]
    stages: List[StageRecord]
    ledger: HarnessBudgetLedger
    selection: SelectionRecord
    selected: Optional[ProbeResult] = None
    status: str = "ok"  # "ok" | "base-failed" | "no-delivery"

    @property
    def best_harness(self) -> Harness:
        """The selected harness, or the base harness if nothing was selected."""
        return self.selected.harness if self.selected is not None else self.base


# ------------------------------------------------------------------ evidence text
def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip() or '(empty)'}"


def _harness_json(h: Harness) -> str:
    return json.dumps(h.to_dict(), indent=2, ensure_ascii=False)


def _run_text(run: ProbeResult) -> str:
    status = "delivered" if run.delivered else ("failed" if not run.ok else "no answer")
    return (
        f"### Run {run.label} ({status})\n"
        f"Trajectory:\n{run.trajectory}\n"
        f"Deliverable:\n{(run.answer or '(none)').strip()}"
    )


@dataclass
class _PoolEntry:
    stage: StageRecord
    run: ProbeResult


def _pool_text(entries: List[_PoolEntry], relative_to: Harness) -> str:
    blocks = []
    for e in entries:
        blocks.append(
            f"### Candidate {e.stage.label} ({e.stage.role})\n"
            f"Editor notes: {e.stage.notes or '(none)'}\n"
            f"Changes vs current harness: "
            f"{relative_to.diff(e.run.harness).summary()}\n"
            f"Harness:\n{_harness_json(e.run.harness)}\n"
            f"{_run_text(e.run)}"
        )
    return "\n\n".join(blocks)


def _evidence(
    task: str,
    current: Harness,
    base_runs: List[ProbeResult],
    library: str,
    pools: Tuple[Tuple[str, List[_PoolEntry]], ...] = (),
) -> str:
    parts = [
        _section("Task", task),
        _section("Current harness", _harness_json(current)),
        _section(
            f"Base runs ({len(base_runs)} runs of the base harness)",
            "\n\n".join(_run_text(r) for r in base_runs),
        ),
    ]
    for title, entries in pools:
        if entries:
            parts.append(_section(title, _pool_text(entries, current)))
    if library.strip():
        parts.append(_section("Experience from earlier tasks", library))
    return "\n\n".join(parts)


# ------------------------------------------------------------------------ stages
def _run_stage(
    label: str,
    role: str,
    current: Harness,
    call: Callable[[str], EditProposal],
    ledger: HarnessBudgetLedger,
    stages: List[StageRecord],
    max_retries: int,
    always_on_cap: int,
) -> Harness:
    """Runs one editor session with retry-on-invalid, then falls back to ``current``."""
    feedback, error = "", ""
    for attempt in range(1, max_retries + 2):
        try:
            proposal = call(feedback)
            harness = current.copy() if proposal.keep else proposal.harness
            harness.validate(always_on_cap=always_on_cap)
        except (RoleOutputError, HarnessValidationError) as exc:
            feedback = error = str(exc)
            continue
        except Exception as exc:  # model/transport failure: retry, then fall back
            error = f"{type(exc).__name__}: {exc}"
            feedback = "the previous call failed; reply again"
            continue
        diff = current.diff(harness)
        record = StageRecord(
            label=label,
            role=role,
            attempts=attempt,
            kept_base=diff.is_empty,
            notes=proposal.notes,
            changes=diff.summary(),
        )
        stages.append(record)
        ledger.session(label, attempts=attempt)
        log.info("harness stage %s (%s): %s", label, role, record.changes)
        return harness
    stages.append(
        StageRecord(
            label=label,
            role=role,
            attempts=max_retries + 1,
            fell_back=True,
            kept_base=True,
            error=error,
            changes="no changes",
        )
    )
    ledger.session(label, attempts=max_retries + 1, fell_back=True)
    log.warning("harness stage %s (%s) fell back: %s", label, role, error)
    return current.copy()


def _probe(
    executor: HarnessExecutor,
    harness: Harness,
    task: str,
    label: str,
    ledger: HarnessBudgetLedger,
) -> ProbeResult:
    run = executor.execute(harness, task, label)
    ledger.rollout(label)
    log.info(
        "harness probe %s: %s",
        label,
        "delivered" if run.delivered else (run.error or "no answer"),
    )
    return run


# --------------------------------------------------------------------- selection
def _select(
    task: str,
    task_id: str,
    mode: str,
    k: int,
    seed: int,
    finals: List[ProbeResult],
    selector: Optional[BlindSelector],
    max_retries: int,
) -> Tuple[Optional[ProbeResult], SelectionRecord]:
    pool = [f for f in finals if f.delivered]
    if not pool:
        return None, SelectionRecord()
    if len(pool) == 1:
        only = pool[0]
        return only, SelectionRecord(id_map={"A": only.label}, choice="A", trivial=True)

    digest = hashlib.sha1(f"{task_id}|{k}|{mode}|{seed}".encode()).hexdigest()
    order = list(pool)
    random.Random(int(digest[:8], 16)).shuffle(order)
    ids = list(string.ascii_uppercase[: len(order)])
    by_id = dict(zip(ids, order))
    record = SelectionRecord(id_map={i: r.label for i, r in by_id.items()})

    evidence = "\n\n".join(
        [_section("Task", task)]
        + [_section(f"Deliverable {i}", r.answer or "") for i, r in by_id.items()]
    )
    feedback = ""
    for _ in range(max_retries + 1):
        try:
            picked = selector.select(evidence, ids, feedback=feedback)
        except RoleOutputError as exc:
            feedback = str(exc)
            continue
        except Exception as exc:
            feedback = f"the previous call failed ({type(exc).__name__}); reply again"
            continue
        record.choice, record.why = picked.choice, picked.why
        return by_id[picked.choice], record

    # Selector failed: take the first final in label order, flagged as a fallback.
    first = min(pool, key=lambda r: r.label)
    record.choice = next(i for i, r in by_id.items() if r is first)
    record.fallback = True
    return first, record


# ---------------------------------------------------------------------- driver
def refine_harness(
    task: str,
    base: Harness,
    executor: HarnessExecutor,
    editor: HarnessEditor,
    selector: Optional[BlindSelector] = None,
    *,
    k: int = 1,
    mode: str = PARALLEL,
    task_id: str = "task",
    library: str = "",
    seed: int = 0,
    max_retries: int = 1,
    always_on_cap: int = DEFAULT_ALWAYS_ON_CAP,
) -> RefinementResult:
    """Refines ``base`` on one task without labels.

    Args:
        task: the task text given to the agent (and shown to the editor/selector).
        base: the starting harness.
        executor: runs a harness on the task (e.g. ``RunnerExecutor``).
        editor: proposes and composes harness edits.
        selector: picks among delivered finals; required when ``k > 1``.
        k: task-level budget (rollouts R = 4k, editor sessions P = 3k).
        mode: ``"parallel"`` (wider) or ``"sequential"`` (deeper).
        library: optional experience text from earlier tasks, shown to the editor.
        seed: varies the anonymous ordering shown to the selector.
        max_retries: retries per editor or selector call on invalid output.
    """
    if mode not in (PARALLEL, SEQUENTIAL):
        raise ValueError(f"mode must be {PARALLEL!r} or {SEQUENTIAL!r}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if selector is None and k > 1:
        raise ValueError("a selector is required when k > 1")
    base.validate(always_on_cap=always_on_cap)

    ledger = HarnessBudgetLedger(mode=mode, task_id=task_id, k=k)
    stages: List[StageRecord] = []
    stage_kwargs = dict(
        ledger=ledger,
        stages=stages,
        max_retries=max_retries,
        always_on_cap=always_on_cap,
    )

    base_runs = [
        _probe(executor, base, task, f"base{i}", ledger) for i in range(1, k + 1)
    ]

    def result(**kw) -> RefinementResult:
        defaults = dict(
            task_id=task_id,
            mode=mode,
            k=k,
            base=base,
            base_runs=base_runs,
            candidates=[],
            finals=[],
            stages=stages,
            ledger=ledger,
            selection=SelectionRecord(),
        )
        defaults.update(kw)
        return RefinementResult(**defaults)

    if not any(r.ok for r in base_runs):
        return result(status="base-failed")

    def propose(dim: str, evidence: str) -> Callable:
        return lambda feedback: editor.propose(dim, evidence, feedback=feedback)

    def compose(evidence: str) -> Callable:
        return lambda feedback: editor.compose(evidence, feedback=feedback)

    candidates: List[ProbeResult] = []
    finals: List[ProbeResult] = []

    if mode == PARALLEL:
        evidence = _evidence(task, base, base_runs, library)
        proposed: List[Tuple[StageRecord, Harness]] = []
        for i in range(1, k + 1):
            for dim, prefix in _DIMENSIONS:
                h = _run_stage(
                    f"{prefix}{i}", dim, base, propose(dim, evidence), **stage_kwargs
                )
                proposed.append((stages[-1], h))
        pool: List[_PoolEntry] = []
        for stage, h in proposed:
            run = _probe(executor, h, task, stage.label, ledger)
            candidates.append(run)
            pool.append(_PoolEntry(stage, run))

        compose_evidence = _evidence(
            task, base, base_runs, library, pools=(("Candidates", pool),)
        )
        for i in range(1, k + 1):
            label = f"agg{i}"
            h = _run_stage(
                label, "compose", base, compose(compose_evidence), **stage_kwargs
            )
            finals.append(_probe(executor, h, task, label, ledger))
    else:
        current = base
        history: List[_PoolEntry] = []
        for t in range(1, k + 1):
            evidence = _evidence(
                task, current, base_runs, library, pools=(("Earlier rounds", history),)
            )
            round_pool: List[_PoolEntry] = []
            for dim, prefix in _DIMENSIONS:
                label = f"{prefix}{t}"
                h = _run_stage(
                    label, dim, current, propose(dim, evidence), **stage_kwargs
                )
                run = _probe(executor, h, task, label, ledger)
                candidates.append(run)
                round_pool.append(_PoolEntry(stages[-1], run))

            compose_evidence = _evidence(
                task,
                current,
                base_runs,
                library,
                pools=(("This round", round_pool), ("Earlier rounds", history)),
            )
            label = f"c{t}"
            h = _run_stage(
                label, "compose", current, compose(compose_evidence), **stage_kwargs
            )
            final = _probe(executor, h, task, label, ledger)
            finals.append(final)
            history.extend(round_pool + [_PoolEntry(stages[-1], final)])
            current = h  # a fallback keeps the current checkpoint, not the base

    ledger.assert_identity()
    selected, selection = _select(
        task, task_id, mode, k, seed, finals, selector, max_retries
    )
    return result(
        candidates=candidates,
        finals=finals,
        selection=selection,
        selected=selected,
        status="ok" if selected is not None else "no-delivery",
    )
