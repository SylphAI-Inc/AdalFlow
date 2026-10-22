"""The evolution loop: select a parent, propose, evaluate, archive, repeat."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from adalflow.optim.evolution.archive import (
    MAP_ELITES,
    PARETO,
    RESAMPLE,
    STRATEGIES,
    Archive,
)
from adalflow.optim.evolution.proposer import Proposer
from adalflow.optim.evolution.types import Candidate, Evaluation, to_evaluation

__all__ = ["EvolutionResult", "evolve"]

log = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    best: Optional[Candidate]
    seed: Candidate
    archive: Archive
    evaluations: int
    proposal_failures: int = 0
    duplicates: int = 0
    stopped: str = "budget"  # "budget" | "stop_score" | "stalled"
    events: List[Dict[str, Any]] = field(default_factory=list)
    # "seed" / "best" -> Evaluation on the held-out evaluator, when one was given.
    holdout: Dict[str, Evaluation] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return (
            self.best is not None
            and self.best.id != self.seed.id
            and (self.seed.score is None or self.best.score > self.seed.score)
        )

    def pareto_front(self) -> List[Candidate]:
        return self.archive.pareto_front()


def _evaluate(evaluate: Callable[[str], Any], text: str) -> Evaluation:
    try:
        return to_evaluation(evaluate(text))
    except Exception as exc:  # an evaluator failure is recorded, not raised
        return Evaluation(error=f"{type(exc).__name__}: {exc}")


def evolve(
    seed: str,
    evaluate: Callable[[str], Any],
    proposer: Proposer,
    *,
    max_evaluations: int = 20,
    selection: str = PARETO,
    random_seed: int = 0,
    stop_score: Optional[float] = None,
    max_stalls: int = 5,
    log_path: Union[str, Path, None] = None,
    holdout: Optional[Callable[[str], Any]] = None,
    confirm: int = 0,
    islands: int = 1,
    migrate_every: int = 4,
    descriptor: Optional[Callable[[Candidate], Tuple[Any, ...]]] = None,
) -> EvolutionResult:
    """Evolves ``seed`` with ``proposer`` under a budget of evaluator calls.

    Args:
        seed: the starting artifact (it is evaluated first and counts toward budget).
        evaluate: ``text -> score | dict | Evaluation`` (e.g. ``PythonEvaluator``).
            Exceptions are recorded as failed evaluations.
        proposer: e.g. ``LLMProposer``.
        max_evaluations: total evaluator calls, seed and re-evaluations included.
        selection: ``"pareto"`` (per-instance, GEPA-style), ``"best"`` (greedy),
            ``"map_elites"`` (uniform over behavior cells; needs ``descriptor``) or
            ``"resample"`` (every proposal from the seed, no inspiration: the
            best-of-N baseline to compare a search strategy against).
        stop_score: stop once the best candidate reaches this (mean) score.
        max_stalls: stop after this many proposals in a row that failed or repeated
            an earlier candidate.
        log_path: append one JSON line per event (candidate, sample, failure,
            duplicate, migration, holdout).
        holdout: an evaluator on instances the search never sees. The seed and the
            final best are scored on it after the search (not counted in
            ``max_evaluations``), so overfitting to ``evaluate`` shows up as a gap.
        confirm: for noisy evaluators. When a new candidate's first score beats the
            best so far, both are re-evaluated until each has ``1 + confirm``
            samples, and the means decide. Only candidates with that many samples
            can be the best, so one lucky score does not win the run. A proposal
            starts only if the budget left can finish its comparison.
        islands: number of sub-populations. Proposals rotate over islands, parents
            and inspirations come from the current island, and every
            ``migrate_every`` proposals the inspiration is the best of the next
            island instead (0 never migrates).
        descriptor: ``candidate -> tuple`` behavior cell, called after each
            evaluation (it can read ``candidate.evaluation.metrics``). Required for
            ``"map_elites"``.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be >= 1")
    if selection not in STRATEGIES:
        raise ValueError(f"selection must be one of {STRATEGIES}")
    if selection == MAP_ELITES and descriptor is None:
        raise ValueError("selection='map_elites' needs a descriptor")
    if confirm < 0 or islands < 1 or migrate_every < 0:
        raise ValueError("confirm and migrate_every must be >= 0, islands >= 1")
    need = 1 + confirm
    rng = Random(random_seed)
    archive = Archive()
    events: List[Dict[str, Any]] = []
    sink = Path(log_path) if log_path is not None else None
    if sink is not None:
        sink.parent.mkdir(parents=True, exist_ok=True)

    def record(event: Dict[str, Any]) -> None:
        events.append(event)
        if sink is not None:
            with sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")

    def sample(candidate: Candidate) -> None:
        candidate.add_sample(_evaluate(evaluate, candidate.text))
        result.evaluations += 1
        if descriptor is not None and candidate.ok:
            try:
                candidate.cell = tuple(descriptor(candidate))
            except Exception as exc:  # a bad descriptor leaves the cell unset
                log.warning("descriptor failed on %s: %s", candidate.id, exc)
                candidate.cell = None

    def add(candidate: Candidate) -> None:
        sample(candidate)
        archive.add(candidate)
        record({"event": "candidate", **_summary(candidate)})
        ev = candidate.evaluation
        log.info(
            "evolve %s (parent %s): %s",
            candidate.id,
            candidate.parent_id,
            f"score {ev.score:.4g}" if ev.ok else f"failed: {ev.error}",
        )

    def resample(candidate: Candidate) -> None:
        sample(candidate)
        record(
            {
                "event": "sample",
                "id": candidate.id,
                "n": candidate.n_samples,
                "last": candidate.samples[-1].score,
                "mean": candidate.score,
            }
        )

    def race(challenger: Candidate) -> None:
        """Re-evaluates a challenger and the incumbent until both have ``need``."""
        others = [c for c in archive.evaluated() if c is not challenger]
        confirmed = [c for c in others if c.n_samples >= need]
        pool = confirmed or others
        incumbent = max(pool, key=lambda c: c.score) if pool else None
        if not challenger.ok or incumbent is None:
            return
        if challenger.score <= incumbent.score:
            return
        while result.evaluations < max_evaluations and (
            challenger.n_samples < need or incumbent.n_samples < need
        ):
            target = (
                challenger if challenger.n_samples <= incumbent.n_samples else incumbent
            )
            resample(target)

    seed_candidate = Candidate(id="c0", text=seed, origin="seed")
    result = EvolutionResult(
        best=None, seed=seed_candidate, archive=archive, evaluations=0, events=events
    )
    add(seed_candidate)

    def reached() -> bool:
        best = archive.best(min_samples=need)
        return (
            stop_score is not None
            and best is not None
            and best.n_samples >= need
            and best.score >= stop_score
        )

    stalls = attempts = 0
    while result.evaluations < max_evaluations:
        if reached():
            result.stopped = "stop_score"
            break
        if stalls >= max_stalls:
            result.stopped = "stalled"
            break
        if confirm:
            # Start a proposal only if its race can finish within the budget.
            incumbent = archive.best(min_samples=need)
            reserve = need + max(0, need - incumbent.n_samples) if incumbent else need
            if max_evaluations - result.evaluations < reserve:
                break
        island = attempts % islands
        attempts += 1
        scope = island if islands > 1 else None
        parent = archive.select_parent(rng, selection, scope)
        inspiration = None
        if selection != RESAMPLE:
            migrant = None
            if islands > 1 and migrate_every and attempts % migrate_every == 0:
                migrant = archive.best((island + 1) % islands, min_samples=need)
            if migrant is not None and migrant.id != parent.id:
                inspiration = migrant
                record(
                    {
                        "event": "migration",
                        "id": migrant.id,
                        "from": (island + 1) % islands,
                        "to": island,
                    }
                )
            else:
                inspiration = archive.select_inspiration(rng, parent, scope, selection)
        try:
            proposal = proposer.propose(parent, inspiration)
        except Exception as exc:  # model or parse failure: count it, try again
            stalls += 1
            result.proposal_failures += 1
            record(
                {
                    "event": "proposal_failure",
                    "parent": parent.id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if archive.has_text(proposal.text):
            stalls += 1
            result.duplicates += 1
            record({"event": "duplicate", "parent": parent.id})
            continue
        stalls = 0
        child = Candidate(
            id=f"c{len(archive)}",
            text=proposal.text,
            generation=parent.generation + 1,
            parent_id=parent.id,
            inspiration_id=inspiration.id if inspiration is not None else None,
            origin="llm",
            rationale=proposal.rationale,
            island=island,
        )
        add(child)
        if confirm:
            race(child)
    else:
        if reached():
            result.stopped = "stop_score"

    result.best = archive.best(min_samples=need)
    if holdout is not None:
        targets = {"seed": seed_candidate}
        if result.best is not None:
            targets["best"] = result.best
        for name, candidate in targets.items():
            result.holdout[name] = _evaluate(holdout, candidate.text)
            ev = result.holdout[name]
            record(
                {
                    "event": "holdout",
                    "role": name,
                    "id": candidate.id,
                    "score": ev.score,
                    "error": ev.error,
                }
            )
    return result


def _summary(candidate: Candidate) -> Dict[str, Any]:
    """A candidate for the event log (its samples are logged as they arrive)."""
    data = candidate.to_dict()
    data.pop("samples", None)
    return data
