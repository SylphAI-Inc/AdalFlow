"""The evolution loop: select a parent, propose, evaluate, archive, repeat."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Optional, Union

from adalflow.optim.evolution.archive import PARETO, Archive
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
) -> EvolutionResult:
    """Evolves ``seed`` with ``proposer`` under a budget of evaluator calls.

    Args:
        seed: the starting artifact (it is evaluated first and counts toward budget).
        evaluate: ``text -> score | dict | Evaluation`` (e.g. ``PythonEvaluator``).
            Exceptions are recorded as failed evaluations.
        proposer: e.g. ``LLMProposer``.
        max_evaluations: total evaluator calls, seed included.
        selection: ``"pareto"`` (per-instance, GEPA-style) or ``"best"`` (greedy).
        stop_score: stop once a candidate reaches this score.
        max_stalls: stop after this many proposals in a row that failed or repeated
            an earlier candidate.
        log_path: append one JSON line per event (candidate, failure, duplicate).
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be >= 1")
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

    def add(candidate: Candidate) -> None:
        candidate.evaluation = _evaluate(evaluate, candidate.text)
        archive.add(candidate)
        record({"event": "candidate", **candidate.to_dict()})
        ev = candidate.evaluation
        log.info(
            "evolve %s (parent %s): %s",
            candidate.id,
            candidate.parent_id,
            f"score {ev.score:.4g}" if ev.ok else f"failed: {ev.error}",
        )

    seed_candidate = Candidate(id="c0", text=seed, origin="seed")
    add(seed_candidate)
    result = EvolutionResult(
        best=None, seed=seed_candidate, archive=archive, evaluations=1, events=events
    )

    def reached() -> bool:
        best = archive.best()
        return stop_score is not None and best is not None and best.score >= stop_score

    stalls = 0
    while result.evaluations < max_evaluations:
        if reached():
            result.stopped = "stop_score"
            break
        if stalls >= max_stalls:
            result.stopped = "stalled"
            break
        parent = archive.select_parent(rng, selection)
        inspiration = archive.select_inspiration(rng, parent)
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
        add(
            Candidate(
                id=f"c{len(archive)}",
                text=proposal.text,
                generation=parent.generation + 1,
                parent_id=parent.id,
                inspiration_id=inspiration.id if inspiration is not None else None,
                origin="llm",
                rationale=proposal.rationale,
            )
        )
        result.evaluations += 1
    else:
        if reached():
            result.stopped = "stop_score"

    result.best = archive.best()
    return result
