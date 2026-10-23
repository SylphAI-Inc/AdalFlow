"""Candidates and evaluations shared by the evolution loop, proposers and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import statistics
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Evaluation", "Candidate", "combine", "to_evaluation"]


@dataclass
class Evaluation:
    """How one candidate did.

    ``score`` is ``None`` when the candidate could not be evaluated (``error`` says
    why). ``instance_scores`` (e.g. one entry per test case) drive Pareto parent
    selection; ``feedback`` is what the proposer reads to diagnose failures.
    """

    score: Optional[float] = None
    instance_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None


_KNOWN_KEYS = ("score", "instance_scores", "feedback", "error")


def to_evaluation(value: Any) -> Evaluation:
    """Normalizes an evaluator's return value.

    Accepts an ``Evaluation``, a number (the score), or a dict with ``score``,
    ``instance_scores``, ``feedback`` and ``error``; other dict keys go to
    ``metrics``. A dict without ``score`` but with ``instance_scores`` scores as
    their mean.
    """
    if isinstance(value, Evaluation):
        return value
    if isinstance(value, bool):
        raise TypeError("evaluator returned a bool; return a float score instead")
    if isinstance(value, (int, float)):
        return Evaluation(score=float(value))
    if not isinstance(value, dict):
        raise TypeError(
            f"evaluator must return a number, dict or Evaluation, got {type(value)}"
        )
    instances = {
        str(k): float(v) for k, v in (value.get("instance_scores") or {}).items()
    }
    score = value.get("score")
    if score is None and instances:
        score = sum(instances.values()) / len(instances)
    error = value.get("error")
    return Evaluation(
        score=None if score is None or error else float(score),
        instance_scores=instances,
        feedback=str(value.get("feedback") or ""),
        error=None if error is None else str(error),
        metrics={k: v for k, v in value.items() if k not in _KNOWN_KEYS},
    )


def combine(samples: List[Evaluation]) -> Evaluation:
    """Averages repeated evaluations of one candidate (for noisy evaluators).

    Failed samples are left out of the means; the feedback and other metrics are the
    latest sample's. ``metrics`` gets ``samples`` (scored samples) and, with two or
    more, ``score_sd``.
    """
    if not samples:
        raise ValueError("no samples to combine")
    if len(samples) == 1:
        return samples[0]
    ok = [s for s in samples if s.ok]
    latest = samples[-1]
    if not ok:
        return latest
    keys = sorted({k for s in ok for k in s.instance_scores})
    instances = {}
    for key in keys:
        values = [s.instance_scores[key] for s in ok if key in s.instance_scores]
        instances[key] = statistics.mean(values)
    scores = [s.score for s in ok]
    metrics = dict(latest.metrics)
    metrics["samples"] = len(ok)
    if len(scores) > 1:
        metrics["score_sd"] = statistics.stdev(scores)
    return Evaluation(
        score=statistics.mean(scores),
        instance_scores=instances,
        feedback=latest.feedback if latest.ok else ok[-1].feedback,
        metrics=metrics,
    )


@dataclass
class Candidate:
    """One artifact in the archive.

    ``rationale`` is written by the proposer before the candidate is evaluated, so a
    log of rationales never carries outcome annotations. ``evaluation`` is the mean
    of ``samples`` when a candidate was evaluated more than once. ``island`` and
    ``cell`` place it for island and MAP-Elites selection.
    """

    id: str
    text: str
    generation: int = 0
    parent_id: Optional[str] = None
    inspiration_id: Optional[str] = None
    origin: str = "seed"
    rationale: str = ""
    evaluation: Optional[Evaluation] = None
    samples: List[Evaluation] = field(default_factory=list)
    island: int = 0
    cell: Optional[Tuple[Any, ...]] = None

    def add_sample(self, evaluation: Evaluation) -> None:
        self.samples.append(evaluation)
        self.evaluation = combine(self.samples)

    @property
    def n_samples(self) -> int:
        if self.samples:
            return len(self.samples)
        return 1 if self.evaluation is not None else 0

    @property
    def ok(self) -> bool:
        return self.evaluation is not None and self.evaluation.ok

    @property
    def score(self) -> Optional[float]:
        return self.evaluation.score if self.evaluation is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
