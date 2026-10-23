"""Candidates and evaluations shared by the evolution loop, proposers and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["Evaluation", "Candidate", "to_evaluation"]


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


@dataclass
class Candidate:
    """One artifact in the archive.

    ``rationale`` is written by the proposer before the candidate is evaluated, so a
    log of rationales never carries outcome annotations.
    """

    id: str
    text: str
    generation: int = 0
    parent_id: Optional[str] = None
    inspiration_id: Optional[str] = None
    origin: str = "seed"
    rationale: str = ""
    evaluation: Optional[Evaluation] = None

    @property
    def ok(self) -> bool:
        return self.evaluation is not None and self.evaluation.ok

    @property
    def score(self) -> Optional[float]:
        return self.evaluation.score if self.evaluation is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
