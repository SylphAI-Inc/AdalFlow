"""The archive of evaluated candidates and parent selection.

Pareto selection follows GEPA (Agrawal et al., 2025): a candidate is a possible parent
if it is the best on at least one instance and not dominated across instances, and it
is sampled in proportion to how many instances it is best on. This keeps candidates
that solve different cases alive instead of collapsing onto the mean-score leader.

``RESAMPLE`` always returns the seed, so every proposal is an independent sample from
it: a best-of-N baseline at the same evaluation budget.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional

from adalflow.optim.evolution.types import Candidate

__all__ = ["Archive", "PARETO", "BEST", "RESAMPLE"]

PARETO = "pareto"
BEST = "best"
RESAMPLE = "resample"
STRATEGIES = (PARETO, BEST, RESAMPLE)
_EPS = 1e-12


class Archive:
    """Every candidate tried, in order, including failed ones."""

    def __init__(self) -> None:
        self.candidates: List[Candidate] = []
        self._texts = set()

    def __len__(self) -> int:
        return len(self.candidates)

    def add(self, candidate: Candidate) -> None:
        self.candidates.append(candidate)
        self._texts.add(candidate.text)

    def has_text(self, text: str) -> bool:
        return text in self._texts

    def evaluated(self) -> List[Candidate]:
        return [c for c in self.candidates if c.ok]

    def best(self) -> Optional[Candidate]:
        """Highest score; the earliest candidate wins ties."""
        ok = self.evaluated()
        return max(ok, key=lambda c: c.score) if ok else None

    def _wins(self) -> Dict[str, int]:
        """Candidate id -> number of instances where it has the (tied) best score."""
        ok = self.evaluated()
        keys = sorted({k for c in ok for k in c.evaluation.instance_scores})
        wins: Dict[str, int] = {}
        for key in keys:
            scored = [c for c in ok if key in c.evaluation.instance_scores]
            top = max(c.evaluation.instance_scores[key] for c in scored)
            for c in scored:
                if c.evaluation.instance_scores[key] >= top - _EPS:
                    wins[c.id] = wins.get(c.id, 0) + 1
        return wins

    @staticmethod
    def _dominates(a: Candidate, b: Candidate) -> bool:
        keys = set(a.evaluation.instance_scores) | set(b.evaluation.instance_scores)
        ra = [a.evaluation.instance_scores.get(k, float("-inf")) for k in keys]
        rb = [b.evaluation.instance_scores.get(k, float("-inf")) for k in keys]
        return all(x >= y for x, y in zip(ra, rb)) and any(
            x > y + _EPS for x, y in zip(ra, rb)
        )

    def pareto_front(self) -> List[Candidate]:
        """Non-dominated per-instance winners; ``[best]`` without instance scores.

        Among candidates with identical instance scores only the earliest is kept.
        """
        wins = self._wins()
        if not wins:
            best = self.best()
            return [best] if best is not None else []
        winners = [c for c in self.evaluated() if c.id in wins]
        front: List[Candidate] = []
        for c in winners:
            if any(self._dominates(o, c) for o in winners if o is not c):
                continue
            if any(
                o.evaluation.instance_scores == c.evaluation.instance_scores
                for o in front
            ):
                continue
            front.append(c)
        return front

    def select_parent(self, rng: Random, strategy: str = PARETO) -> Candidate:
        """A parent to improve. With nothing evaluated yet, the latest candidate."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not self.candidates:
            raise ValueError("the archive is empty")
        if strategy == RESAMPLE:
            return self.candidates[0]
        if not self.evaluated():
            return self.candidates[-1]  # e.g. a seed that crashed: fix it first
        if strategy == BEST:
            return self.best()
        front = self.pareto_front()
        wins = self._wins()
        weights = [max(wins.get(c.id, 0), 1) for c in front]
        return rng.choices(front, weights=weights, k=1)[0]

    def select_inspiration(self, rng: Random, parent: Candidate) -> Optional[Candidate]:
        """Another strong candidate to show next to the parent, if there is one."""
        pool = [c for c in self.pareto_front() if c.id != parent.id]
        if not pool:
            ranked = sorted(self.evaluated(), key=lambda c: -c.score)
            pool = [c for c in ranked[:3] if c.id != parent.id]
        return rng.choice(pool) if pool else None
