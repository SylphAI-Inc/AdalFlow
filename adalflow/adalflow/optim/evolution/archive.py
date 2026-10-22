"""The archive of evaluated candidates and parent selection.

Pareto selection follows GEPA (Agrawal et al., 2025): a candidate is a possible parent
if it is the best on at least one instance and not dominated across instances, and it
is sampled in proportion to how many instances it is best on. This keeps candidates
that solve different cases alive instead of collapsing onto the mean-score leader.

``RESAMPLE`` always returns the seed, so every proposal is an independent sample from
it: a best-of-N baseline at the same evaluation budget.

``MAP_ELITES`` keeps the best candidate per behavior cell (``Candidate.cell``, set from
a user descriptor) and samples parents uniformly over cells (after Mouret & Clune,
2015), so a weak but different approach still gets improved.

Every selection can be restricted to one island (``Candidate.island``); the seed
belongs to every island.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

from adalflow.optim.evolution.types import Candidate

__all__ = ["Archive", "PARETO", "BEST", "RESAMPLE", "MAP_ELITES"]

PARETO = "pareto"
BEST = "best"
RESAMPLE = "resample"
MAP_ELITES = "map_elites"
STRATEGIES = (PARETO, BEST, RESAMPLE, MAP_ELITES)
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

    @property
    def seed(self) -> Candidate:
        return self.candidates[0]

    def members(self, island: Optional[int] = None) -> List[Candidate]:
        """Candidates of one island (the seed included), or all of them."""
        if island is None:
            return list(self.candidates)
        return [
            c for i, c in enumerate(self.candidates) if i == 0 or c.island == island
        ]

    def evaluated(self, island: Optional[int] = None) -> List[Candidate]:
        return [c for c in self.members(island) if c.ok]

    def best(
        self, island: Optional[int] = None, min_samples: int = 1
    ) -> Optional[Candidate]:
        """Highest mean score; the earliest candidate wins ties.

        With ``min_samples`` > 1, only candidates evaluated that many times compete
        (falling back to all evaluated candidates when none has been).
        """
        ok = self.evaluated(island)
        confirmed = [c for c in ok if c.n_samples >= min_samples]
        pool = confirmed or ok
        return max(pool, key=lambda c: c.score) if pool else None

    @staticmethod
    def _wins(pool: List[Candidate]) -> Dict[str, int]:
        """Candidate id -> number of instances where it has the (tied) best score."""
        keys = sorted({k for c in pool for k in c.evaluation.instance_scores})
        wins: Dict[str, int] = {}
        for key in keys:
            scored = [c for c in pool if key in c.evaluation.instance_scores]
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

    def pareto_front(self, island: Optional[int] = None) -> List[Candidate]:
        """Non-dominated per-instance winners; ``[best]`` without instance scores.

        Among candidates with identical instance scores only the earliest is kept.
        """
        pool = self.evaluated(island)
        wins = self._wins(pool)
        if not wins:
            best = self.best(island)
            return [best] if best is not None else []
        winners = [c for c in pool if c.id in wins]
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

    def elites(self, island: Optional[int] = None) -> Dict[Tuple, Candidate]:
        """Behavior cell -> its best candidate (candidates without a cell skipped)."""
        cells: Dict[Tuple, Candidate] = {}
        for c in self.evaluated(island):
            if c.cell is None:
                continue
            if c.cell not in cells or c.score > cells[c.cell].score:
                cells[c.cell] = c
        return cells

    def select_parent(
        self, rng: Random, strategy: str = PARETO, island: Optional[int] = None
    ) -> Candidate:
        """A parent to improve. With nothing evaluated yet, the latest candidate."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not self.candidates:
            raise ValueError("the archive is empty")
        if strategy == RESAMPLE:
            return self.seed
        if not self.evaluated(island):
            return self.members(island)[-1]  # e.g. a seed that crashed: fix it first
        if strategy == BEST:
            return self.best(island)
        if strategy == MAP_ELITES:
            elites = self.elites(island)
            if elites:
                return elites[rng.choice(sorted(elites, key=repr))]
            return self.best(island)
        front = self.pareto_front(island)
        wins = self._wins(self.evaluated(island))
        weights = [max(wins.get(c.id, 0), 1) for c in front]
        return rng.choices(front, weights=weights, k=1)[0]

    def select_inspiration(
        self,
        rng: Random,
        parent: Candidate,
        island: Optional[int] = None,
        strategy: str = PARETO,
    ) -> Optional[Candidate]:
        """Another strong candidate to show next to the parent, if there is one.

        Under MAP-Elites it is the elite of another cell, so the proposer sees a
        different approach rather than a near copy of the parent.
        """
        if strategy == MAP_ELITES:
            pool = [c for c in self.elites(island).values() if c.id != parent.id]
            if pool:
                return rng.choice(sorted(pool, key=lambda c: c.id))
        pool = [c for c in self.pareto_front(island) if c.id != parent.id]
        if not pool:
            ranked = sorted(self.evaluated(island), key=lambda c: -c.score)
            pool = [c for c in ranked[:3] if c.id != parent.id]
        return rng.choice(pool) if pool else None
