"""Budget accounting for one harness-refinement step.

With task-level budget ``k``, both parallel and sequential refinement spend:

- R = 4k rollouts: k base rollouts, 2k candidate probes, k final probes;
- P = 3k editor sessions: k procedure edits, k domain edits, k compositions.

The ledger records each as it happens and can check that identity, so a stage that
silently skipped (or ran twice under one label) is caught instead of mis-reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

__all__ = [
    "expected_rollouts",
    "expected_sessions",
    "BudgetError",
    "BudgetEntry",
    "HarnessBudgetLedger",
]


def expected_rollouts(k: int) -> int:
    return 4 * int(k)


def expected_sessions(k: int) -> int:
    return 3 * int(k)


class BudgetError(AssertionError):
    """The recorded entries do not match the R = 4k, P = 3k identity."""


@dataclass
class BudgetEntry:
    kind: str  # "rollout" | "session"
    label: str
    attempts: int = 1
    fell_back: bool = False
    tokens: int = 0


@dataclass
class HarnessBudgetLedger:
    """Everything one refinement mode spent on one task."""

    mode: str
    task_id: str
    k: int
    entries: List[BudgetEntry] = field(default_factory=list)

    def rollout(self, label: str, tokens: int = 0) -> None:
        self.entries.append(BudgetEntry("rollout", label, tokens=tokens))

    def session(
        self, label: str, attempts: int = 1, fell_back: bool = False, tokens: int = 0
    ) -> None:
        self.entries.append(
            BudgetEntry(
                "session", label, attempts=attempts, fell_back=fell_back, tokens=tokens
            )
        )

    def labels(self, kind: str) -> List[str]:
        return [e.label for e in self.entries if e.kind == kind]

    @property
    def rollouts(self) -> int:
        return len(self.labels("rollout"))

    @property
    def sessions(self) -> int:
        return len(self.labels("session"))

    def assert_identity(self) -> None:
        """Raises ``BudgetError`` unless R = 4k and P = 3k with no repeated label.

        Duplicates are checked first: a repeated label can make the counts look right
        while one stage overwrote another.
        """
        problems = []
        for kind in ("rollout", "session"):
            dupes = sorted(n for n, c in Counter(self.labels(kind)).items() if c > 1)
            if dupes:
                problems.append(f"duplicate {kind} labels: {dupes}")
        if self.rollouts != expected_rollouts(self.k):
            problems.append(f"R={self.rollouts}, expected {expected_rollouts(self.k)}")
        if self.sessions != expected_sessions(self.k):
            problems.append(f"P={self.sessions}, expected {expected_sessions(self.k)}")
        if problems:
            raise BudgetError(
                f"{self.mode}-{self.k} {self.task_id}: " + "; ".join(problems)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "R": self.rollouts,
            "expected_R": expected_rollouts(self.k),
            "P": self.sessions,
            "expected_P": expected_sessions(self.k),
            "session_attempts": sum(
                e.attempts for e in self.entries if e.kind == "session"
            ),
            "fell_back": [e.label for e in self.entries if e.fell_back],
            "tokens": sum(e.tokens for e in self.entries),
            "entries": [asdict(e) for e in self.entries],
        }
