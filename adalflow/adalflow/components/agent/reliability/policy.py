"""Retry, timeout and backoff policies for the Runner reliability layer.

All policies are frozen dataclasses so they are hashable and safe to share
across runs.  Defaults are conservative: ``RetryPolicy.none()`` disables retry
entirely, matching the pre-reliability Runner behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from .classification import ErrorCategory, ErrorClassifier, default_classifier


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackoffPolicy:
    """Exponential backoff with optional jitter.

    ``delay(attempt, rng, retry_after)`` returns the seconds to wait before the
    *next* attempt.  ``attempt`` is 0-indexed (attempt 0 is the first try, the
    delay before attempt 1 is computed from it).

    ``retry_after`` (from a 429 ``Retry-After`` header) overrides the computed
    delay but is still capped by ``max_delay``.
    """

    initial: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: Literal["none", "full", "equal"] = "full"

    def delay(
        self,
        attempt: int,
        rng: random.Random | None = None,
        retry_after: float | None = None,
    ) -> float:
        if retry_after is not None:
            return max(0.0, min(float(retry_after), self.max_delay))
        # base exponential: initial * multiplier^attempt
        base = self.initial * (self.multiplier ** max(0, attempt))
        base = min(base, self.max_delay)
        if self.jitter == "none" or rng is None:
            return base
        if self.jitter == "full":
            return rng.uniform(0.0, base)
        # "equal" jitter: half fixed + half random.
        half = base / 2.0
        return half + rng.uniform(0.0, half)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    """When and how many times to retry an operation.

    ``retry_on`` restricts retries to a set of categories.  ``retry_parse_errors``
    additionally allows retrying PARSE errors (off by default - parse errors are
    usually deterministic).  ``max_total_delay`` caps the cumulative time spent
    in backoff sleeps across all retries of a single operation.
    """

    max_attempts: int = 3
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)
    retry_on: frozenset[ErrorCategory] = frozenset(
        {ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMITED, ErrorCategory.TIMEOUT}
    )
    retry_parse_errors: bool = False
    max_total_delay: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.backoff.initial < 0:
            raise ValueError("BackoffPolicy.initial must be >= 0")
        if self.backoff.multiplier < 1.0:
            raise ValueError("BackoffPolicy.multiplier must be >= 1.0")
        if self.backoff.max_delay < 0:
            raise ValueError("BackoffPolicy.max_delay must be >= 0")
        if self.max_total_delay is not None and self.max_total_delay < 0:
            raise ValueError("RetryPolicy.max_total_delay must be >= 0")

    @classmethod
    def none(cls) -> RetryPolicy:
        """A policy that never retries (matches pre-reliability behaviour)."""
        return cls(max_attempts=1, backoff=BackoffPolicy(initial=0.0, jitter="none"))

    def should_retry(
        self,
        classified,
        attempt: int,
        elapsed_delay: float = 0.0,
    ) -> bool:
        """Return True if we should retry after ``attempt`` (0-indexed)."""
        if attempt + 1 >= self.max_attempts:
            return False
        if self.max_total_delay is not None and elapsed_delay >= self.max_total_delay:
            return False
        category = classified.category
        if category in self.retry_on:
            return True
        if self.retry_parse_errors and category == ErrorCategory.PARSE:
            return True
        return False


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeoutPolicy:
    """Per-operation and whole-run timeouts (seconds).

    Any field left ``None`` means "no limit for this scope".  The effective
    per-call timeout is ``min(op_timeout, remaining_run_budget)``.
    """

    planner_timeout: float | None = None
    tool_timeout: float | None = None
    run_timeout: float | None = None
    cancel_grace: float = 1.0

    def remaining(self, started_at: float, now: float) -> float | None:
        """Return the remaining run budget in seconds, or None if unlimited."""
        if self.run_timeout is None:
            return None
        elapsed = now - started_at
        return max(0.0, self.run_timeout - elapsed)

    def effective(
        self, op_timeout: float | None, started_at: float, now: float
    ) -> float | None:
        """Combine an op-level timeout with the remaining run budget."""
        remaining = self.remaining(started_at, now)
        candidates = [c for c in (op_timeout, remaining) if c is not None]
        if not candidates:
            return None
        return min(candidates)


# ---------------------------------------------------------------------------
# Tool policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolPolicy:
    """Per-tool overrides.

    Tools are non-idempotent by default: a tool that may have started executing
    is never auto-retried.  Only TIMEOUT failures on idempotent tools, or
    failures that happened before the tool body was invoked, are retryable.
    """

    timeout: float | None = None
    retry: RetryPolicy | None = None
    idempotent: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityConfig:
    """Top-level reliability configuration for the Runner.

    By default retry is disabled (``RetryPolicy.none()``) and no timeouts are
    set, so a Runner constructed with a default :class:`ReliabilityConfig`
    behaves identically to a Runner constructed without one.
    """

    retry: RetryPolicy = field(default_factory=RetryPolicy.none)
    timeouts: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    tools: dict[str, ToolPolicy] = field(default_factory=dict)
    classifier: ErrorClassifier = field(default_factory=lambda: default_classifier)
    seed: int | None = None

    def tool_policy(self, name: str) -> ToolPolicy:
        """Return the effective :class:`ToolPolicy` for ``name``.

        Tool-specific entries override the defaults.  Tool-specific ``None``
        fields fall back to the global :class:`TimeoutPolicy` / :class:`RetryPolicy`.
        """
        specific = self.tools.get(name)
        if specific is None:
            return ToolPolicy(
                timeout=self.timeouts.tool_timeout,
                retry=self.retry,
                idempotent=False,
            )
        timeout = specific.timeout
        if timeout is None:
            timeout = self.timeouts.tool_timeout
        retry = specific.retry
        if retry is None:
            retry = self.retry
        return ToolPolicy(timeout=timeout, retry=retry, idempotent=specific.idempotent)

    # ------------------------------------------------------------------ (de)ser
    def to_dict(self) -> dict[str, Any]:
        return {
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "retry_parse_errors": self.retry_parse_errors_flag(),
                "max_total_delay": self.retry.max_total_delay,
                "backoff": {
                    "initial": self.retry.backoff.initial,
                    "multiplier": self.retry.backoff.multiplier,
                    "max_delay": self.retry.backoff.max_delay,
                    "jitter": self.retry.backoff.jitter,
                },
                "retry_on": sorted(c.value for c in self.retry.retry_on),
            },
            "timeouts": {
                "planner_timeout": self.timeouts.planner_timeout,
                "tool_timeout": self.timeouts.tool_timeout,
                "run_timeout": self.timeouts.run_timeout,
                "cancel_grace": self.timeouts.cancel_grace,
            },
            "tools": {
                name: {
                    "timeout": p.timeout,
                    "idempotent": p.idempotent,
                }
                for name, p in self.tools.items()
            },
            "seed": self.seed,
        }

    def retry_parse_errors_flag(self) -> bool:
        return self.retry.retry_parse_errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReliabilityConfig:
        retry_data = data.get("retry") or {}
        backoff_data = retry_data.get("backoff") or {}
        retry_on = (
            frozenset(ErrorCategory(v) for v in retry_data.get("retry_on", []))
            or RetryPolicy().retry_on
        )
        retry = RetryPolicy(
            max_attempts=retry_data.get("max_attempts", 3),
            backoff=BackoffPolicy(
                initial=backoff_data.get("initial", 0.5),
                multiplier=backoff_data.get("multiplier", 2.0),
                max_delay=backoff_data.get("max_delay", 30.0),
                jitter=backoff_data.get("jitter", "full"),
            ),
            retry_on=retry_on,
            retry_parse_errors=retry_data.get("retry_parse_errors", False),
            max_total_delay=retry_data.get("max_total_delay"),
        )
        timeouts_data = data.get("timeouts") or {}
        timeouts = TimeoutPolicy(
            planner_timeout=timeouts_data.get("planner_timeout"),
            tool_timeout=timeouts_data.get("tool_timeout"),
            run_timeout=timeouts_data.get("run_timeout"),
            cancel_grace=timeouts_data.get("cancel_grace", 1.0),
        )
        tools_data = data.get("tools") or {}
        tools = {
            name: ToolPolicy(
                timeout=t.get("timeout"),
                idempotent=t.get("idempotent", False),
            )
            for name, t in tools_data.items()
        }
        return cls(
            retry=retry,
            timeouts=timeouts,
            tools=tools,
            seed=data.get("seed"),
        )
