"""Exception hierarchy for the Runner reliability layer.

All exceptions here subclass :class:`RunnerReliabilityError` so callers can catch
the entire family with a single ``except``.  Each exception carries the structured
context (step number, tool name, call id, deadline, ...) that the rest of the
reliability layer needs to make decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid cycles
    from .executor import AttemptRecord


class RunnerReliabilityError(Exception):
    """Base class for every error raised by the reliability layer."""


class StepTimeoutError(RunnerReliabilityError):
    """Raised when a single planner step exceeds its per-step timeout."""

    def __init__(self, step: int, timeout: float):
        self.step = step
        self.timeout = timeout
        super().__init__(f"Planner step {step} timed out after {timeout}s")


class ToolTimeoutError(RunnerReliabilityError):
    """Raised when a tool call exceeds its per-tool timeout."""

    def __init__(self, tool_name: str, call_id: str | None, timeout: float):
        self.tool_name = tool_name
        self.call_id = call_id
        self.timeout = timeout
        super().__init__(
            f"Tool '{tool_name}' (call_id={call_id}) timed out after {timeout}s"
        )


class RunDeadlineExceededError(RunnerReliabilityError):
    """Raised when the whole run exceeds its overall deadline."""

    def __init__(self, deadline: float, elapsed: float):
        self.deadline = deadline
        self.elapsed = elapsed
        super().__init__(
            f"Run deadline exceeded: elapsed {elapsed:.3f}s > budget {deadline:.3f}s"
        )


class RetryExhaustedError(RunnerReliabilityError):
    """Raised when retries are exhausted for an operation.

    The ``.attempts`` list keeps one :class:`AttemptRecord` per attempt so callers
    can inspect the failure history.
    """

    def __init__(
        self,
        operation: str,
        attempts: list[AttemptRecord],
        last_error: BaseException | None = None,
    ):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        msg = f"Retry exhausted for '{operation}' after {len(attempts)} attempt(s)"
        if last_error is not None:
            msg += f": {last_error}"
        super().__init__(msg)


class CheckpointError(RunnerReliabilityError):
    """Base class for checkpoint persistence / integrity errors."""


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint's integrity hash does not match its payload."""

    def __init__(self, run_id: str | None, reason: str):
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"Checkpoint '{run_id}' corrupted: {reason}")


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint's schema version is unsupported."""

    def __init__(self, found: int, supported: int):
        self.found = found
        self.supported = supported
        super().__init__(
            f"Unsupported checkpoint schema version: found {found}, supported {supported}"
        )


class ResumeMismatchError(RunnerReliabilityError):
    """Raised when resuming a run with a different agent/tools/prompt fingerprint.

    The ``field`` identifies what changed, ``expected`` and ``actual`` carry the
    fingerprints for a useful error message.
    """

    def __init__(
        self,
        run_id: str | None,
        field: str,
        expected: Any,
        actual: Any,
    ):
        self.run_id = run_id
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Resume mismatch for run '{run_id}' on field '{field}': "
            f"expected {expected!r}, got {actual!r}"
        )


class ToolReplayRefusedError(RunnerReliabilityError):
    """Raised when a non-idempotent tool was started but never completed.

    On resume the ledger has a ``started`` entry for a non-idempotent tool, which
    means the process likely died mid-execution.  Re-running it could double-charge
    a credit card, send a second email, etc.  The caller must explicitly opt in to
    ``rerun`` or ``skip`` via :meth:`Runner.resume` / :meth:`Runner.aresume`.
    """

    def __init__(self, tool_name: str, call_id: str | None):
        self.tool_name = tool_name
        self.call_id = call_id
        super().__init__(
            f"Refusing to replay non-idempotent tool '{tool_name}' "
            f"(call_id={call_id}) that was started but never completed"
        )
