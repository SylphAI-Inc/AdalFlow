"""The reliability execution engine.

:class:`ReliableExecutor` wraps planner and tool calls with timeout, retry and
backoff.  It is fully injectable (clock, sleep, RNG) so tests run instantly and
deterministically.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import (
    Literal,
)

from adalflow.core.types import (
    Function,
    FunctionOutput,
    GeneratorOutput,
    ToolOutput,
)

from .classification import ClassifiedError, ErrorCategory
from .errors import (
    RunDeadlineExceededError,
    ToolTimeoutError,
)
from .policy import ReliabilityConfig, ToolPolicy

log = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    """One attempt of a retried operation."""

    attempt: int
    classified: ClassifiedError
    delay: float | None = None
    timestamp: float = 0.0


@dataclass
class ReliabilityEvent:
    """Emitted by the executor for observability / streaming."""

    kind: Literal[
        "retry_scheduled",
        "attempt_failed",
        "timeout",
        "retry_exhausted",
        "deadline_exceeded",
        "checkpoint_saved",
        "resumed",
        "tool_replayed_from_checkpoint",
    ]
    step: int | None = None
    operation: str | None = None
    attempt: int = 0
    delay: float | None = None
    category: str | None = None
    message: str | None = None
    timestamp: float = 0.0


@dataclass
class ExecutorStats:
    """Per-operation counters."""

    attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    deadline_exceeded: int = 0


Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
SyncSleep = Callable[[float], None]
EventCallback = Callable[[ReliabilityEvent], None]


class ReliableExecutor:
    """Apply timeout + retry policies to planner and tool calls.

    The executor is stateful: :meth:`start_run` records the run start time used
    for the run-deadline check.  ``clock`` and ``sleep`` are injectable so tests
    can run without real time.
    """

    def __init__(
        self,
        config: ReliabilityConfig,
        clock: Clock = time.monotonic,
        sleep: Sleep | None = None,
        sync_sleep: SyncSleep | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._sleep: Sleep = sleep or asyncio.sleep
        self._sync_sleep: SyncSleep = sync_sleep or time.sleep
        self._on_event = on_event
        self._run_started: float | None = None
        self._rng = random.Random(config.seed)
        self._stats: dict[str, ExecutorStats] = {}
        self._thread_pool: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def config(self) -> ReliabilityConfig:
        return self._config

    def start_run(self) -> float:
        self._run_started = self._clock()
        return self._run_started

    def _now(self) -> float:
        return self._clock()

    def _remaining_run(self) -> float | None:
        if self._run_started is None:
            return None
        return self._config.timeouts.remaining(self._run_started, self._now())

    def _emit(self, event: ReliabilityEvent) -> None:
        if event.timestamp == 0.0:
            event.timestamp = self._now()
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # pragma: no cover
                log.exception("reliability event callback raised")

    def _stat(self, operation: str) -> ExecutorStats:
        return self._stats.setdefault(operation, ExecutorStats())

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            op: {
                "attempts": s.attempts,
                "retries": s.retries,
                "timeouts": s.timeouts,
                "deadline_exceeded": s.deadline_exceeded,
            }
            for op, s in self._stats.items()
        }

    def close(self) -> None:
        """Shut down the internal thread pool used for sync timeouts."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
            self._thread_pool = None

    # ------------------------------------------------------------------ planner
    async def run_planner(
        self,
        fn: Callable[[], Awaitable[GeneratorOutput]],
        step: int,
    ) -> GeneratorOutput:
        """Run the planner with timeout + retry.

        Never raises for retryable failures; returns the last GeneratorOutput
        instead so the Runner keeps its existing error-step semantics.
        """
        operation = f"planner:step{step}"
        retry = self._config.retry
        stats = self._stat(operation)
        attempts: list[AttemptRecord] = []
        elapsed_delay = 0.0
        last_output: GeneratorOutput | None = None
        attempt = 0
        while True:
            stats.attempts += 1
            remaining = self._remaining_run()
            if remaining is not None and remaining <= 0:
                stats.deadline_exceeded += 1
                self._emit(
                    ReliabilityEvent(
                        kind="deadline_exceeded",
                        step=step,
                        operation=operation,
                        message="run deadline exceeded before attempt",
                    )
                )
                raise RunDeadlineExceededError(
                    deadline=self._config.timeouts.run_timeout or 0.0,
                    elapsed=self._now() - (self._run_started or 0.0),
                )
            try:
                output = await self._call_async_with_timeout(
                    fn,
                    self._config.timeouts.effective(
                        self._config.timeouts.planner_timeout,
                        self._run_started or self._now(),
                        self._now(),
                    ),
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as e:
                stats.timeouts += 1
                classified = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    retryable=True,
                    message=str(e) or "planner timeout",
                    exception_type="TimeoutError",
                )
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                last_output = GeneratorOutput(
                    data=None, error=f"Planner step {step} timed out"
                )
                self._emit(
                    ReliabilityEvent(
                        kind="timeout",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
            except Exception as e:
                classified = self._config.classifier.classify(e)
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                last_output = GeneratorOutput(data=None, error=str(e))
                self._emit(
                    ReliabilityEvent(
                        kind="attempt_failed",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
            else:
                if output is not None and not isinstance(output, GeneratorOutput):
                    return output  # type: ignore[return-value]
                if output is not None and output.data is None and output.error:
                    classified = self._config.classifier.classify(output.error)
                    attempts.append(
                        AttemptRecord(
                            attempt=attempt,
                            classified=classified,
                            timestamp=self._now(),
                        )
                    )
                    last_output = output
                    self._emit(
                        ReliabilityEvent(
                            kind="attempt_failed",
                            step=step,
                            operation=operation,
                            attempt=attempt,
                            category=classified.category.value,
                            message=classified.message,
                        )
                    )
                    if not retry.should_retry(classified, attempt, elapsed_delay):
                        return output
                else:
                    return output

            last_classified = (
                attempts[-1].classified
                if attempts
                else ClassifiedError(ErrorCategory.UNKNOWN, False)
            )
            if not retry.should_retry(last_classified, attempt, elapsed_delay):
                stats.retries = max(stats.retries, max(0, len(attempts) - 1))
                self._emit(
                    ReliabilityEvent(
                        kind="retry_exhausted",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=last_classified.category.value,
                        message=last_classified.message,
                    )
                )
                if last_output is not None:
                    return last_output
                return GeneratorOutput(data=None, error="retry exhausted")

            delay = retry.backoff.delay(
                attempt, rng=self._rng, retry_after=last_classified.retry_after
            )
            elapsed_delay += delay
            stats.retries += 1
            self._emit(
                ReliabilityEvent(
                    kind="retry_scheduled",
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    delay=delay,
                    category=last_classified.category.value,
                    message=last_classified.message,
                )
            )
            try:
                await self._sleep(delay)
            except asyncio.CancelledError:
                raise
            attempt += 1

    async def _call_async_with_timeout(
        self,
        fn: Callable[[], Awaitable[GeneratorOutput]],
        timeout: float | None,
    ) -> GeneratorOutput:
        if timeout is None:
            return await fn()
        try:
            return await asyncio.wait_for(fn(), timeout)
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(str(e) or "timeout") from e

    # ------------------------------------------------------------------ tool
    async def run_tool(
        self,
        fn: Callable[[], Awaitable[FunctionOutput]],
        function: Function,
        step: int,
    ) -> FunctionOutput:
        """Run a tool with timeout + retry.

        On timeout returns a FunctionOutput whose ``output`` is a
        ``ToolOutput(status="timeout")`` instead of raising.  Non-idempotent
        tools are never retried after the body may have run.
        """
        operation = f"tool:{function.name}"
        tool_policy = self._config.tool_policy(function.name)
        retry = tool_policy.retry or self._config.retry
        stats = self._stat(operation)
        attempts: list[AttemptRecord] = []
        elapsed_delay = 0.0
        last_output: FunctionOutput | None = None
        attempt = 0
        while True:
            stats.attempts += 1
            remaining = self._remaining_run()
            if remaining is not None and remaining <= 0:
                stats.deadline_exceeded += 1
                self._emit(
                    ReliabilityEvent(
                        kind="deadline_exceeded",
                        step=step,
                        operation=operation,
                        message="run deadline exceeded before tool attempt",
                    )
                )
                raise RunDeadlineExceededError(
                    deadline=self._config.timeouts.run_timeout or 0.0,
                    elapsed=self._now() - (self._run_started or 0.0),
                )
            try:
                output = await self._call_tool_with_timeout(
                    fn, tool_policy, function, step
                )
                return output
            except asyncio.CancelledError:
                raise
            except ToolTimeoutError as e:
                stats.timeouts += 1
                classified = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    retryable=True,
                    message=str(e),
                    exception_type="ToolTimeoutError",
                )
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                self._emit(
                    ReliabilityEvent(
                        kind="timeout",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
                last_output = self._timeout_output(function, tool_policy.timeout)
                if not tool_policy.idempotent:
                    return last_output
            except Exception as e:
                classified = self._config.classifier.classify(e)
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                self._emit(
                    ReliabilityEvent(
                        kind="attempt_failed",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
                err_output = FunctionOutput(
                    name=function.name,
                    input=function,
                    output=ToolOutput(output=None, observation=str(e), status="error"),
                    error=str(e),
                )
                last_output = err_output
                if not tool_policy.idempotent:
                    return err_output

            last_classified = (
                attempts[-1].classified
                if attempts
                else ClassifiedError(ErrorCategory.UNKNOWN, False)
            )
            if not retry.should_retry(last_classified, attempt, elapsed_delay):
                stats.retries = max(stats.retries, max(0, len(attempts) - 1))
                self._emit(
                    ReliabilityEvent(
                        kind="retry_exhausted",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=last_classified.category.value,
                        message=last_classified.message,
                    )
                )
                if last_output is not None:
                    return last_output
                return FunctionOutput(
                    name=function.name,
                    input=function,
                    output=ToolOutput(
                        output=None, observation="retry exhausted", status="error"
                    ),
                    error="retry exhausted",
                )
            delay = retry.backoff.delay(
                attempt, rng=self._rng, retry_after=last_classified.retry_after
            )
            elapsed_delay += delay
            stats.retries += 1
            self._emit(
                ReliabilityEvent(
                    kind="retry_scheduled",
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    delay=delay,
                    category=last_classified.category.value,
                    message=last_classified.message,
                )
            )
            try:
                await self._sleep(delay)
            except asyncio.CancelledError:
                raise
            attempt += 1

    async def _call_tool_with_timeout(
        self,
        fn: Callable[[], Awaitable[FunctionOutput]],
        tool_policy: ToolPolicy,
        function: Function,
        step: int,
    ) -> FunctionOutput:
        timeout = self._config.timeouts.effective(
            tool_policy.timeout,
            self._run_started or self._now(),
            self._now(),
        )
        if timeout is None:
            return await fn()
        try:
            return await asyncio.wait_for(fn(), timeout)
        except asyncio.TimeoutError as e:
            raise ToolTimeoutError(
                tool_name=function.name,
                call_id=function.id,
                timeout=timeout,
            ) from e

    @staticmethod
    def _timeout_output(function: Function, timeout: float | None) -> FunctionOutput:
        msg = f"Tool '{function.name}' timed out after {timeout}s"
        return FunctionOutput(
            name=function.name,
            input=function,
            output=ToolOutput(output=None, observation=msg, status="timeout"),
            error=msg,
        )

    # ------------------------------------------------------------------ sync
    def _thread_pool_get(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self._thread_pool

    def run_planner_sync(
        self,
        fn: Callable[[], GeneratorOutput],
        step: int,
    ) -> GeneratorOutput:
        """Sync version of :meth:`run_planner`.

        Timeout is implemented by running ``fn`` in a worker thread and waiting
        up to ``timeout`` seconds.  A timed-out sync call cannot be killed, only
        abandoned: the thread keeps running in the background.
        """
        operation = f"planner:step{step}"
        retry = self._config.retry
        stats = self._stat(operation)
        attempts: list[AttemptRecord] = []
        elapsed_delay = 0.0
        last_output: GeneratorOutput | None = None
        attempt = 0
        while True:
            stats.attempts += 1
            remaining = self._remaining_run()
            if remaining is not None and remaining <= 0:
                stats.deadline_exceeded += 1
                raise RunDeadlineExceededError(
                    deadline=self._config.timeouts.run_timeout or 0.0,
                    elapsed=self._now() - (self._run_started or 0.0),
                )
            try:
                output = self._call_sync_with_timeout(
                    fn,
                    self._config.timeouts.effective(
                        self._config.timeouts.planner_timeout,
                        self._run_started or self._now(),
                        self._now(),
                    ),
                )
            except concurrent.futures.TimeoutError:
                stats.timeouts += 1
                classified = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    retryable=True,
                    message="planner sync timeout",
                    exception_type="TimeoutError",
                )
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                last_output = GeneratorOutput(
                    data=None, error=f"Planner step {step} timed out"
                )
                self._emit(
                    ReliabilityEvent(
                        kind="timeout",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
            except Exception as e:
                classified = self._config.classifier.classify(e)
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                last_output = GeneratorOutput(data=None, error=str(e))
                self._emit(
                    ReliabilityEvent(
                        kind="attempt_failed",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
            else:
                if output is not None and not isinstance(output, GeneratorOutput):
                    return output  # type: ignore[return-value]
                if output is not None and output.data is None and output.error:
                    classified = self._config.classifier.classify(output.error)
                    attempts.append(
                        AttemptRecord(
                            attempt=attempt,
                            classified=classified,
                            timestamp=self._now(),
                        )
                    )
                    last_output = output
                    if not retry.should_retry(classified, attempt, elapsed_delay):
                        return output
                else:
                    return output

            last_classified = (
                attempts[-1].classified
                if attempts
                else ClassifiedError(ErrorCategory.UNKNOWN, False)
            )
            if not retry.should_retry(last_classified, attempt, elapsed_delay):
                stats.retries = max(stats.retries, max(0, len(attempts) - 1))
                self._emit(
                    ReliabilityEvent(
                        kind="retry_exhausted",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=last_classified.category.value,
                        message=last_classified.message,
                    )
                )
                if last_output is not None:
                    return last_output
                return GeneratorOutput(data=None, error="retry exhausted")

            delay = retry.backoff.delay(
                attempt, rng=self._rng, retry_after=last_classified.retry_after
            )
            elapsed_delay += delay
            stats.retries += 1
            self._emit(
                ReliabilityEvent(
                    kind="retry_scheduled",
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    delay=delay,
                    category=last_classified.category.value,
                    message=last_classified.message,
                )
            )
            try:
                self._sync_sleep(delay)
            except Exception:  # pragma: no cover - sync sleep rarely raises
                pass
            attempt += 1

    def _call_sync_with_timeout(
        self,
        fn: Callable[[], GeneratorOutput],
        timeout: float | None,
    ) -> GeneratorOutput:
        if timeout is None:
            return fn()
        pool = self._thread_pool_get()
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            raise concurrent.futures.TimeoutError(str(e) or "sync timeout") from e

    def run_tool_sync(
        self,
        fn: Callable[[], FunctionOutput],
        function: Function,
        step: int,
    ) -> FunctionOutput:
        """Sync version of :meth:`run_tool`."""
        operation = f"tool:{function.name}"
        tool_policy = self._config.tool_policy(function.name)
        retry = tool_policy.retry or self._config.retry
        stats = self._stat(operation)
        attempts: list[AttemptRecord] = []
        elapsed_delay = 0.0
        last_output: FunctionOutput | None = None
        attempt = 0
        while True:
            stats.attempts += 1
            remaining = self._remaining_run()
            if remaining is not None and remaining <= 0:
                stats.deadline_exceeded += 1
                raise RunDeadlineExceededError(
                    deadline=self._config.timeouts.run_timeout or 0.0,
                    elapsed=self._now() - (self._run_started or 0.0),
                )
            try:
                output = self._call_tool_sync_with_timeout(fn, tool_policy, function)
                return output
            except concurrent.futures.TimeoutError:
                stats.timeouts += 1
                classified = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    retryable=True,
                    message=f"tool {function.name} sync timeout",
                    exception_type="TimeoutError",
                )
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                self._emit(
                    ReliabilityEvent(
                        kind="timeout",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
                last_output = self._timeout_output(function, tool_policy.timeout)
                if not tool_policy.idempotent:
                    return last_output
            except Exception as e:
                classified = self._config.classifier.classify(e)
                attempts.append(
                    AttemptRecord(
                        attempt=attempt, classified=classified, timestamp=self._now()
                    )
                )
                self._emit(
                    ReliabilityEvent(
                        kind="attempt_failed",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=classified.category.value,
                        message=classified.message,
                    )
                )
                err_output = FunctionOutput(
                    name=function.name,
                    input=function,
                    output=ToolOutput(output=None, observation=str(e), status="error"),
                    error=str(e),
                )
                last_output = err_output
                if not tool_policy.idempotent:
                    return err_output

            last_classified = (
                attempts[-1].classified
                if attempts
                else ClassifiedError(ErrorCategory.UNKNOWN, False)
            )
            if not retry.should_retry(last_classified, attempt, elapsed_delay):
                stats.retries = max(stats.retries, max(0, len(attempts) - 1))
                self._emit(
                    ReliabilityEvent(
                        kind="retry_exhausted",
                        step=step,
                        operation=operation,
                        attempt=attempt,
                        category=last_classified.category.value,
                        message=last_classified.message,
                    )
                )
                if last_output is not None:
                    return last_output
                return FunctionOutput(
                    name=function.name,
                    input=function,
                    output=ToolOutput(
                        output=None, observation="retry exhausted", status="error"
                    ),
                    error="retry exhausted",
                )
            delay = retry.backoff.delay(
                attempt, rng=self._rng, retry_after=last_classified.retry_after
            )
            elapsed_delay += delay
            stats.retries += 1
            self._emit(
                ReliabilityEvent(
                    kind="retry_scheduled",
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    delay=delay,
                    category=last_classified.category.value,
                    message=last_classified.message,
                )
            )
            try:
                self._sync_sleep(delay)
            except Exception:  # pragma: no cover
                pass
            attempt += 1

    def _call_tool_sync_with_timeout(
        self,
        fn: Callable[[], FunctionOutput],
        tool_policy: ToolPolicy,
        function: Function,
    ) -> FunctionOutput:
        timeout = self._config.timeouts.effective(
            tool_policy.timeout,
            self._run_started or self._now(),
            self._now(),
        )
        if timeout is None:
            return fn()
        pool = self._thread_pool_get()
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            raise concurrent.futures.TimeoutError(str(e) or "sync timeout") from e
