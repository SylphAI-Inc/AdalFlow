"""Reliability layer for :class:`adalflow.components.agent.Runner`.

Opt-in retry + timeout + structured error classification + durable run
checkpoints + resume that never re-executes a tool call that already
completed.  Default behaviour (no policies passed) is byte-for-byte the
current Runner behaviour.
"""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    RunCheckpoint,
    RunFingerprint,
    ToolLedgerEntry,
    register_migration,
)
from .classification import (
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
    default_classifier,
    is_unrecoverable,
)
from .errors import (
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointVersionError,
    ResumeMismatchError,
    RetryExhaustedError,
    RunDeadlineExceededError,
    RunnerReliabilityError,
    StepTimeoutError,
    ToolReplayRefusedError,
    ToolTimeoutError,
)
from .executor import (
    AttemptRecord,
    ExecutorStats,
    ReliabilityEvent,
    ReliableExecutor,
)
from .policy import (
    BackoffPolicy,
    ReliabilityConfig,
    RetryPolicy,
    TimeoutPolicy,
    ToolPolicy,
)
from .store import (
    CheckpointStore,
    CheckpointSummary,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)

__all__ = [
    # classification
    "ClassifiedError",
    "ErrorCategory",
    "ErrorClassifier",
    "default_classifier",
    "is_unrecoverable",
    # checkpoint
    "CHECKPOINT_SCHEMA_VERSION",
    "RunCheckpoint",
    "RunFingerprint",
    "ToolLedgerEntry",
    "register_migration",
    # errors
    "CheckpointCorruptedError",
    "CheckpointError",
    "CheckpointVersionError",
    "ResumeMismatchError",
    "RetryExhaustedError",
    "RunDeadlineExceededError",
    "RunnerReliabilityError",
    "StepTimeoutError",
    "ToolReplayRefusedError",
    "ToolTimeoutError",
    # executor
    "AttemptRecord",
    "ExecutorStats",
    "ReliableExecutor",
    "ReliabilityEvent",
    # policy
    "BackoffPolicy",
    "ReliabilityConfig",
    "RetryPolicy",
    "TimeoutPolicy",
    "ToolPolicy",
    # store
    "CheckpointStore",
    "CheckpointSummary",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
]
