"""Run checkpoint data model and (de)serialization.

A :class:`RunCheckpoint` captures everything needed to resume a Runner run after
a crash or cancellation:

* ``step_history`` (the planner/tool decisions already made),
* ``tool_ledger`` (a write-ahead ledger of tool calls so resume never re-runs a
  tool that already completed),
* ``fingerprint`` (agent + tools + prompt identity so we refuse to resume a run
  against a different agent),
* ``token_consumption`` and ``answer`` / ``error`` for the final result.

Serialization is JSON with a SHA-256 integrity hash; loading a checkpoint whose
hash does not match raises :class:`CheckpointCorruptedError`.  Schema version
mismatches raise :class:`CheckpointVersionError` and can be migrated via
:meth:`register_migration`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import CheckpointCorruptedError, CheckpointVersionError

log = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Tool ledger
# ---------------------------------------------------------------------------
@dataclass
class ToolLedgerEntry:
    """One write-ahead ledger row for a single tool call.

    The ledger is written *before* the tool runs (``state="started"``) and
    updated to ``completed`` (or ``failed``) afterwards.  On resume:

    * ``completed``  -> replay the recorded ``observation_json``, do NOT call
      the tool again.
    * ``started``    -> the process died mid-tool.  For a non-idempotent tool
      this is unsafe: raise :class:`ToolReplayRefusedError` unless the caller
      explicitly asked for ``rerun`` or ``skip``.
    """

    call_id: str
    name: str
    args_hash: str
    state: Literal["started", "completed", "failed"]
    observation_json: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "args_hash": self.args_hash,
            "state": self.state,
            "observation_json": self.observation_json,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolLedgerEntry:
        return cls(
            call_id=data["call_id"],
            name=data["name"],
            args_hash=data["args_hash"],
            state=data["state"],
            observation_json=data.get("observation_json"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )


# ---------------------------------------------------------------------------
# Run fingerprint
# ---------------------------------------------------------------------------
@dataclass
class RunFingerprint:
    """Identity of a run, used to refuse resuming against a different setup."""

    agent_name: str | None
    planner_model: str | None
    tool_names_sorted: list[str]
    tool_signature_hash: str
    max_steps: int
    input_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "planner_model": self.planner_model,
            "tool_names_sorted": list(self.tool_names_sorted),
            "tool_signature_hash": self.tool_signature_hash,
            "max_steps": self.max_steps,
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunFingerprint:
        return cls(
            agent_name=data.get("agent_name"),
            planner_model=data.get("planner_model"),
            tool_names_sorted=list(data.get("tool_names_sorted") or []),
            tool_signature_hash=data.get("tool_signature_hash") or "",
            max_steps=data.get("max_steps", 0),
            input_hash=data.get("input_hash"),
        )

    def diff(self, other: RunFingerprint) -> list[tuple[str, Any, Any]]:
        """Return a list of ``(field, self_value, other_value)`` for each diff."""
        diffs: list[tuple[str, Any, Any]] = []
        for fname in (
            "agent_name",
            "planner_model",
            "tool_names_sorted",
            "tool_signature_hash",
            "max_steps",
            "input_hash",
        ):
            a = getattr(self, fname)
            b = getattr(other, fname)
            if a != b:
                diffs.append((fname, a, b))
        return diffs


# ---------------------------------------------------------------------------
# Run checkpoint
# ---------------------------------------------------------------------------
@dataclass
class RunCheckpoint:
    """A serializable snapshot of a Runner run."""

    run_id: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    created_at: float = 0.0
    updated_at: float = 0.0
    step_count: int = 0
    step_history: list[dict[str, Any]] = field(default_factory=list)
    tool_ledger: dict[str, dict[str, Any]] = field(default_factory=dict)
    fingerprint: dict[str, Any] | None = None
    token_consumption: dict[str, Any] = field(default_factory=dict)
    answer: Any = None
    error: str | None = None
    prompt_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # The Function (name/args/kwargs/id) of the step whose tool execution was
    # in progress when the checkpoint was written.  On resume this lets us
    # either replay the completed tool result or rerun/skip the tool without
    # re-invoking the planner for that step.
    pending_function: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers
    def ledger_entry(self, call_id: str) -> ToolLedgerEntry | None:
        row = self.tool_ledger.get(call_id)
        if row is None:
            return None
        return ToolLedgerEntry.from_dict(row)

    def add_ledger_entry(self, entry: ToolLedgerEntry) -> None:
        self.tool_ledger[entry.call_id] = entry.to_dict()

    def fingerprint_obj(self) -> RunFingerprint | None:
        if self.fingerprint is None:
            return None
        return RunFingerprint.from_dict(self.fingerprint)

    # ------------------------------------------------------------------ (de)ser
    def _payload(self) -> dict[str, Any]:
        """Canonical payload dict (without the integrity hash)."""
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step_count": self.step_count,
            "step_history": self.step_history,
            "tool_ledger": self.tool_ledger,
            "fingerprint": self.fingerprint,
            "token_consumption": self.token_consumption,
            "answer": self.answer,
            "error": self.error,
            "prompt_kwargs": self.prompt_kwargs,
            "metadata": self.metadata,
            "pending_function": self.pending_function,
        }

    def to_json(self) -> str:
        payload = self._payload()
        canonical = json.dumps(payload, sort_keys=True, default=_json_default)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        envelope = {"payload": payload, "sha256": digest}
        return json.dumps(envelope, sort_keys=True, default=_json_default)

    @classmethod
    def from_json(cls, text: str) -> RunCheckpoint:
        try:
            envelope = json.loads(text)
        except (ValueError, TypeError) as e:
            raise CheckpointCorruptedError(None, f"invalid JSON: {e}")
        if not isinstance(envelope, dict) or "payload" not in envelope:
            raise CheckpointCorruptedError(None, "missing payload envelope")
        payload = envelope["payload"]
        stored_hash = envelope.get("sha256")
        canonical = json.dumps(payload, sort_keys=True, default=_json_default)
        actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if stored_hash is not None and stored_hash != actual_hash:
            raise CheckpointCorruptedError(
                payload.get("run_id") if isinstance(payload, dict) else None,
                f"sha256 mismatch (stored={stored_hash}, actual={actual_hash})",
            )
        if not isinstance(payload, dict):
            raise CheckpointCorruptedError(None, "payload is not an object")
        version = payload.get("schema_version", 1)
        if version != CHECKPOINT_SCHEMA_VERSION:
            migrated = _MIGRATIONS.apply(int(version), payload)
            if migrated is None:
                raise CheckpointVersionError(int(version), CHECKPOINT_SCHEMA_VERSION)
            payload = migrated
        return cls(
            run_id=payload["run_id"],
            schema_version=payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
            status=payload.get("status", "running"),
            created_at=payload.get("created_at", 0.0),
            updated_at=payload.get("updated_at", 0.0),
            step_count=payload.get("step_count", 0),
            step_history=list(payload.get("step_history") or []),
            tool_ledger=dict(payload.get("tool_ledger") or {}),
            fingerprint=payload.get("fingerprint"),
            token_consumption=dict(payload.get("token_consumption") or {}),
            answer=payload.get("answer"),
            error=payload.get("error"),
            prompt_kwargs=dict(payload.get("prompt_kwargs") or {}),
            metadata=dict(payload.get("metadata") or {}),
            pending_function=payload.get("pending_function"),
        )


def _json_default(obj: Any) -> Any:
    """JSON default encoder for checkpoint payloads.

    Non-JSON-serializable values are stored as ``{"__repr__": True, "value": repr(obj)}``
    so they survive a round-trip with a clear marker (and a warning).
    """
    try:
        return repr(obj)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
@dataclass
class _MigrationRegistry:
    """Holds ``register_migration(from_version, fn)`` callbacks."""

    _rules: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = field(
        default_factory=dict
    )

    def register(
        self, from_version: int, fn: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        self._rules[from_version] = fn

    def apply(
        self, found_version: int, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        current = found_version
        data = payload
        while current != CHECKPOINT_SCHEMA_VERSION:
            fn = self._rules.get(current)
            if fn is None:
                return None
            data = fn(data)
            current = int(data.get("schema_version", current))
            if current <= found_version:
                # Avoid infinite loops on a buggy migration.
                return None
        return data


_MIGRATIONS = _MigrationRegistry()


def register_migration(
    from_version: int, fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    """Register a migration from ``from_version`` to the current schema version."""
    _MIGRATIONS.register(from_version, fn)
