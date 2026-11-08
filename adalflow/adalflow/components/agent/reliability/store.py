"""Checkpoint storage backends.

* :class:`InMemoryCheckpointStore` - thread-safe, deep-copies to avoid aliasing.
* :class:`FileCheckpointStore` - one JSON file per run, atomic writes, optional
  history rotation, path-traversal-safe run ids, and a cross-process advisory
  lock file when ``fcntl`` is available.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

from .checkpoint import RunCheckpoint
from .errors import CheckpointCorruptedError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
@dataclass
class CheckpointSummary:
    """Lightweight summary used by :meth:`CheckpointStore.list_runs`."""

    run_id: str
    status: str
    step_count: int
    updated_at: float


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class CheckpointStore(Protocol):
    """Storage backend protocol for :class:`RunCheckpoint`."""

    def save(self, cp: RunCheckpoint) -> None: ...

    def load(self, run_id: str) -> RunCheckpoint | None: ...

    def delete(self, run_id: str) -> None: ...

    def list_runs(self, status: str | None = None) -> list[CheckpointSummary]: ...

    async def asave(self, cp: RunCheckpoint) -> None:
        await asyncio.to_thread(self.save, cp)

    async def aload(self, run_id: str) -> RunCheckpoint | None:
        return await asyncio.to_thread(self.load, run_id)

    async def adelete(self, run_id: str) -> None:
        await asyncio.to_thread(self.delete, run_id)

    async def alist_runs(self, status: str | None = None) -> list[CheckpointSummary]:
        return await asyncio.to_thread(self.list_runs, status)


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------
class InMemoryCheckpointStore:
    """Thread-safe in-memory checkpoint store.

    Deep-copies on save and load so callers cannot mutate stored checkpoints by
    holding onto a reference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, RunCheckpoint] = {}

    def save(self, cp: RunCheckpoint) -> None:
        with self._lock:
            self._store[cp.run_id] = copy.deepcopy(cp)

    def load(self, run_id: str) -> RunCheckpoint | None:
        with self._lock:
            cp = self._store.get(run_id)
            return copy.deepcopy(cp) if cp is not None else None

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._store.pop(run_id, None)

    def list_runs(self, status: str | None = None) -> list[CheckpointSummary]:
        with self._lock:
            items = list(self._store.values())
        summaries = [
            CheckpointSummary(
                run_id=cp.run_id,
                status=cp.status,
                step_count=cp.step_count,
                updated_at=cp.updated_at,
            )
            for cp in items
            if status is None or cp.status == status
        ]
        summaries.sort(key=lambda s: s.updated_at)
        return summaries


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _sanitize_run_id(run_id: str) -> str:
    """Return a filesystem-safe filename for ``run_id``.

    Rejects path traversal (``/``, ``..``, NUL) via a strict allow-list regex and
    falls back to a SHA-256 hash of the original id when the regex fails.  This
    means a caller-controlled ``run_id`` can never escape the store directory.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if "\x00" in run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        return hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    if _SAFE_RUN_ID_RE.match(run_id):
        return run_id
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def _try_flock(path: str) -> Any | None:
    """Best-effort cross-process advisory lock.  No-op on Windows."""
    try:
        import fcntl  # type: ignore
    except ImportError:
        return None  # Windows
    lock_file = path + ".lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        os.close(fd)
        return None

    class _Lock:
        def __init__(self, fd: int) -> None:
            self._fd = fd

        def release(self) -> None:
            try:
                import fcntl  # type: ignore

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)

    return _Lock(fd)


class FileCheckpointStore:
    """One JSON file per run, written atomically.

    Atomicity: write to a temp file in the same directory, ``fsync``, then
    ``os.replace``.  A crash during the write leaves the previous file intact.

    Optional history rotation: when ``keep_history > 0``, the previous primary
    file is rotated to ``<id>.<n>.json`` before the new one is written, up to
    ``keep_history`` generations.  :meth:`load` falls back to the newest valid
    history file if the primary is corrupted.
    """

    def __init__(
        self,
        directory: str,
        keep_history: int = 0,
        fsync: bool = True,
    ) -> None:
        if keep_history < 0:
            raise ValueError("keep_history must be >= 0")
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)
        self._keep_history = keep_history
        self._fsync = fsync
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _lock_for(self, safe_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(safe_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[safe_id] = lock
            return lock

    def _primary_path(self, safe_id: str) -> str:
        return os.path.join(self._directory, f"{safe_id}.json")

    def _history_paths(self, safe_id: str) -> list[str]:
        return [
            os.path.join(self._directory, f"{safe_id}.{n}.json")
            for n in range(1, self._keep_history + 1)
        ]

    def _rotate(self, safe_id: str) -> None:
        if self._keep_history <= 0:
            return
        primary = self._primary_path(safe_id)
        if not os.path.exists(primary):
            return
        # Shift existing history files: <n> -> <n+1>, oldest dropped.
        for n in range(self._keep_history, 0, -1):
            src = (
                self._primary_path(safe_id)
                if n == 1
                else os.path.join(self._directory, f"{safe_id}.{n - 1}.json")
            )
            dst = os.path.join(self._directory, f"{safe_id}.{n}.json")
            if os.path.exists(src):
                try:
                    os.replace(src, dst)
                except OSError as e:  # pragma: no cover - best effort
                    log.warning("failed to rotate %s -> %s: %s", src, dst, e)

    def _atomic_write(self, path: str, data: str) -> None:
        tmp = tempfile.NamedTemporaryFile(
            dir=self._directory, delete=False, mode="w", encoding="utf-8"
        )
        try:
            tmp.write(data)
            tmp.flush()
            if self._fsync:
                os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, path)

    # ------------------------------------------------------------------ public
    def save(self, cp: RunCheckpoint) -> None:
        safe_id = _sanitize_run_id(cp.run_id)
        lock = self._lock_for(safe_id)
        with lock:
            cross_lock = _try_flock(self._primary_path(safe_id))
            try:
                self._rotate(safe_id)
                self._atomic_write(self._primary_path(safe_id), cp.to_json())
            finally:
                if cross_lock is not None:
                    cross_lock.release()

    def load(self, run_id: str) -> RunCheckpoint | None:
        safe_id = _sanitize_run_id(run_id)
        lock = self._lock_for(safe_id)
        with lock:
            primary = self._primary_path(safe_id)
            if os.path.exists(primary):
                try:
                    with open(primary, "r", encoding="utf-8") as f:
                        text = f.read()
                    return RunCheckpoint.from_json(text)
                except CheckpointCorruptedError as e:
                    log.warning(
                        "primary checkpoint for %s corrupted (%s); trying history",
                        safe_id,
                        e.reason,
                    )
            # Fall back to newest valid history file.
            for n in range(1, self._keep_history + 1):
                hist = os.path.join(self._directory, f"{safe_id}.{n}.json")
                if not os.path.exists(hist):
                    continue
                try:
                    with open(hist, "r", encoding="utf-8") as f:
                        text = f.read()
                    return RunCheckpoint.from_json(text)
                except CheckpointCorruptedError as e:
                    log.warning("history %s corrupted: %s", hist, e.reason)
                    continue
            return None

    def delete(self, run_id: str) -> None:
        safe_id = _sanitize_run_id(run_id)
        lock = self._lock_for(safe_id)
        with lock:
            for path in [self._primary_path(safe_id), *self._history_paths(safe_id)]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue
            lockfile = self._primary_path(safe_id) + ".lock"
            try:
                os.remove(lockfile)
            except FileNotFoundError:
                pass

    def list_runs(self, status: str | None = None) -> list[CheckpointSummary]:
        summaries: list[CheckpointSummary] = []
        if not os.path.isdir(self._directory):
            return summaries
        for name in os.listdir(self._directory):
            if not name.endswith(".json") or name.endswith(".lock.json"):
                continue
            # Skip history files like "<id>.<n>.json" - we only list primaries.
            base = name[: -len(".json")]
            if re.match(r"^[A-Za-z0-9._-]+\.\d+$", base):
                continue
            path = os.path.join(self._directory, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                cp = RunCheckpoint.from_json(text)
            except CheckpointCorruptedError:
                continue
            except OSError:
                continue
            if status is not None and cp.status != status:
                continue
            summaries.append(
                CheckpointSummary(
                    run_id=cp.run_id,
                    status=cp.status,
                    step_count=cp.step_count,
                    updated_at=cp.updated_at,
                )
            )
        summaries.sort(key=lambda s: s.updated_at)
        return summaries

    def prune(
        self,
        older_than: timedelta,
        statuses: list[str] | None = None,
    ) -> int:
        """Delete checkpoints older than ``older_than``.  Returns count removed."""
        cutoff = datetime.now(timezone.utc) - older_than
        removed = 0
        for name in list(os.listdir(self._directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self._directory, name)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            except OSError:
                continue
            if mtime > cutoff:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                cp = RunCheckpoint.from_json(text)
            except (CheckpointCorruptedError, OSError, ValueError):
                # Remove unreadable files too if they're old.
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
                continue
            if statuses is not None and cp.status not in statuses:
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed
