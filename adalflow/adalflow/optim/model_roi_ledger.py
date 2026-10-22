"""Cost-, cache-, and quality-aware model ROI ledger and frozen memory snapshot.

Builds on ``optimize_anything`` (#488) and the persistent agent memory tutorial (#450):

1. ``ModelROILedger``: a local JSON ledger of per-``(task_category, model)`` token
   usage, prompt-cache hit rate, step success rate, constraint score and TTFT, with a
   composite ROI score used to recommend a model per task category.
2. ``FrozenPrefixMemorySnapshot``: keeps the system-prompt memory index frozen at
   session start and surfaces mid-session memory writes as a short tail reminder,
   so the cached prompt prefix is not invalidated.
3. ``DecisionCard``: a file-backed, non-expiring human-in-the-loop decision
   checkpoint that a worker can park on instead of blocking on an interactive prompt.

Token counts are whatever the caller records; the ledger does no LLM calls itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "DEFAULT_MODEL_TIER_WEIGHTS",
    "InvocationTelemetry",
    "TaskModelSummary",
    "DecisionOption",
    "DecisionCard",
    "FrozenPrefixMemorySnapshot",
    "ModelROILedger",
]

# Illustrative relative price weights (1.0 = reference tier). Override via
# ``ModelROILedger(model_tier_weights=...)`` with your provider's actual pricing.
DEFAULT_MODEL_TIER_WEIGHTS: Dict[str, float] = {
    "claude-opus-4-high": 2.2,
    "claude-sonnet-4-high": 1.5,
    "gpt-4o": 1.4,
    "gemini-2.5-pro": 1.0,
    "gemini-2.5-flash": 0.35,
    "gemini-2.5-flash-lite": 0.2,
    "qwen-2.5-72b-instruct": 0.4,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class InvocationTelemetry:
    """Telemetry record for a single model or worker invocation."""

    session_id: str
    task_category: str
    role: str  # "coordinator" | "worker" | "evaluator"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    total_steps: int = 1
    error_steps: int = 0
    constraint_score: float = 1.0
    ttft_s: float = 0.0
    stream_s: float = 0.0
    timestamp: str = field(default_factory=_utc_now)


@dataclass
class TaskModelSummary:
    """Aggregated cost-performance metrics for a ``(task_category, model)`` pair."""

    task_category: str
    model: str
    invocations: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_hit_rate: float
    step_success_rate: float
    mean_constraint_score: float
    avg_weighted_cost_per_turn: float
    ttft_avg_s: float
    roi_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionOption:
    """Single selectable choice on a ``DecisionCard``."""

    key: str
    label: str
    detail: str = ""
    recommended: bool = False


@dataclass
class DecisionCard:
    """Non-expiring, file-backed decision checkpoint for worker threads."""

    decision_id: str
    thread_name: str
    question: str
    options: List[DecisionOption]
    context: str = ""
    asked_at: str = field(default_factory=_utc_now)
    selected_key: Optional[str] = None
    write_in: Optional[str] = None
    answered_at: Optional[str] = None

    @property
    def is_answered(self) -> bool:
        return self.answered_at is not None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "DecisionCard":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(
            decision_id=raw["decision_id"],
            thread_name=raw["thread_name"],
            question=raw["question"],
            options=[DecisionOption(**o) for o in raw.get("options", [])],
            context=raw.get("context", ""),
            asked_at=raw.get("asked_at", ""),
            selected_key=raw.get("selected_key"),
            write_in=raw.get("write_in"),
            answered_at=raw.get("answered_at"),
        )

    def answer(
        self, key: str, write_in: Optional[str] = None, path: Optional[str] = None
    ) -> None:
        """Records the answer; a ``write_in`` allows a key outside ``options``."""
        valid_keys = {o.key for o in self.options}
        if key not in valid_keys and not write_in:
            raise ValueError(
                f"Invalid option key {key!r}; expected one of {sorted(valid_keys)}"
            )
        self.selected_key = key
        self.write_in = write_in
        self.answered_at = _utc_now()
        if path:
            self.save(path)


class FrozenPrefixMemorySnapshot:
    """Keeps the memory index in the system prompt frozen for the whole session.

    Rewriting the system-prompt prefix whenever an agent stores a new memory
    invalidates the provider's prompt cache for everything after it. Instead:

    1. The rendered index and its SHA-256 (``frozen_sha``) are captured once, at
       session start, and ``frozen_prefix`` never changes afterwards.
    2. ``build_tail_delta_reminder`` returns a short ``<system-reminder>`` to append
       at the tail of the next turn, only when entries changed since the last call.
    """

    def __init__(self, initial_entries: Optional[Dict[str, str]] = None) -> None:
        self._entries: Dict[str, str] = {
            self._normalize_key(k): v.strip()
            for k, v in (initial_entries or {}).items()
        }
        self._frozen_prefix: str = self._render_index(self._entries)
        self._frozen_sha: str = self._sha256(self._frozen_prefix)
        self._last_notified_sha: str = self._frozen_sha
        self._pending_keys: List[str] = []

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower().replace(" ", "-")

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _render_index(entries: Dict[str, str]) -> str:
        if not entries:
            return "<project_memory_index>\n(empty)\n</project_memory_index>"
        lines = ["<project_memory_index>"]
        lines.extend(f"- {k}: {entries[k]}" for k in sorted(entries))
        lines.append("</project_memory_index>")
        return "\n".join(lines)

    @property
    def frozen_prefix(self) -> str:
        """The session-start memory index; never mutated mid-session."""
        return self._frozen_prefix

    @property
    def frozen_sha(self) -> str:
        return self._frozen_sha

    @property
    def entries(self) -> Dict[str, str]:
        """Current (live) memory entries, including mid-session writes."""
        return dict(self._entries)

    def upsert_memory(self, key: str, summary: str) -> None:
        """Writes or updates a memory entry without touching ``frozen_prefix``."""
        clean_key = self._normalize_key(key)
        self._entries[clean_key] = summary.strip()
        if clean_key not in self._pending_keys:
            self._pending_keys.append(clean_key)

    def build_tail_delta_reminder(self) -> Optional[str]:
        """Returns a tail reminder if memory changed since the last call, else None."""
        current_sha = self._sha256(self._render_index(self._entries))
        if current_sha == self._last_notified_sha:
            self._pending_keys.clear()
            return None
        self._last_notified_sha = current_sha
        changed = ", ".join(f"{k}={self._entries[k]!r}" for k in self._pending_keys)
        self._pending_keys.clear()
        return (
            f"<system-reminder>Project memory updated mid-session ({changed}). "
            "Prefix kept frozen for prompt-cache stability.</system-reminder>"
        )


class ModelROILedger:
    """Tracks per-``(task_category, model)`` telemetry and recommends models by ROI."""

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        model_tier_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.ledger_path = ledger_path
        self.model_tier_weights = dict(DEFAULT_MODEL_TIER_WEIGHTS)
        if model_tier_weights:
            self.model_tier_weights.update(model_tier_weights)
        self._records: List[InvocationTelemetry] = []
        if self.ledger_path and os.path.isfile(self.ledger_path):
            self.load(self.ledger_path)

    @property
    def records(self) -> List[InvocationTelemetry]:
        return list(self._records)

    def record(self, telemetry: InvocationTelemetry) -> None:
        """Appends a telemetry entry and persists it if ``ledger_path`` is set."""
        self._records.append(telemetry)
        if self.ledger_path:
            self.save(self.ledger_path)

    @staticmethod
    def compute_cache_hit_rate(input_tokens: int, cache_read_tokens: int) -> float:
        """Share of prompt tokens served from cache (``input_tokens`` = uncached)."""
        denom = int(input_tokens) + int(cache_read_tokens)
        if denom <= 0:
            return 0.0
        return round(float(cache_read_tokens) / float(denom), 4)

    def compute_weighted_cost(
        self, model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int
    ) -> float:
        """Weighted cost in kilo-token units: cached reads 0.1x, output tokens 3x."""
        tier = self.model_tier_weights.get(model, 1.0)
        raw_units = (
            input_tokens + 0.1 * cache_read_tokens + 3.0 * output_tokens
        ) / 1000.0
        return round(raw_units * tier, 4)

    def summarize_by_task(self) -> Dict[str, List[TaskModelSummary]]:
        """Groups records by ``(task_category, model)``, ranked by ROI score."""
        grouped: Dict[Tuple[str, str], List[InvocationTelemetry]] = {}
        for r in self._records:
            grouped.setdefault((r.task_category, r.model), []).append(r)

        by_task: Dict[str, List[TaskModelSummary]] = {}
        for (task_cat, model), items in grouped.items():
            inv = len(items)
            in_tok = sum(x.input_tokens for x in items)
            out_tok = sum(x.output_tokens for x in items)
            cache_tok = sum(x.cache_read_tokens for x in items)
            tot_steps = sum(max(1, x.total_steps) for x in items)
            err_steps = sum(max(0, x.error_steps) for x in items)

            hit_rate = self.compute_cache_hit_rate(in_tok, cache_tok)
            step_succ = round(1.0 - float(err_steps) / float(max(1, tot_steps)), 4)
            mean_q = round(sum(x.constraint_score for x in items) / float(inv), 4)
            tot_cost = self.compute_weighted_cost(model, in_tok, out_tok, cache_tok)
            avg_cost = round(tot_cost / float(inv), 4)
            ttft_avg = round(sum(x.ttft_s for x in items) / float(inv), 3)

            # Composite ROI: quality * step_success^2 * cache bonus, damped by
            # log(cost) and latency.
            cost_factor = math.log10(max(10.0, avg_cost * 10.0 + 10.0))
            latency_factor = 1.0 + 0.03 * ttft_avg
            cache_bonus = 1.0 + 0.25 * hit_rate
            roi = round(
                (mean_q * (step_succ**2) * cache_bonus)
                / (cost_factor * latency_factor)
                * 100.0,
                2,
            )

            by_task.setdefault(task_cat, []).append(
                TaskModelSummary(
                    task_category=task_cat,
                    model=model,
                    invocations=inv,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_read_tokens=cache_tok,
                    cache_hit_rate=hit_rate,
                    step_success_rate=step_succ,
                    mean_constraint_score=mean_q,
                    avg_weighted_cost_per_turn=avg_cost,
                    ttft_avg_s=ttft_avg,
                    roi_score=roi,
                )
            )

        for summaries in by_task.values():
            summaries.sort(key=lambda s: (s.roi_score, s.invocations), reverse=True)
        return by_task

    def recommend_model(
        self,
        task_category: str,
        min_quality: float = 0.85,
        fallback_model: str = "gemini-2.5-pro",
    ) -> str:
        """Returns the highest-ROI model for ``task_category`` meeting ``min_quality``.

        If no model meets the bar, the highest-ROI model is returned; with no
        history for the task, ``fallback_model`` is returned.
        """
        candidates = self.summarize_by_task().get(task_category, [])
        for cand in candidates:
            if (
                cand.mean_constraint_score >= min_quality
                and cand.step_success_rate >= min_quality
            ):
                return cand.model
        return candidates[0].model if candidates else fallback_model

    def save(self, path: Optional[str] = None) -> None:
        """Atomically writes records and per-task summaries as JSON."""
        target = os.path.abspath(path or self.ledger_path or "model_roi_ledger.json")
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        payload = {
            "updated_at": _utc_now(),
            "total_records": len(self._records),
            "summaries_by_task": {
                k: [s.to_dict() for s in v] for k, v in self.summarize_by_task().items()
            },
            "records": [asdict(r) for r in self._records],
        }
        fd, tmp_path = tempfile.mkstemp(prefix=".roi_", suffix=".json", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._records = [
            InvocationTelemetry(**r)
            for r in data.get("records", [])
            if isinstance(r, dict)
        ]
