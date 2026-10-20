"""Import Harness² run outputs into ``ModelROILedger`` for cost / cache / quality ROI.

Harness² (https://github.com/google-research/harness2) writes, per namespace
``<bench>-<substrate>-<mode><k>[...]`` under its runs root:

- ``config.json``: run identity, including the ``executor`` and ``improver`` models.
- ``<task>/summary.json``: per-mode stages (label, dim, attempts, usd, fell_back)
  and the ledger's rollout labels.
- ``judged.json``: the judge's score for the run selected per (task, mode).

Each rollout's raw executor stream sits in its run directory. Codex streams
(``codex exec --json``) carry ``turn.completed`` events with ``usage`` = input,
cached input, output and reasoning tokens; those are summed per rollout. OpenCode
streams carry no usage that Harness² parses, so OpenCode rollouts import with
zero tokens (USD from ``summary.json`` is still kept).

Quality: Harness² scores only the run selected for each (task, mode), so that
score (``n_passed / n_criteria``) is attributed to every rollout spent on the task
in that mode. Tasks without a graded score are counted in the report but not
recorded in the ledger.

Usage::

    python benchmarks/harness2_roi/import_harness2_runs.py \
        --runs /path/to/harness2/runs --output artifacts/harness2_roi/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADALFLOW_PKG = PROJECT_ROOT / "adalflow"
if str(ADALFLOW_PKG) not in sys.path:
    sys.path.insert(0, str(ADALFLOW_PKG))

from adalflow.optim.model_roi_ledger import (  # noqa: E402
    InvocationTelemetry,
    ModelROILedger,
)

_MODE_IN_NS = re.compile(r"-(agg|seq)(\d+)")
_STAGE_KIND = re.compile(r"^(base|inv|dom|agg|c)\d+$")


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    turns: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        # Codex/OpenAI usage reports cached tokens as a subset of input tokens.
        return max(0, self.input_tokens - self.cached_input_tokens)

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.turns += other.turns


@dataclass
class Rollout:
    namespace: str
    benchmark: str
    mode: str
    task_id: str
    label: str
    run_dir: Optional[Path]
    usage: TokenUsage
    session_usd: float = 0.0
    attempts: int = 1
    failed: bool = False


@dataclass
class NamespaceReport:
    namespace: str
    executor_model: str
    improver_model: str
    rollouts: List[Rollout] = field(default_factory=list)
    scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    ungraded: List[Tuple[str, str]] = field(default_factory=list)
    usd_by_mode: Dict[str, float] = field(default_factory=dict)


def read_codex_usage(run_dir: Path) -> TokenUsage:
    """Sums ``turn.completed`` usage over every ``*.jsonl`` stream under ``run_dir``."""
    total = TokenUsage()
    for path in sorted(run_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                usage = ev.get("usage") if ev.get("type") == "turn.completed" else None
                if not isinstance(usage, dict):
                    continue
                total.add(
                    TokenUsage(
                        input_tokens=int(usage.get("input_tokens") or 0),
                        cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        reasoning_output_tokens=int(
                            usage.get("reasoning_output_tokens") or 0
                        ),
                        turns=1,
                    )
                )
    return total


def _mode_namespace(namespace: str, mode: str) -> str:
    """``lab-cx-agg2`` -> ``lab-cx-seq2`` for ``mode='seq'`` (seed/tag suffix kept)."""
    return _MODE_IN_NS.sub(lambda m: f"-{mode}{m.group(2)}", namespace, count=1)


def resolve_run_dirs(run_id: str, base_run_dir: Path, base_run_id: str) -> List[Path]:
    """Locates a rollout's directories from a sibling base rollout's recorded path.

    Harness² run ids are ``<mode namespace>/<task>/<label>``. LAB (and the local demo)
    nest them as paths, ``<root>/<run_id>``; JobBench and WorkBuddy flatten them to
    one directory name with ``/`` -> ``__`` (WorkBuddy may add ``__x<n>``), next to
    the base rollout. JobBench keeps its stream in a sibling ``model_traj`` tree.
    """
    flat_base = base_run_id.replace("/", "__")
    if base_run_dir.name.startswith(flat_base):
        flat = run_id.replace("/", "__")
        dirs = sorted(
            p
            for p in base_run_dir.parent.glob(f"{flat}*")
            if p.is_dir() and (p.name == flat or p.name.startswith(f"{flat}__x"))
        )
    else:
        depth = len(Path(base_run_id).parts)
        dirs = [base_run_dir.parents[depth - 1] / run_id]
    return [p for d in dirs for p in with_stream_dirs(d)]


def with_stream_dirs(run_dir: Path) -> List[Path]:
    """``run_dir`` plus JobBench's sibling ``model_traj/<label>`` stream directory."""
    if run_dir.parent.name == "model_output":
        return [run_dir, run_dir.parent.parent / "model_traj" / run_dir.name]
    return [run_dir]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_scores(ns_dir: Path) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    rows = _load_json(ns_dir / "judged.json") or []
    for row in rows if isinstance(rows, list) else []:
        frac = row.get("frac")
        if frac is None and row.get("n_criteria"):
            frac = row["n_passed"] / row["n_criteria"]
        if frac is not None:
            scores[(row.get("task_id", ""), row.get("mode", ""))] = float(frac)
    return scores


def load_namespace(ns_dir: Path) -> NamespaceReport:
    """Collects rollouts, usage and scores for one Harness² namespace directory."""
    namespace = ns_dir.name
    identity = _load_json(ns_dir / "config.json") or {}
    report = NamespaceReport(
        namespace=namespace,
        executor_model=identity.get("executor") or "unknown",
        improver_model=identity.get("improver") or "unknown",
        scores=_load_scores(ns_dir),
    )
    benchmark = namespace.split("-", 1)[0]

    for summary_path in sorted(ns_dir.glob("**/summary.json")):
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        task_id = (
            summary.get("task_id") or summary_path.parent.relative_to(ns_dir).as_posix()
        )
        bases = [
            b
            for b in summary.get("base", [])
            if b.get("label") and b.get("run_dir") and b.get("run_id")
        ]
        base_dirs = {b["label"]: with_stream_dirs(Path(b["run_dir"])) for b in bases}

        for mode, info in (summary.get("modes") or {}).items():
            stages = {s.get("label"): s for s in info.get("stages", [])}
            ledger = info.get("ledger") or {}
            # The mode ledger's usd covers rollouts and improvement sessions.
            report.usd_by_mode[mode] = round(
                report.usd_by_mode.get(mode, 0.0) + float(ledger.get("usd") or 0.0), 6
            )
            for label in ledger.get("rollouts", []):
                if label in base_dirs:
                    run_dirs = base_dirs[label]
                elif bases:
                    run_id = f"{_mode_namespace(namespace, mode)}/{task_id}/{label}"
                    run_dirs = resolve_run_dirs(
                        run_id, Path(bases[0]["run_dir"]), bases[0]["run_id"]
                    )
                else:
                    run_dirs = []
                stage = stages.get(label, {})
                usage = TokenUsage()
                for d in run_dirs:
                    if d.is_dir():
                        usage.add(read_codex_usage(d))
                run_dir = next((d for d in run_dirs if d.is_dir()), None)
                report.rollouts.append(
                    Rollout(
                        namespace=namespace,
                        benchmark=benchmark,
                        mode=mode,
                        task_id=task_id,
                        label=label,
                        run_dir=run_dir,
                        usage=usage,
                        session_usd=float(stage.get("usd") or 0.0),
                        attempts=int(stage.get("attempts") or 1),
                        failed=bool(stage.get("fell_back") or stage.get("error")),
                    )
                )
            if (task_id, mode) not in report.scores:
                report.ungraded.append((task_id, mode))
    return report


def stage_kind(label: str) -> str:
    m = _STAGE_KIND.match(label)
    return m.group(1) if m else label


def to_ledger(
    reports: Iterable[NamespaceReport], ledger: Optional[ModelROILedger] = None
) -> ModelROILedger:
    """Records one ``InvocationTelemetry`` per graded rollout.

    ``task_category`` is ``harness2/<benchmark>/<mode>`` and ``model`` is the executor
    model, so ``summarize_by_task`` compares model configs on the same benchmark.
    """
    ledger = ledger or ModelROILedger()
    for rep in reports:
        for r in rep.rollouts:
            score = rep.scores.get((r.task_id, r.mode))
            if score is None:
                continue
            ledger.record(
                InvocationTelemetry(
                    session_id=f"{r.namespace}/{r.task_id}/{r.label}",
                    task_category=f"harness2/{r.benchmark}/{r.mode}",
                    role="executor",
                    model=rep.executor_model,
                    input_tokens=r.usage.uncached_input_tokens,
                    output_tokens=r.usage.output_tokens,
                    cache_read_tokens=r.usage.cached_input_tokens,
                    total_steps=max(1, r.attempts),
                    error_steps=1 if r.failed else 0,
                    constraint_score=score,
                )
            )
    return ledger


def build_report(runs_root: Path) -> Dict[str, Any]:
    # Only namespaces that hold task summaries; a mode namespace may hold just
    # reflection notes, with its summary living under the shared namespace.
    ns_dirs = sorted(
        p
        for p in runs_root.iterdir()
        if p.is_dir() and _MODE_IN_NS.search(p.name) and any(p.glob("**/summary.json"))
    )
    reports = [load_namespace(d) for d in ns_dirs]
    ledger = to_ledger(reports)

    namespaces = []
    for rep in reports:
        total = TokenUsage()
        by_stage: Dict[str, TokenUsage] = {}
        seen = set()
        for r in rep.rollouts:
            # Base rollouts are shared by both modes; count them once here.
            key = (r.task_id, r.label) if stage_kind(r.label) == "base" else id(r)
            if key in seen:
                continue
            seen.add(key)
            total.add(r.usage)
            by_stage.setdefault(stage_kind(r.label), TokenUsage()).add(r.usage)
        scores = list(rep.scores.values())
        namespaces.append(
            {
                "namespace": rep.namespace,
                "executor_model": rep.executor_model,
                "improver_model": rep.improver_model,
                "rollouts": len(rep.rollouts),
                "rollouts_with_usage": sum(1 for r in rep.rollouts if r.usage.turns),
                "graded": len(scores),
                "ungraded": len(rep.ungraded),
                "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                "usd_by_mode": rep.usd_by_mode,
                "tokens": {
                    "input": total.input_tokens,
                    "cached_input": total.cached_input_tokens,
                    "output": total.output_tokens,
                    "reasoning_output": total.reasoning_output_tokens,
                    "cache_hit_rate": ModelROILedger.compute_cache_hit_rate(
                        total.uncached_input_tokens, total.cached_input_tokens
                    ),
                },
                "input_tokens_by_stage": {
                    k: v.input_tokens for k, v in sorted(by_stage.items())
                },
            }
        )

    return {
        "source": "harness2",
        "runs_root": str(runs_root),
        "namespaces": namespaces,
        "roi_by_task": {
            k: [s.to_dict() for s in v] for k, v in ledger.summarize_by_task().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True, help="Harness² runs root.")
    parser.add_argument(
        "--output", type=Path, help="Optional path for the JSON report."
    )
    args = parser.parse_args()

    report = build_report(args.runs.expanduser().resolve())
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
