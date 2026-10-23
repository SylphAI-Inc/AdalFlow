"""Compare search strategies at an equal evaluation budget, scored on held-out instances.

    python benchmarks/evolution/run.py --provider anthropic --model claude-haiku-5-5 \
        --tasks duration binpack --strategies pareto best resample --repeats 3 \
        --budget 12 --out runs/evolution_bench

``resample`` is the baseline: every proposal is an independent sample from the seed
(best-of-N). ``pareto`` and ``best`` build on earlier candidates; ``map_elites`` keeps
one elite per behavior cell (tasks with a descriptor); ``islands`` is ``pareto`` on 3
islands with migration every 4 proposals. Each run writes its
event log to ``<out>/<task>-<strategy>-<repeat>.jsonl``; ``summary.json`` and
``summary.md`` hold the per-run numbers and the aggregate table. Requires an API key
for the chosen provider; not run in CI. ``--check`` scores each task's seed on both
evaluators without calling a model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adalflow"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adalflow.optim.evolution import (  # noqa: E402
    LLMProposer,
    PythonEvaluator,
    evolve,
    to_evaluation,
)
from tasks import TASKS  # noqa: E402


def make_client(provider: str):
    if provider == "anthropic":
        from adalflow.components.model_client import AnthropicAPIClient

        return AnthropicAPIClient()
    from adalflow.components.model_client import OpenAIClient

    return OpenAIClient()


# strategy name -> evolve() keyword arguments
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "pareto": {"selection": "pareto"},
    "best": {"selection": "best"},
    "resample": {"selection": "resample"},
    "map_elites": {"selection": "map_elites"},
    "islands": {"selection": "pareto", "islands": 3, "migrate_every": 4},
}


def one_run(args, task_name: str, strategy: str, repeat: int) -> Dict[str, Any]:
    task = TASKS[task_name]
    options = dict(STRATEGIES[strategy])
    if task.descriptor is not None:
        options["descriptor"] = task.descriptor
    proposer = LLMProposer(
        task.objective,
        model_client=make_client(args.provider),
        model_kwargs={"model": args.model},
    )
    log_path = args.out / f"{task_name}-{strategy}-{repeat}.jsonl"
    log_path.unlink(missing_ok=True)
    res = evolve(
        task.seed,
        PythonEvaluator(task.search_evaluator, timeout=60),
        proposer,
        max_evaluations=args.budget,
        random_seed=repeat,
        stop_score=task.stop_score,
        max_stalls=args.budget,  # keep sampling to the full budget
        log_path=log_path,
        holdout=PythonEvaluator(task.holdout_evaluator, timeout=60),
        **options,
    )
    curve, best = [], float("-inf")
    for c in res.archive.candidates:
        if c.ok:
            best = max(best, c.score)
        curve.append(round(best, 4) if best > float("-inf") else None)
    return {
        "task": task_name,
        "strategy": strategy,
        "repeat": repeat,
        "evaluations": res.evaluations,
        "stopped": res.stopped,
        "proposal_failures": res.proposal_failures,
        "duplicates": res.duplicates,
        "cells": len(res.archive.elites()) if task.descriptor else None,
        "search_seed": res.seed.score,
        "search_best": res.best.score if res.best else None,
        "holdout_seed": res.holdout["seed"].score,
        "holdout_best": res.holdout["best"].score if "best" in res.holdout else None,
        "best_id": res.best.id if res.best else None,
        "best_generation": res.best.generation if res.best else None,
        "best_so_far": curve,
    }


def _fmt(values: List[float]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = statistics.mean(values)
    return f"{mean:.3f} [{min(values):.3f}, {max(values):.3f}]"


def summarize(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| task | strategy | runs | search best | held-out seed | held-out best "
        "| best generation | evals used | duplicates | cells |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = sorted({(r["task"], r["strategy"]) for r in rows})
    for task, strategy in keys:
        group = [r for r in rows if r["task"] == task and r["strategy"] == strategy]
        lines.append(
            f"| {task} | {strategy} | {len(group)} "
            f"| {_fmt([r['search_best'] for r in group])} "
            f"| {_fmt([r['holdout_seed'] for r in group])} "
            f"| {_fmt([r['holdout_best'] for r in group])} "
            f"| {_fmt([r['best_generation'] for r in group])} "
            f"| {_fmt([r['evaluations'] for r in group])} "
            f"| {_fmt([r['duplicates'] for r in group])} "
            f"| {_fmt([r['cells'] for r in group])} |"
        )
    lines.append("")
    lines.append("Mean [min, max] over runs.")
    return "\n".join(lines)


def check(task_names: List[str]) -> None:
    for name in task_names:
        task = TASKS[name]
        for split, source in (
            ("search", task.search_evaluator),
            ("holdout", task.holdout_evaluator),
        ):
            ev = to_evaluation(PythonEvaluator(source, timeout=60)(task.seed))
            if not ev.ok:
                raise SystemExit(f"{name} {split}: seed evaluation failed: {ev.error}")
            print(f"{name:10} {split:8} seed score {ev.score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS), choices=TASKS)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["pareto", "best", "resample"],
        choices=sorted(STRATEGIES),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=12, help="evaluations per run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/evolution_bench"))
    parser.add_argument("--check", action="store_true", help="seed scores only")
    args = parser.parse_args()
    if args.check:
        return check(args.tasks)
    if not (args.provider and args.model):
        parser.error("--provider and --model are required (or pass --check)")
    for task in args.tasks:
        if "map_elites" in args.strategies and TASKS[task].descriptor is None:
            parser.error(f"task {task!r} has no descriptor for map_elites")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (t, s, r)
        for t in args.tasks
        for s in args.strategies
        for r in range(args.repeats)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: one_run(args, *job), jobs))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    table = summarize(rows)
    (args.out / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
