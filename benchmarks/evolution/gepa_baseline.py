"""GEPA itself on the prompt tasks of ``prompt_anchoring.py`` (``--task``).

Same task model call (``PROMPT_TEMPLATE``, the task's output format), same exact-match
metric, same search and held-out examples, and the same budget counted in task-model
calls: ``evolve`` at budget 8 runs 8 x (search examples) of them, 192 on ``labels``,
and GEPA gets that as ``max_metric_calls``. GEPA's reflection sees a random minibatch of
``--minibatch`` examples (right or wrong, with the expected label), accepts on that
minibatch and selects parents from its Pareto front. The proposer objective of
``prompt_anchoring.py`` is passed in each reflective record as ``Task``.

``tokens_to_solve`` is proposer plus task tokens at the first full search-set
evaluation that scores 1.0 (GEPA has no early stop, so it keeps spending after it);
held-out scoring has its own meter. ``--early-stop`` ends the run at that point
(GEPA 0.0.7 has no stop hook, so the adapter raises ``Solved`` and the solving
instruction is scored on held-out), which is what ``evolve`` does with
``stop_score=1.0``. ``trace`` lists every task-model batch
(``full`` search set or ``minibatch``, tokens, prompt length) and reflection call.

    pip install gepa==0.0.7  # the version the results use
    python benchmarks/evolution/gepa_baseline.py --provider openai --model gpt-4.1-mini \
        --out runs/factorial/gepa_openai
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import factorial  # noqa: E402
import prompt_anchoring as P  # noqa: E402
from factorial import TokenMeter, make_client  # noqa: E402

from adalflow.optim.evolution.presets import (  # noqa: E402
    PROMPT_TEMPLATE,
    prompt_evaluator,
)


class Solved(Exception):
    """Raised by the adapter at the first full search-set evaluation scoring 1.0."""

    def __init__(self, instruction: str) -> None:
        super().__init__(instruction)
        self.instruction = instruction


class Adapter:
    """GEPA adapter: one component, ``instruction``; DataInst is ``(input, label)``."""

    propose_new_texts = None  # use GEPA's own reflection prompt

    def __init__(
        self, task: P.PromptTask, generator, reflector, early_stop: bool = False
    ) -> None:
        self.task, self.generator, self.reflector = task, generator, reflector
        self.early_stop = early_stop
        self.tokens_to_solve = None
        self.trace: List[Dict[str, Any]] = []

    def _answer(self, instruction: str, item: str) -> str:
        out = self.generator.call(
            prompt_kwargs={
                "instruction": instruction,
                "input": item,
                "output_format": self.task.output_format,
            }
        )
        return str(out.data if out.data is not None else "")

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        before = self.generator.total

        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(
                pool.map(
                    lambda ex: self._answer(candidate["instruction"], ex[0]), batch
                )
            )
        scores = [self.task.metric(a, want) for a, (_, want) in zip(answers, batch)]
        full = len(batch) == len(self.task.search)
        self.trace.append(
            {
                "kind": "full" if full else "minibatch",
                "n": len(batch),
                "tokens": self.generator.total - before,
                "chars": len(candidate["instruction"]),
                "score": sum(scores) / len(scores),
            }
        )
        if full and min(scores) >= 1.0 and self.tokens_to_solve is None:
            self.tokens_to_solve = self.generator.total + self.reflector.generator.total
            if self.early_stop:
                raise Solved(candidate["instruction"])
        traces = (
            [
                {"input": item, "answer": a, "expected": want}
                for a, (item, want) in zip(answers, batch)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=answers, scores=scores, trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for t, s in zip(eval_batch.trajectories, eval_batch.scores):
            records.append(
                {
                    "Task": self.task.objective,
                    "Inputs": t["input"],
                    "Generated Outputs": t["answer"],
                    "Feedback": (
                        "correct"
                        if s >= 1.0
                        else f"wrong; the expected answer is {t['expected']!r}"
                    ),
                }
            )
        return {name: records for name in components_to_update}


class Reflector:
    """``prompt -> text`` for GEPA's reflection step, through an AdalFlow client."""

    def __init__(self, provider: str, model: str) -> None:
        from adalflow.core.generator import Generator

        self.generator = TokenMeter(
            Generator(
                model_client=make_client(provider),
                model_kwargs={"model": model},
                use_cache=False,
            )
        )
        self.calls: List[int] = []

    def __call__(self, prompt: str) -> str:
        before = self.generator.total
        out = self.generator.call(prompt_kwargs={"input_str": prompt})
        self.calls.append(self.generator.total - before)
        return str(out.data if out.data is not None else "")


def one_run(args, repeat: int) -> Dict[str, Any]:
    import gepa
    from adalflow.core.generator import Generator

    pt = P.make_task(args.task)

    def generator():
        return TokenMeter(
            Generator(
                model_client=make_client(args.provider),
                model_kwargs={"model": args.model},
                template=PROMPT_TEMPLATE,
                use_cache=False,
            )
        )

    task, held = generator(), generator()
    reflector = Reflector(args.provider, args.model)
    adapter = Adapter(pt, task, reflector, args.early_stop)
    try:
        res = gepa.optimize(
            seed_candidate={"instruction": pt.seed},
            trainset=pt.search,
            valset=pt.search,
            adapter=adapter,
            reflection_lm=reflector,
            reflection_minibatch_size=args.minibatch,
            max_metric_calls=args.calls or 8 * len(pt.search),
            seed=repeat,
            raise_on_exception=True,
        )
        best = res.best_candidate["instruction"]
        search_best = res.val_aggregate_scores[res.best_idx]
        candidates, metric_calls = len(res.candidates), res.total_metric_calls
    except Solved as stop:
        best, search_best = stop.instruction, 1.0
        candidates = 1 + sum(t["kind"] == "full" for t in adapter.trace[1:])
        metric_calls = sum(t["n"] for t in adapter.trace)
    holdout = prompt_evaluator(
        held,
        pt.holdout,
        metric=pt.metric,
        output_format=pt.output_format,
        show_failures=len(pt.holdout),
    )
    h = holdout(best)
    return {
        "key": f"gepa-mb{args.minibatch}" + ("-stop" if args.early_stop else ""),
        "repeat": repeat,
        "search_best": search_best,
        "holdout_best": sum(h["instance_scores"].values()) / len(pt.holdout),
        "holdout_best_scores": h["instance_scores"],
        "holdout_best_feedback": h["feedback"],
        "trace": adapter.trace,
        "reflections": reflector.calls,
        "candidates": candidates,
        "metric_calls": metric_calls,
        "proposer_tokens": reflector.generator.total,
        "task_tokens": task.total,
        "holdout_tokens": held.total,
        "tokens_to_solve": adapter.tokens_to_solve,
        "best_chars": len(best),
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--task", choices=P.TASKS, default="labels")
    parser.add_argument("--calls", type=int, help="default: 8 x search examples")
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda r: one_run(args, r), range(args.repeats)))
    stop = "-stop" if args.early_stop else ""
    path = args.out / f"summary-mb{args.minibatch}{stop}.json"
    path.write_text(json.dumps(rows, indent=2))
    pt = P.make_task(args.task)
    hb = [r["holdout_best"] for r in rows]
    covered = [
        P.labels_covered(r["holdout_best_scores"], pt.holdout_groups) for r in rows
    ]
    solved = sorted(r["tokens_to_solve"] for r in rows if r["tokens_to_solve"])
    print(
        f"gepa {args.task} minibatch {args.minibatch}: held-out best "
        f"{factorial._fmt(hb, random.Random(0))}, search best "
        f"{[round(r['search_best'], 3) for r in rows]}, candidates "
        f"{[r['candidates'] for r in rows]}, tokens/run "
        f"{sum(r['proposer_tokens'] + r['task_tokens'] for r in rows) / len(rows):.0f}, "
        f"solved {len(solved)}/{len(rows)}, tokens to solve {solved}, best chars "
        f"{sum(r['best_chars'] for r in rows) / len(rows):.0f}"
    )
    print({g: sum(g in c for c in covered) for g in dict.fromkeys(pt.holdout_groups)})
    if args.task == "replies":
        print(
            f"judge: {P.make_task('replies').metric.calls} calls, "
            f"{P.make_task('replies').metric.total} tokens"
        )


if __name__ == "__main__":
    main()
