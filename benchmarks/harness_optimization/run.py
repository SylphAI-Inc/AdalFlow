"""Held-out check for label-free harness optimization on a tool-using domain.

The agent answers account questions with a paginated ``list_transactions`` tool and a
``calculator``. The base harness is stale: it says the tool returns every transaction
in one call, which was true of an older tool version. ``run_domain`` refines the
harness on the training tasks without labels; the refined harness is then scored
with labels on held-out tasks (other accounts, and one question type the training
tasks never ask). Conditions, all scored on the same held-out tasks:

- ``base``: the stale harness.
- ``prompt_only``: the editor model rewrites the prompt once from the tool docs and
  the training task texts, without running the agent (no trajectories).
- ``refined``: the harness selected on the last training task by ``run_domain``.
- ``hand_fixed``: the stale sentence replaced by a correct one (reference point).

    python benchmarks/harness_optimization/run.py --provider anthropic \
        --agent-model claude-haiku-4-5-20251001 --editor-model claude-haiku-5-5 \
        --trials 3 --repeats 4 --out runs/harness_bench

Requires an API key for the chosen provider; not run in CI.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adalflow"))

from adalflow.core.func_tool import FunctionTool  # noqa: E402
from adalflow.core.generator import Generator  # noqa: E402
from adalflow.optim.harness import Harness, HarnessOptimizer, ProbeResult  # noqa: E402

PAGE_SIZE = 5
INTRO = "You answer questions about customer accounts using the tools."
FORMAT = "End with 'Answer: <number>'."
STALE = Harness(
    system_prompt=f"{INTRO} list_transactions returns all of an account's "
    f"transactions in one call. {FORMAT}"
)
HAND_FIXED = Harness(
    system_prompt=f"{INTRO} list_transactions is paginated: call it again with the "
    f"returned next_cursor until next_cursor is null. {FORMAT}"
)


def _amounts(account_id: str) -> List[float]:
    rng = random.Random(account_id)
    return [round(rng.uniform(-80, 200), 2) for _ in range(rng.randint(12, 20))]


def list_transactions(account_id: str, cursor: str = "") -> str:
    """List transactions for an account."""
    if not re.fullmatch(r"acct-\d+", account_id):
        return json.dumps({"error": f"unknown account {account_id}"})
    rows = _amounts(account_id)
    start = int(cursor[1:]) if re.fullmatch(r"c\d+", cursor or "") else 0
    page = rows[start : start + PAGE_SIZE]
    end = start + PAGE_SIZE
    return json.dumps(
        {
            "items": [{"id": f"t{start + i}", "amount": a} for i, a in enumerate(page)],
            "next_cursor": f"c{end}" if end < len(rows) else None,
        }
    )


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (numbers, + - * / and parentheses)."""
    if not re.fullmatch(r"[0-9.+\-*/() eE]+", expression):
        return "error: only numbers and + - * / ( ) are allowed"
    try:
        return str(round(eval(expression, {"__builtins__": {}}), 6))
    except Exception as exc:
        return f"error: {exc}"


# question type -> (template, answer from the account's amounts)
QUESTIONS: Dict[str, Tuple[str, Callable[[List[float]], float]]] = {
    "total": ("What is the total of all transactions in account {a}?", sum),
    "count_over": (
        "How many transactions in account {a} are larger than 100?",
        lambda xs: sum(1 for x in xs if x > 100),
    ),
    "refunds": (
        "What is the total of the refunds (negative amounts) in account {a}?",
        lambda xs: sum(x for x in xs if x < 0),
    ),
    "largest": ("What is the largest single transaction in account {a}?", max),
}
TRAIN = [("acct-1", "total"), ("acct-2", "count_over"), ("acct-3", "refunds")]
HOLDOUT = [
    ("acct-11", "total"),
    ("acct-12", "count_over"),
    ("acct-13", "refunds"),
    ("acct-14", "largest"),  # a question type no training task asks
]


def tasks(specs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(f"{a}:{q}", QUESTIONS[q][0].format(a=a)) for a, q in specs]


def answer_of(task_id: str) -> float:
    account, question = task_id.split(":")
    return round(QUESTIONS[question][1](_amounts(account)), 2)


def score(task_id: str, probe: ProbeResult) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", (probe.answer or "").replace(",", ""))
    return float(bool(numbers) and abs(float(numbers[-1]) - answer_of(task_id)) < 0.01)


PROMPT_ONLY_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You improve the system prompt of a tool-using agent. Return only the new system
prompt, with no commentary.
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
Current system prompt:
{{ prompt }}

Tools:
{{ tools }}

Example tasks:
{{ examples }}
<END_OF_USER_PROMPT>"""


def prompt_only(client, model_kwargs) -> Harness:
    gen = Generator(
        model_client=client,
        model_kwargs=model_kwargs,
        template=PROMPT_ONLY_TEMPLATE,
        use_cache=False,
    )
    tools = "\n".join(
        f"- {fn.__name__}: {fn.__doc__}" for fn in (list_transactions, calculator)
    )
    out = gen.call(
        prompt_kwargs={
            "prompt": STALE.system_prompt,
            "tools": tools,
            "examples": "\n".join(f"- {text}" for _, text in tasks(TRAIN)),
        }
    )
    text = str(out.data if out.data is not None else out.raw_response or "").strip()
    if out.error or not text:
        raise RuntimeError(f"prompt-only rewrite failed: {out.error}")
    return Harness(system_prompt=text)


def make_client(provider: str):
    if provider == "anthropic":
        from adalflow.components.model_client import AnthropicAPIClient

        return AnthropicAPIClient()
    from adalflow.components.model_client import OpenAIClient

    return OpenAIClient()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--editor-model", required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--mode", choices=["parallel", "sequential"], default="parallel"
    )
    parser.add_argument("--trials", type=int, default=3, help="independent refinements")
    parser.add_argument("--repeats", type=int, default=4, help="rollouts per task")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/harness_bench"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def optimizer(seed: int) -> HarnessOptimizer:
        return HarnessOptimizer(
            model_client=make_client(args.provider),
            model_kwargs={"model": args.agent_model},
            editor_client=make_client(args.provider),
            editor_model_kwargs={"model": args.editor_model},
            base=STALE,
            tools=[FunctionTool(list_transactions), FunctionTool(calculator)],
            agent_kwargs={"max_steps": args.max_steps},
            k=args.k,
            mode=args.mode,
            seed=seed,
        )

    est = optimizer(0).estimate(len(TRAIN), reflect=True)
    print(f"refinement: at most {est.total_calls_max} model calls per trial")
    holdout = tasks(HOLDOUT)
    rows: List[Dict] = []

    def measure(name: str, harness: Harness, trial: Optional[int], **extra) -> None:
        ev = optimizer(0).evaluate(
            harness, holdout, score, repeats=args.repeats, workers=args.workers
        )
        rows.append(
            {
                "condition": name,
                "trial": trial,
                "holdout_mean": ev.mean,
                "holdout_by_task": ev.task_means(),
                "harness": harness.to_dict(),
                **extra,
            }
        )
        print(f"{name:12} trial={trial} held-out {ev.mean:.3f} {ev.task_means()}")

    measure("base", STALE, None)
    measure("hand_fixed", HAND_FIXED, None)
    for trial in range(args.trials):
        opt = optimizer(trial)
        measure(
            "prompt_only",
            prompt_only(make_client(args.provider), {"model": args.editor_model}),
            trial,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runs = opt.run_domain(tasks(TRAIN), "accounts", Path(tmp))
        train = [
            {
                "task": run.task_id,
                "status": run.result.status,
                "selected_correct": (
                    score(run.task_id, run.result.selected)
                    if run.result.selected is not None
                    else None
                ),
                "base_correct": [score(run.task_id, b) for b in run.result.base_runs],
                "lesson_stored": run.stored,
            }
            for run in runs
        ]
        measure("refined", runs[-1].result.best_harness, trial, train=train)

    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = ["| condition | runs | held-out accuracy |", "|---|---|---|"]
    for name in ("base", "prompt_only", "refined", "hand_fixed"):
        means = [r["holdout_mean"] for r in rows if r["condition"] == name]
        spread = f" [{min(means):.3f}, {max(means):.3f}]" if len(means) > 1 else ""
        lines.append(
            f"| {name} | {len(means)} | {statistics.mean(means):.3f}{spread} |"
        )
    table = "\n".join(lines)
    (args.out / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
