"""
Refine the harness of an existing Agent, then check it on held-out tasks.

The agent answers account questions with a paginated ``list_transactions`` tool, but
its system prompt is stale: it says the tool returns everything in one call. The
optimizer runs the agent, has an editor model propose harness changes from the
trajectories (no labels), and picks a final harness blind. Labels are used only at
the end, to score the base and refined harnesses on tasks the refinement never saw.

    python use_cases/harness_optimization/refine_existing_agent.py \
        --provider openai --agent-model gpt-4o-mini --editor-model gpt-4o

Requires an API key for the chosen provider.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "adalflow") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "adalflow"))

from adalflow.components.agent import Agent, Runner  # noqa: E402
from adalflow.optim.harness import HarnessOptimizer  # noqa: E402

STALE_PROMPT = (
    "You answer questions about customer accounts using the tools. "
    "list_transactions returns all of an account's transactions in one call. "
    "End with 'Answer: <number>'."
)


def _amounts(account_id: str) -> List[float]:
    rng = random.Random(account_id)
    return [round(rng.uniform(-80, 200), 2) for _ in range(rng.randint(12, 20))]


def list_transactions(account_id: str, cursor: str = "") -> str:
    """List transactions for an account."""
    rows = _amounts(account_id)
    start = int(cursor[1:]) if re.fullmatch(r"c\d+", cursor or "") else 0
    end = start + 5
    return json.dumps(
        {
            "items": [
                {"id": f"t{i}", "amount": rows[i]}
                for i in range(start, min(end, len(rows)))
            ],
            "next_cursor": f"c{end}" if end < len(rows) else None,
        }
    )


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (numbers, + - * / and parentheses)."""
    if not re.fullmatch(r"[0-9.+\-*/() eE]+", expression):
        return "error: only numbers and + - * / ( ) are allowed"
    return str(round(eval(expression, {"__builtins__": {}}), 6))


def question(account_id: str) -> str:
    return f"What is the total of all transactions in account {account_id}?"


def score(task_id: str, probe) -> float:
    """1.0 if the last number in the answer is the account's true total."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", (probe.answer or "").replace(",", ""))
    want = round(sum(_amounts(task_id)), 2)
    return float(bool(numbers) and abs(float(numbers[-1]) - want) < 0.01)


def make_client(provider: str):
    if provider == "anthropic":
        from adalflow.components.model_client import AnthropicAPIClient

        return AnthropicAPIClient()
    from adalflow.components.model_client import OpenAIClient

    return OpenAIClient()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--agent-model", default="gpt-4o-mini")
    parser.add_argument("--editor-model", default="gpt-4o")
    parser.add_argument("--out", type=Path, default=Path("runs/accounts_harness.json"))
    args = parser.parse_args()

    # 1. The agent you already have.
    agent = Agent(
        name="accounts",
        role_desc=STALE_PROMPT,
        tools=[list_transactions, calculator],
        model_client=make_client(args.provider),
        model_kwargs={"model": args.agent_model},
        max_steps=8,
    )

    # 2. An optimizer that keeps its model, tools and settings; a stronger model edits.
    opt = HarnessOptimizer.from_agent(
        agent,
        editor_client=make_client(args.provider),
        editor_model_kwargs={"model": args.editor_model},
        k=1,
    )
    print(f"at most {opt.estimate(1).total_calls_max} model calls for one task")

    # 3. Refine on one task. No answer key is involved.
    res = opt.refine(question("acct-1"), task_id="acct-1")
    best = res.best_harness
    print(
        f"status: {res.status}, rollouts R={res.ledger.rollouts}, sessions P={res.ledger.sessions}"
    )
    print("base answers:", [r.answer for r in res.base_runs])
    print("selected answer:", res.selected.answer if res.selected else None)
    print("changes:", opt.base.diff(best).summary())
    print("refined system prompt:\n" + best.system_prompt)
    best.save(args.out)

    # 4. Measure on held-out tasks, with labels, 3 rollouts each.
    held_out = [(a, question(a)) for a in ("acct-11", "acct-12", "acct-13")]
    before = opt.evaluate(opt.base, held_out, score, repeats=3, workers=3)
    after = opt.evaluate(best, held_out, score, repeats=3, workers=3)
    print(f"held-out accuracy: base {before.mean:.2f} -> refined {after.mean:.2f}")

    # 5. Deploy: an Agent built from the refined harness (same model and tools).
    deployed = opt.build_agent(best, name="accounts_v2")
    answer = Runner(agent=deployed).call(
        prompt_kwargs={"input_str": question("acct-20")}
    )
    print("deployed agent:", answer.answer)


if __name__ == "__main__":
    main()
