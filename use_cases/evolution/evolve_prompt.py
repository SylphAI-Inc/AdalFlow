"""
Evolve a prompt, not a program: a support-ticket routing instruction.

The artifact is the instruction text given to a ``Generator``. The evaluator runs it
on labeled tickets and reports accuracy plus the misrouted tickets, which the
proposer reads to work out the routing policy (e.g. "delivered but not here" goes to
``fulfillment``, not ``logistics``). The final instruction is scored on held-out
tickets.

    python use_cases/evolution/evolve_prompt.py --provider openai --model gpt-4o-mini
    python use_cases/evolution/evolve_prompt.py --provider anthropic --model claude-haiku-5-5

Requires an API key for the chosen provider.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "adalflow") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "adalflow"))

from adalflow.core.generator import Generator  # noqa: E402
from adalflow.optim.evolution import LLMProposer, evolve  # noqa: E402

LABELS = ["payments", "fulfillment", "logistics", "trust_safety", "privacy", "care"]

# (ticket, queue). The routing policy, including its less obvious rules (missing
# deliveries go to fulfillment, email changes to privacy, promo codes and account
# merges to care), is only visible through these examples.
TRAIN: List[Tuple[str, str]] = [
    ("I was charged twice for order 4471.", "payments"),
    ("The mug arrived cracked, I want my money back.", "fulfillment"),
    ("Where is my package? Tracking hasn't moved in 5 days.", "logistics"),
    ("I can't log in after resetting my password.", "trust_safety"),
    ("Do you sell gift cards?", "care"),
    ("Please cancel the subscription renewal that hit my card today.", "payments"),
    ("Wrong size shoes were delivered, how do I send them back?", "fulfillment"),
    ("Can I change the delivery address for an order that shipped?", "logistics"),
    ("How do I update the email on my profile?", "privacy"),
    ("Tracking says delivered but there is no parcel anywhere.", "fulfillment"),
    ("My invoice shows a tax amount that looks wrong.", "payments"),
    ("Someone logged into my account from another country.", "trust_safety"),
    ("Delete my account and all my data.", "privacy"),
    ("Why did my promo code not apply at checkout?", "care"),
    ("I have two accounts, can you combine them?", "care"),
    ("Is Saturday delivery available?", "logistics"),
]
HOLDOUT: List[Tuple[str, str]] = [
    ("You billed my card for an order I cancelled.", "payments"),
    ("The jacket has a torn seam, I'd like a refund.", "fulfillment"),
    ("My order shows delivered but nothing is at my door.", "fulfillment"),
    ("Two-factor codes never reach my phone.", "trust_safety"),
    ("What are your store opening hours?", "care"),
    ("Please change the email address on my account.", "privacy"),
    ("The discount code from your newsletter is rejected.", "care"),
    ("Can you ship to a P.O. box?", "logistics"),
    ("Please merge my old account into my new one.", "care"),
    ("Send me a copy of all the personal data you hold on me.", "privacy"),
    ("I was charged in euros instead of dollars.", "payments"),
    ("I received someone else's order.", "fulfillment"),
]

SEED = "Route the support ticket to a queue."

TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{ instruction }}
Reply with exactly one label from: {{ labels }}.
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
{{ ticket }}
<END_OF_USER_PROMPT>"""


def make_client(provider: str):
    if provider == "anthropic":
        from adalflow.components.model_client import AnthropicAPIClient

        return AnthropicAPIClient()
    from adalflow.components.model_client import OpenAIClient

    return OpenAIClient()


def make_evaluator(generator: Generator, examples: List[Tuple[str, str]]):
    """instruction text -> {"instance_scores", "feedback"} on ``examples``."""

    def classify(instruction: str, ticket: str) -> str:
        out = generator.call(
            prompt_kwargs={
                "instruction": instruction,
                "labels": ", ".join(LABELS),
                "ticket": ticket,
            }
        )
        return str(out.data or "").strip().strip(".").lower()

    def evaluate(instruction: str) -> Dict:
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(lambda ex: classify(instruction, ex[0]), examples))
        scores, misses = {}, []
        for i, ((ticket, want), got) in enumerate(zip(examples, answers)):
            scores[f"ticket-{i:02d}"] = float(got == want)
            if got != want:
                misses.append(f"{ticket!r}: answered {got!r}, expected {want!r}")
        return {"instance_scores": scores, "feedback": "\n".join(misses)}

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()

    # The task model: classifies tickets with the candidate instruction.
    task_model = Generator(
        model_client=make_client(args.provider),
        model_kwargs={"model": args.model},
        template=TEMPLATE,
        use_cache=False,
    )
    proposer = LLMProposer(
        "An instruction for a support-ticket router. Make the routing policy "
        "explicit so the router sends every ticket to the right queue.",
        kind="prompt",
        model_client=make_client(args.provider),
        model_kwargs={"model": args.model},
    )
    result = evolve(
        SEED,
        make_evaluator(task_model, TRAIN),
        proposer,
        max_evaluations=args.budget,
        stop_score=1.0,
        holdout=make_evaluator(task_model, HOLDOUT),
    )

    print(f"stopped: {result.stopped} after {result.evaluations} evaluations")
    for c in result.archive.candidates:
        print(f"  {c.id} (parent {c.parent_id}): train accuracy {c.score:.2f}")
    seed, best = result.holdout["seed"], result.holdout.get("best")
    print(f"held-out accuracy: seed {seed.score:.2f} -> best {best.score:.2f}")
    print("\nbest instruction:\n" + result.best.text)


if __name__ == "__main__":
    main()
