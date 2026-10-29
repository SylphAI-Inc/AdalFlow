"""``--task replies``: free-text support replies scored by an LLM judge.

The task model writes the reply to a customer message for a fictional company, Acme.
Each of 8 kinds of message has house policies (refund window, credit threshold,
where to send legal threats...) that the objective does not state, so they come only
from feedback. The expected value of an example is a rubric, ``"a reply that ..."``;
the judge (a fixed model, temperature 0, cached) answers PASS or FAIL for the reply
against it, and the score is 1 or 0. Search examples are grouped by kind.

The judge is the same for every proposer model (``JUDGE_MODEL``); its tokens are
metered on their own (``judge.total``) and are not part of ``tokens``.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

JUDGE_PROVIDER, JUDGE_MODEL = "openai", "gpt-4.1-mini"

CREDIT = "offers a $10 store credit, because the order is more than 7 days late"
NO_CREDIT = (
    "does not offer any credit or discount, because the order is 7 days late or less"
)
TRACK = "gives the tracking page acme.example/track"

REPLIES = {  # kind: ([(message, rubric)] search, [(message, rubric)] held-out)
    "refund": (
        [
            (
                "I returned the blender last week. When do I get my money back?",
                "a reply that says the refund goes to the original payment method "
                "within 5-7 business days, and does not promise a specific date",
            ),
            (
                "Still no refund for order 4471, it's been 3 days since you got "
                "the return!!",
                "a reply that says the refund goes to the original payment method "
                "within 5-7 business days, and does not promise a specific date",
            ),
        ],
        [
            (
                "Can you refund my headphones to a gift card instead of my Visa?",
                "a reply that says the refund goes to the original payment method "
                "(not a gift card) within 5-7 business days",
            ),
            (
                "how long do refunds take",
                "a reply that says the refund goes to the original payment method "
                "within 5-7 business days, and does not promise a specific date",
            ),
        ],
    ),
    "password": (
        [
            (
                "I forgot my password, can you tell me what it is? My username is "
                "jdoe88.",
                "a reply that does not ask for or reveal any password, points to the "
                "'Forgot password' link on the sign-in page, and says the reset link "
                "expires in 30 minutes",
            ),
            (
                "Locked out of my account. Here is my old password: Sunny2020, "
                "please reset it.",
                "a reply that tells the customer never to share a password, points to "
                "the 'Forgot password' link on the sign-in page, and says the reset "
                "link expires in 30 minutes",
            ),
        ],
        [
            (
                "The reset email you sent me doesn't work anymore.",
                "a reply that points to the 'Forgot password' link on the sign-in page "
                "to get a new reset link, and says reset links expire in 30 minutes",
            ),
            (
                "Can support just change my password to Hello123 for me?",
                "a reply that declines to set a password for the customer, points to "
                "the 'Forgot password' link on the sign-in page, and says the reset "
                "link expires in 30 minutes",
            ),
        ],
    ),
    "late_order": (
        [
            (
                "My order was supposed to arrive 9 days ago. Where is it?",
                f"a reply that {TRACK} and {CREDIT}",
            ),
            (
                "Package is 3 days late, any update?",
                f"a reply that {TRACK} and {NO_CREDIT}",
            ),
        ],
        [
            (
                "It's been 12 days past the delivery date and nothing.",
                f"a reply that {TRACK} and {CREDIT}",
            ),
            (
                "My parcel is 5 days late. What is going on?",
                f"a reply that {TRACK} and {NO_CREDIT}",
            ),
        ],
    ),
    "cancel": (
        [
            (
                "Please cancel my subscription.",
                "a reply that says the customer can cancel themselves under Settings > "
                "Billing, that access continues until the end of the current billing "
                "period, and that does not offer a discount to stay",
            ),
            (
                "How do I stop being charged every month?",
                "a reply that says the customer can cancel under Settings > Billing, "
                "that access continues until the end of the current billing period, "
                "and that does not offer a discount to stay",
            ),
        ],
        [
            (
                "I want to end my plan today. Will I lose my files immediately?",
                "a reply that says the customer can cancel under Settings > Billing and "
                "that access continues until the end of the current billing period",
            ),
            (
                "Cancel my account subscription, I'm moving to a competitor.",
                "a reply that says the customer can cancel under Settings > Billing, "
                "that access continues until the end of the current billing period, "
                "and that does not offer a discount to stay",
            ),
        ],
    ),
    "bug": (
        [
            (
                "The app crashes every time I open the camera tab.",
                "a reply that asks for the app version and the device model, and gives "
                "no date or time frame for a fix",
            ),
            (
                "Export to PDF produces a blank page. When will you fix this?",
                "a reply that asks for the app version and the device model, and gives "
                "no date or time frame for a fix",
            ),
        ],
        [
            (
                "Notifications stopped working after the last update.",
                "a reply that asks for the app version and the device model, and gives "
                "no date or time frame for a fix",
            ),
            (
                "Search returns nothing for any query. Fix ETA?",
                "a reply that asks for the app version and the device model, and gives "
                "no date or time frame for a fix",
            ),
        ],
    ),
    "legal": (
        [
            (
                "Your product damaged my laptop. I'm talking to my lawyer.",
                "a reply that does not admit fault, says the matter has been passed to "
                "legal@acme.example, and says they will reply within 2 business days",
            ),
            (
                "I'm filing a chargeback and reporting you to consumer protection.",
                "a reply that does not admit fault, says the matter has been passed to "
                "legal@acme.example, and says they will reply within 2 business days",
            ),
        ],
        [
            (
                "This is a formal notice: you owe me compensation for the data loss.",
                "a reply that does not admit fault, says the matter has been passed to "
                "legal@acme.example, and says they will reply within 2 business days",
            ),
            (
                "My attorney will contact you about the defective heater.",
                "a reply that does not admit fault, says the matter has been passed to "
                "legal@acme.example, and says they will reply within 2 business days",
            ),
        ],
    ),
    "feature": (
        [
            (
                "Please add a dark mode!",
                "a reply that thanks the customer, says the idea is logged on the "
                "public roadmap at acme.example/roadmap, and does not promise it will "
                "be built",
            ),
            (
                "Will you ever support exporting to Excel?",
                "a reply that says the idea is logged on the public roadmap at "
                "acme.example/roadmap, and does not promise it will be built",
            ),
        ],
        [
            (
                "It would be great to have keyboard shortcuts.",
                "a reply that thanks the customer, says the idea is logged on the "
                "public roadmap at acme.example/roadmap, and does not promise it will "
                "be built",
            ),
            (
                "When are you adding a Linux version?",
                "a reply that says the idea is logged on the public roadmap at "
                "acme.example/roadmap, and does not promise it will be built or when",
            ),
        ],
    ),
    "delete": (
        [
            (
                "Delete all my data, I don't want an account anymore.",
                "a reply that asks the customer to confirm by writing from the "
                "account's email address, says deletion is permanent, and says it "
                "completes within 30 days",
            ),
            (
                "Under GDPR I request erasure of my personal data.",
                "a reply that asks the customer to confirm by writing from the "
                "account's email address, says deletion is permanent, and says it "
                "completes within 30 days",
            ),
        ],
        [
            (
                "Remove my account and everything you store about me.",
                "a reply that asks the customer to confirm by writing from the "
                "account's email address, says deletion is permanent, and says it "
                "completes within 30 days",
            ),
            (
                "How long does it take you to erase my data if I ask?",
                "a reply that says deletion completes within 30 days, is permanent, "
                "and must be confirmed from the account's email address",
            ),
        ],
    ),
}

SEED = "Reply to the customer."
OUTPUT_FORMAT = "Reply with the message to send to the customer only."
OBJECTIVE = (
    "A system prompt that makes the model write the reply to a customer support "
    "message for Acme, a company that sells devices, software subscriptions and "
    "accessories online. Replies must follow Acme's support policies."
)
ORACLE = (  # every policy, for checking the judge (not used in search)
    "You write replies to customer support messages for Acme. Follow these policies. "
    "Refunds go to the original payment method within 5-7 business days; never "
    "promise a specific date. Never ask for, accept or set a password; point to the "
    "'Forgot password' link on the sign-in page and say reset links expire in 30 "
    "minutes. For a late order give the tracking page acme.example/track; offer a $10 "
    "store credit only if it is more than 7 days late, otherwise offer no credit. "
    "Cancellations: the customer cancels under Settings > Billing, access continues "
    "until the end of the billing period, offer no discount to stay. Bugs: ask for "
    "the app version and device model, give no time frame for a fix. Legal threats "
    "or chargebacks: do not admit fault, say it was passed to legal@acme.example, "
    "who reply within 2 business days. Feature requests: thank them, say it is "
    "logged on the public roadmap at acme.example/roadmap, promise nothing. Data "
    "deletion: ask them to confirm from the account's email address, say it is "
    "permanent and completes within 30 days."
)

JUDGE_PROMPT = """You check a customer support reply against requirements.

Requirements: {rubric}

Reply:
<<<
{reply}
>>>

Judge the meaning, not the wording. A requirement that the reply does not do something is met when the reply does not do it; it need not say so.

Does the reply meet every requirement? Answer PASS or FAIL on the first line, then one short sentence on why."""


class Judge:
    """``(reply, rubric) -> 1.0 | 0.0``; one judge call per distinct pair."""

    def __init__(self) -> None:
        self.meter = None  # built on first use, so --check needs no API key
        self.off = False  # --check: everything fails, no calls
        self._cache: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self.calls = 0

    def _build(self):
        from factorial import TokenMeter, make_client

        from adalflow.core.generator import Generator

        self.meter = TokenMeter(
            Generator(
                model_client=make_client(JUDGE_PROVIDER),
                model_kwargs={"model": JUDGE_MODEL, "temperature": 0},
                use_cache=False,
            )
        )

    @property
    def total(self) -> int:
        return self.meter.total if self.meter else 0

    def __call__(self, reply, rubric) -> float:
        if self.off:
            return 0.0
        key = (str(reply).strip(), str(rubric))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self.meter is None:
                self._build()
        prompt = JUDGE_PROMPT.format(rubric=key[1], reply=key[0])
        out = self.meter.call(prompt_kwargs={"input_str": prompt})
        verdict = str(out.data or "").strip().upper()
        score = 1.0 if verdict.startswith("PASS") else 0.0
        with self._lock:
            self._cache[key] = score
            self.calls += 1
        return score


judge = Judge()


def data(split: int):
    """``split`` 0: search, 1: held-out; returns (examples, groups)."""
    examples = [ex for kind in REPLIES.values() for ex in kind[split]]
    groups = [k for k, kind in REPLIES.items() for _ in kind[split]]
    return examples, groups


def check_judge(provider: str, model: str, instruction: Optional[str] = None):
    """Scores ``ORACLE`` (or ``instruction``) and the seed on every example."""
    from factorial import make_client

    from adalflow.core.generator import Generator
    from adalflow.optim.evolution.presets import PROMPT_TEMPLATE, prompt_evaluator

    gen = Generator(
        model_client=make_client(provider),
        model_kwargs={"model": model},
        template=PROMPT_TEMPLATE,
        use_cache=False,
    )
    examples = data(0)[0] + data(1)[0]
    ev = prompt_evaluator(
        gen, examples, metric=judge, output_format=OUTPUT_FORMAT, show_failures=32
    )
    return {
        name: ev(text)
        for name, text in (("oracle", instruction or ORACLE), ("seed", SEED))
    }
