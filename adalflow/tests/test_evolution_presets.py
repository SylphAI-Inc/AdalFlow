"""Tests for the evolution shortcuts and model specs (scripted models, real sandbox)."""

import pytest

from adalflow.components.model_client import resolve_model
from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput
from adalflow.optim.evolution import (
    CARRY_FACTS,
    PythonEvaluator,
    cases_evaluator,
    exact_match,
    optimize_code,
    optimize_prompt,
    to_evaluation,
)

SEED = "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"
FIXED = (
    "def median(xs):\n"
    "    if not xs:\n"
    "        raise ValueError('empty')\n"
    "    s = sorted(xs)\n"
    "    m = len(s) // 2\n"
    "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"
)
CASES = {
    "odd": (([3, 1, 2],), 2),
    "even": (([4, 1, 3, 2],), 2.5),
    "kwargs": ({"xs": [1, 2]}, 1.5),
    "empty": (([],), ValueError),
}


class ScriptedClient(ModelClient):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply
        self.prompts = []

    def convert_inputs_to_api_kwargs(
        self, input=None, model_kwargs={}, model_type=None
    ):
        return {"prompt": input}

    def call(self, api_kwargs={}, model_type=None):
        self.prompts.append(api_kwargs["prompt"])
        return self.reply(api_kwargs["prompt"])

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data=completion, raw_response=completion)


# ------------------------------------------------------------------ specs
def test_resolve_model_parses_specs_and_passes_clients_through(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client, kwargs = resolve_model("openai/gpt-4o-mini", {"temperature": 0.2})
    assert type(client).__name__ == "OpenAIClient"
    assert kwargs == {"model": "gpt-4o-mini", "temperature": 0.2}
    mine = ScriptedClient(lambda p: "")
    assert resolve_model(mine, {"model": "x"}) == (mine, {"model": "x"})
    for bad in ("gpt-4o-mini", "/x", "openai/"):
        with pytest.raises(ValueError, match="provider/model"):
            resolve_model(bad)
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_model("nope/x")
    with pytest.raises(TypeError):
        resolve_model(42)


# ------------------------------------------------------------------ cases
def test_cases_evaluator_scores_values_kwargs_and_exceptions():
    ev = cases_evaluator("median", CASES)
    seed = to_evaluation(ev(SEED))
    assert seed.instance_scores == {
        "odd": 1.0,
        "even": 0.0,
        "kwargs": 0.0,
        "empty": 0.0,
    }
    assert "even: median([4, 1, 3, 2]) returned 3; expected 2.5" in seed.feedback
    assert "kwargs: median(xs=[1, 2])" in seed.feedback
    assert (
        "raised IndexError" in seed.feedback and "expected ValueError" in seed.feedback
    )
    assert to_evaluation(ev(FIXED)).score == 1.0
    missing = to_evaluation(ev("x = 1\n"))
    assert not missing.ok and "does not define median()" in missing.feedback


def test_cases_evaluator_diverse_order_spreads_shown_failures():
    cases = [("1", 1), ("2", 2), ("3", 3), ("4", 4), ("PT1H", 1), ("1:30", 1)]
    seed = "def parse(text):\n    return None\n"
    ev = cases_evaluator("parse", cases, show_failures=3, show_order="first")
    first = to_evaluation(ev(seed))
    assert "'PT1H'" not in first.feedback and "'1:30'" not in first.feedback
    ev = cases_evaluator("parse", cases, show_failures=3)  # diverse by default
    diverse = to_evaluation(ev(seed))
    assert "parse('1')" in diverse.feedback  # the first failure is always kept
    assert "'PT1H'" in diverse.feedback and "'1:30'" in diverse.feedback
    assert "... 3 more failing cases" in diverse.feedback
    assert diverse.instance_scores == first.instance_scores
    with pytest.raises(ValueError, match="show_order"):
        cases_evaluator("parse", cases, show_order="random")


def test_prompt_evaluator_show_order():
    from adalflow.core.generator import Generator
    from adalflow.optim.evolution import prompt_evaluator
    from adalflow.optim.evolution.presets import PROMPT_TEMPLATE

    examples = [
        ("2024-03-15", "iso_date"),
        ("1999-12-31", "iso_date"),
        ("2001-01-01", "iso_date"),
        ("192.168.1.10", "ipv4"),
        ("#FFF", "hex_color"),
    ]
    generator = Generator(
        model_client=ScriptedClient(lambda p: "date"), template=PROMPT_TEMPLATE
    )
    first = prompt_evaluator(
        generator, examples, show_failures=2, show_order="first", workers=1
    )("x")
    diverse = prompt_evaluator(generator, examples, show_failures=2, workers=1)("x")
    assert "'1999-12-31'" in first["feedback"] and "'#FFF'" not in first["feedback"]
    assert "'2024-03-15'" in diverse["feedback"] and "'#FFF'" in diverse["feedback"]
    assert "... 3 more wrong answers" in diverse["feedback"]
    assert diverse["instance_scores"] == first["instance_scores"]
    with pytest.raises(ValueError, match="show_order"):
        prompt_evaluator(generator, examples, show_order="random")


def test_cases_evaluator_checks_its_inputs(tmp_path):
    listed = to_evaluation(cases_evaluator("median", [(([5],), 5)])(FIXED))
    assert list(listed.instance_scores) == ["case-00"]
    with pytest.raises(ValueError, match="literal"):
        cases_evaluator("median", {"a": ((object(),), 1)})
    with pytest.raises(ValueError, match="identifier"):
        cases_evaluator("not a name", CASES)
    with pytest.raises(ValueError, match="pair"):
        cases_evaluator("median", {"a": [1, 2]})
    script = tmp_path / "eval.py"
    script.write_text("def evaluate(candidate):\n    return candidate.VALUE\n")
    assert (
        to_evaluation(PythonEvaluator.from_file(script)("VALUE = 0.5\n")).score == 0.5
    )


# ------------------------------------------------------------- one-call
def test_optimize_code_runs_evolve_with_cases_and_holdout():
    client = ScriptedClient(
        lambda p: f"Handle even lengths and empty input.\n```candidate\n{FIXED}```"
    )
    res = optimize_code(
        SEED,
        "median(xs) returns the median; raises ValueError on an empty list.",
        function="median",
        cases=CASES,
        holdout_cases={"pair": (([10, 20],), 15.0)},
        model=client,
        budget=5,
    )
    assert res.stopped == "stop_score" and res.best.text == FIXED
    assert res.holdout["seed"].score == 0.0 and res.holdout["best"].score == 1.0
    report = res.report()
    assert "stopped: stop_score after 2 evaluations" in report
    assert (
        "best: c1 1.000 (seed 0.250)" in report
        and "held-out: seed 0.000 -> best 1.000" in report
    )
    with pytest.raises(ValueError, match="exactly one"):
        optimize_code(SEED, "x", model=client)


def _reversing_model(prompt):
    """Proposer: suggests reversing. Task model: follows its instruction."""
    if "You improve a candidate" in prompt:
        return "Answers are reversed.\n```candidate\nRepeat the input reversed.\n```"
    text = prompt.split("<START_OF_USER_PROMPT>")[1].split("<END_OF_USER_PROMPT>")[0]
    return text.strip()[::-1] if "reversed" in prompt else text.strip()


def test_optimize_prompt_splits_examples_and_learns_from_wrong_answers():
    words = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "theta"]
    examples = [(w, w[::-1]) for w in words]
    client = ScriptedClient(_reversing_model)
    res = optimize_prompt("Repeat the input.", examples, model=client, budget=4)
    assert res.seed.score == 0.0 and res.best.score == 1.0
    assert len(res.best.evaluation.instance_scores) == 6  # 25% held out
    assert res.holdout["seed"].score == 0.0 and res.holdout["best"].score == 1.0
    proposer_prompts = [p for p in client.prompts if "You improve a candidate" in p]
    assert len(proposer_prompts) == 1 and "expected 'ahpla'" in proposer_prompts[0]
    assert exact_match(" Billing. ", "billing") == 1.0 and exact_match("a", "b") == 0.0
    with pytest.raises(ValueError, match="holdout"):
        optimize_prompt("x", examples[:1], model=client)


def test_optimize_prompt_asks_to_carry_feedback_facts_unless_turned_off():
    examples = [(w, w[::-1]) for w in ["alpha", "beta", "gamma", "delta"]]
    for carry in (True, False):
        client = ScriptedClient(_reversing_model)
        optimize_prompt(
            "Repeat.", examples, model=client, budget=2, holdout=None, carry_facts=carry
        )
        proposer = [p for p in client.prompts if "You improve a candidate" in p]
        assert proposer and all((CARRY_FACTS.strip() in p) == carry for p in proposer)


def test_optimize_prompt_accumulates_hidden_labels_across_generations():
    """Realistic Banking77-style case: labels appear only in feedback across rounds."""
    import re

    intents = {
        "card": "card_arrival",
        "pin": "pin_blocked",
        "top-up": "top_up_failed",
        "stolen": "lost_or_stolen_card",
        "exchange": "exchange_rate",
        "refund": "Refund_not_showing_up",
    }
    examples = [
        (f"Customer query #{i} about {kw}", label)
        for kw, label in intents.items()
        for i in range(3)
    ]  # 18 examples -> 14 search (>8 shown per round), 4 holdout

    def banking_model(prompt):
        if "You improve a candidate" in prompt:
            parent = prompt.split("## Current candidate\n```\n")[1].split("\n```")[0]
            known = set(re.findall(r"\b[A-Za-z]+_[a-z_]+\b", parent))
            if CARRY_FACTS.strip() in prompt:
                known.update(re.findall(r"expected '([A-Za-z_]+)'", prompt))
            labels_str = ", ".join(sorted(known))
            return (
                "Carry discovered intent names into the prompt.\n"
                f"```candidate\nClassify the banking query. Valid labels: {labels_str}\n```"
            )
        sys_part, user_part = prompt.split("<START_OF_USER_PROMPT>")
        query = user_part.split("<END_OF_USER_PROMPT>")[0].lower()
        for kw, label in intents.items():
            if kw in query and label in sys_part:
                return label
        return "unknown"

    with_carry = optimize_prompt(
        "Classify the banking query.",
        examples,
        model=ScriptedClient(banking_model),
        budget=4,
        carry_facts=True,
    )
    without_carry = optimize_prompt(
        "Classify the banking query.",
        examples,
        model=ScriptedClient(banking_model),
        budget=4,
        carry_facts=False,
    )
    assert with_carry.seed.score == 0.0
    assert with_carry.best.score == 1.0 and with_carry.holdout["best"].score == 1.0
    for label in intents.values():
        assert label in with_carry.best.text
    assert without_carry.best.score == 0.0 and without_carry.holdout["best"].score == 0.0


def test_optimize_code_multi_format_duration_parser_diverse_vs_first():
    """Real subprocess sandbox: default diverse failure sampling exposes all input families."""
    seed_code = "def parse_seconds(s):\n    return -1\n"
    cases = {f"int_{i}": ((str(i * 10),), i * 10) for i in range(1, 10)}
    cases.update(
        {
            "unit_m": (("5m",), 300),
            "iso_hm": (("PT1H30M",), 5400),
            "clock": (("01:02:03",), 3723),
            "empty": (("",), ValueError),
        }
    )
    holdout = {
        "int_h": (("42",), 42),
        "unit_h": (("2h",), 7200),
        "iso_m": (("PT15M",), 900),
        "clock_2": (("00:05:00",), 300),
        "bad": (("abc",), ValueError),
    }

    def proposer_from_shown_failures(prompt):
        fb = prompt.split("## Its evaluation")[1]
        lines = [
            "import re",
            "def parse_seconds(s):",
            "    s = s.strip()",
            "    if not s:",
            "        raise ValueError('empty')",
        ]
        if "PT1H30M" in fb:
            lines += [
                "    m = re.fullmatch(r'PT(?:(\\d+)H)?(?:(\\d+)M)?', s)",
                "    if m and any(m.groups()):",
                "        return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60",
            ]
        if "01:02:03" in fb:
            lines += [
                "    if ':' in s:",
                "        h, m, sec = (int(x) for x in s.split(':'))",
                "        return h * 3600 + m * 60 + sec",
            ]
        if "5m" in fb or "PT1H30M" in fb:
            lines += [
                "    m = re.fullmatch(r'(\\d+)([smh])', s)",
                "    if m:",
                "        return int(m.group(1)) * {'s': 1, 'm': 60, 'h': 3600}[m.group(2)]",
            ]
        lines += [
            "    return int(s)",
        ]
        body = "\n".join(lines) + "\n"
        return f"Handle the shown duration formats.\n```candidate\n{body}```"

    res_diverse = optimize_code(
        seed_code,
        "Parse duration strings into integer seconds; raise ValueError on invalid input.",
        function="parse_seconds",
        cases=cases,
        holdout_cases=holdout,
        model=ScriptedClient(proposer_from_shown_failures),
        budget=2,
    )
    res_first = optimize_code(
        seed_code,
        "Parse duration strings into integer seconds; raise ValueError on invalid input.",
        evaluator=cases_evaluator("parse_seconds", cases, show_order="first"),
        holdout_cases=holdout,
        function="parse_seconds",
        model=ScriptedClient(proposer_from_shown_failures),
        budget=2,
    )
    assert res_diverse.best.score == 1.0 and res_diverse.holdout["best"].score == 1.0
    assert res_first.best.score < 0.8 and res_first.holdout["best"].score == 0.4

