"""Tests for the evolution shortcuts and model specs (scripted models, real sandbox)."""

import pytest

from adalflow.components.model_client import resolve_model
from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput
from adalflow.optim.evolution import (
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
