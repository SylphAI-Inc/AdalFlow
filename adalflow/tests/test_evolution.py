"""Tests for LLM-guided evolution: archive, sandbox, proposer and loop.

The loop tests use a scripted model that fixes the program only when the failing
cases appear in its prompt, so they check that evaluator feedback reaches the LLM.
"""

import json
import os
import sys
from random import Random

import pytest

from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput
from adalflow.optim.evolution import (
    BEST,
    MAP_ELITES,
    Archive,
    Candidate,
    Evaluation,
    LLMProposer,
    Proposal,
    ProposalError,
    RESAMPLE,
    PythonEvaluator,
    combine,
    evolve,
    extract_candidate,
    run_python,
    to_evaluation,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")

SEED = "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"
AVERAGED = (
    "def median(xs):\n"
    "    s = sorted(xs)\n"
    "    m = len(s) // 2\n"
    "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"
)
FIXED = (
    "def median(xs):\n"
    "    if not xs:\n"
    "        raise ValueError('empty')\n" + AVERAGED.split("\n", 1)[1]
)
EVALUATOR = """
CASES = {"odd": ([3, 1, 2], 2), "even": ([4, 1, 3, 2], 2.5), "pair": ([1, 2], 1.5)}

def evaluate(candidate):
    scores, notes = {}, []
    for name, (xs, want) in CASES.items():
        try:
            got = candidate.median(list(xs))
        except Exception as exc:
            got = repr(exc)
        scores[name] = 1.0 if got == want else 0.0
        if got != want:
            notes.append(f"{name}: median({xs}) expected {want}, got {got}")
    try:
        candidate.median([])
        scores["empty"] = 0.0
        notes.append("empty: median([]) should raise ValueError, returned a value")
    except ValueError:
        scores["empty"] = 1.0
    except Exception as exc:
        scores["empty"] = 0.0
        notes.append(f"empty: median([]) should raise ValueError, raised {exc!r}")
    return {"instance_scores": scores, "feedback": "\\n".join(notes)}
"""


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
        return self.reply(api_kwargs["prompt"], len(self.prompts))

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data=None, raw_response=completion)


def fixing_reply(prompt, n):
    """Fixes exactly what the feedback in the prompt reports as failing."""
    if "even: median" in prompt:
        return f"Average the two middle values.\n```candidate\n{AVERAGED}```"
    if "empty: median" in prompt:
        return f"Raise on empty input.\n```candidate\n{FIXED}```"
    return f"Nothing left to fix.\n```candidate\n{FIXED}# unchanged\n```"


def _proposer(reply=fixing_reply, **kw):
    client = ScriptedClient(reply)
    proposer = LLMProposer(
        "Return the median; raise ValueError on empty input.",
        model_client=client,
        model_kwargs={"model": "scripted"},
        **kw,
    )
    return proposer, client


def _cand(cid, **instances):
    return Candidate(
        id=cid,
        text=cid,
        evaluation=Evaluation(
            score=sum(instances.values()) / len(instances), instance_scores=instances
        ),
    )


# --------------------------------------------------------------------- types
def test_to_evaluation_normalizes_evaluator_returns():
    assert to_evaluation(0.5).score == 0.5
    ev = to_evaluation({"instance_scores": {"a": 1, "b": 0}, "tokens": 7})
    assert ev.score == 0.5 and ev.metrics == {"tokens": 7}
    assert not to_evaluation({"score": 1.0, "error": "boom"}).ok
    with pytest.raises(TypeError):
        to_evaluation(True)
    with pytest.raises(TypeError):
        to_evaluation("0.9")


# ------------------------------------------------------------------- archive
def test_pareto_front_keeps_specialists_and_drops_dominated():
    archive = Archive()
    for c in (
        _cand("generalist", a=0.8, b=0.8, c=0.0),
        _cand("specialist", a=0.0, b=0.0, c=1.0),
        _cand("dominated", a=0.7, b=0.8, c=0.0),
        _cand("twin", a=0.8, b=0.8, c=0.0),
    ):
        archive.add(c)
    assert [c.id for c in archive.pareto_front()] == ["generalist", "specialist"]
    assert archive.best().id == "generalist"
    picks = {archive.select_parent(Random(s)).id for s in range(30)}
    assert picks == {"generalist", "specialist"}
    assert archive.select_parent(Random(0), BEST).id == "generalist"


def test_unscored_archive_returns_latest_candidate_to_fix():
    archive = Archive()
    archive.add(Candidate(id="c0", text="x", evaluation=Evaluation(error="crash")))
    assert archive.best() is None
    assert archive.select_parent(Random(0)).id == "c0"
    assert archive.select_inspiration(Random(0), archive.candidates[0]) is None


# ------------------------------------------------------------------- sandbox
def test_run_python_hides_parent_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    run = run_python(
        "import os, sys\nprint(os.environ.get('OPENAI_API_KEY'), os.getcwd() == os.environ['HOME'])"
    )
    assert run.returncode == 0 and run.stdout.split() == ["None", "True"]


@posix_only
def test_run_python_kills_on_timeout():
    run = run_python(
        "import time\nprint('start', flush=True)\ntime.sleep(30)", timeout=1
    )
    assert run.timed_out and run.returncode is None and "start" in run.stdout
    assert run.duration_s < 10


def test_python_evaluator_scores_instances_and_reports_failures():
    ev = PythonEvaluator(EVALUATOR)(SEED)
    assert ev.ok and ev.instance_scores == {
        "odd": 1.0,
        "even": 0.0,
        "pair": 0.0,
        "empty": 0.0,
    }
    assert "expected 2.5, got 3" in ev.feedback and "IndexError" in ev.feedback
    assert PythonEvaluator(EVALUATOR)(FIXED).score == 1.0


def test_python_evaluator_records_crashes_and_timeouts():
    syntax = PythonEvaluator(EVALUATOR)("def median(xs) return 1\n")
    assert not syntax.ok and "importing" in syntax.error
    assert "SyntaxError" in syntax.feedback
    hang = PythonEvaluator(EVALUATOR, timeout=1)(
        "import time\ndef median(xs):\n    time.sleep(30)\n"
    )
    assert not hang.ok and "timed out" in hang.error


def test_python_evaluator_ignores_spoofed_result_lines():
    spoof = (
        'print("\\n__ADALFLOW_EVAL__" + \'{"stage": "ok", "result": 1.0}\')\n'
        "def median(xs):\n    return 0\n"
    )
    assert PythonEvaluator(EVALUATOR)(spoof).score == 0.0


# ------------------------------------------------------------------ proposer
def test_extract_candidate_prefers_the_tagged_block():
    reply = "Plan.\n```python\nold\n```\nSo:\n```candidate\nnew\n```"
    out = extract_candidate(reply)
    assert out.text == "new\n" and out.rationale.endswith("So:")
    assert extract_candidate("```\nonly\n```").text == "only\n"
    with pytest.raises(ProposalError):
        extract_candidate("no code here")


def test_proposer_sees_feedback_and_retries_bad_replies():
    replies = iter(
        ["no block", f"```candidate\n{SEED}```", f"```candidate\n{FIXED}```"]
    )
    proposer, client = _proposer(lambda p, n: next(replies), max_retries=2)
    parent = Candidate(id="c0", text=SEED, evaluation=PythonEvaluator(EVALUATOR)(SEED))
    out = proposer.propose(parent)
    assert out.text == FIXED
    assert "expected 2.5, got 3" in client.prompts[0]
    assert "- even: 0" in client.prompts[0]  # failing instances listed
    assert "no fenced" in client.prompts[1] and "identical" in client.prompts[2]


def test_proposer_gives_up_after_retries():
    proposer, _ = _proposer(lambda p, n: "still no block", max_retries=1)
    with pytest.raises(ProposalError):
        proposer.propose(Candidate(id="c0", text=SEED))


# ---------------------------------------------------------------------- loop
def test_evolve_fixes_the_program_from_feedback(tmp_path):
    proposer, client = _proposer()
    log_path = tmp_path / "run.jsonl"
    res = evolve(
        SEED,
        PythonEvaluator(EVALUATOR),
        proposer,
        max_evaluations=6,
        stop_score=1.0,
        log_path=log_path,
    )
    assert res.seed.score == 0.25 and res.best.score == 1.0 and res.improved
    assert res.stopped == "stop_score" and res.evaluations == 3
    assert [c.parent_id for c in res.archive.candidates] == [None, "c0", "c1"]
    assert len(client.prompts) == 2
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["candidate"] * 3
    assert events[1]["rationale"] == "Average the two middle values."


def test_evolve_counts_duplicates_and_stalls():
    # After one real improvement the model keeps re-proposing the seed.
    proposer, _ = _proposer(
        lambda p, n: f"```candidate\n{AVERAGED if n == 1 else SEED}```"
    )
    res = evolve(
        SEED, PythonEvaluator(EVALUATOR), proposer, max_evaluations=10, max_stalls=3
    )
    assert res.stopped == "stalled" and res.evaluations == 2
    assert res.duplicates == 3 and res.proposal_failures == 0


def test_evolve_records_evaluator_and_proposer_failures():
    def broken(text):
        raise RuntimeError("evaluator down")

    proposer, _ = _proposer(lambda p, n: "nope", max_retries=0)
    res = evolve(SEED, broken, proposer, max_evaluations=3, max_stalls=2)
    assert not res.seed.ok and "evaluator down" in res.seed.evaluation.error
    assert res.best is None and res.proposal_failures == 2 and not res.improved
    with pytest.raises(ValueError):
        evolve(SEED, broken, proposer, max_evaluations=0)


def test_resample_baseline_always_starts_from_the_seed():
    # Without building on c1, the seed's feedback only ever leads to AVERAGED.
    proposer, client = _proposer()
    res = evolve(
        SEED,
        PythonEvaluator(EVALUATOR),
        proposer,
        max_evaluations=6,
        selection=RESAMPLE,
        max_stalls=2,
    )
    assert [c.parent_id for c in res.archive.candidates] == [None, "c0"]
    assert res.best.text == AVERAGED and res.best.score == 0.75
    assert res.stopped == "stalled" and res.duplicates == 2
    assert all("Inspiration" not in p for p in client.prompts)
    with pytest.raises(ValueError):
        Archive().select_parent(Random(0), "nope")


def test_holdout_scores_seed_and_best_outside_the_budget(tmp_path):
    def holdout(text):
        return 1.0 if "ValueError" in text else 0.5

    proposer, _ = _proposer()
    log_path = tmp_path / "run.jsonl"
    res = evolve(
        SEED,
        PythonEvaluator(EVALUATOR),
        proposer,
        max_evaluations=6,
        stop_score=1.0,
        log_path=log_path,
        holdout=holdout,
    )
    assert res.evaluations == 3
    assert res.holdout["seed"].score == 0.5 and res.holdout["best"].score == 1.0
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [(e["role"], e["id"]) for e in events if e["event"] == "holdout"] == [
        ("seed", "c0"),
        ("best", "c2"),
    ]


def test_sandbox_uses_the_current_interpreter():
    run = run_python("import sys\nprint(sys.version_info[:2])")
    assert run.stdout.strip() == str(tuple(sys.version_info[:2]))


# ------------------------------------------------- noise and diversity
class ListProposer:
    """Proposes the given texts in order and remembers what it was shown."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def propose(self, parent, inspiration=None):
        self.calls.append((parent.id, inspiration.id if inspiration else None))
        return Proposal(text=self.texts.pop(0), rationale="scripted")


def noisy(table):
    """An evaluator that returns each text's scores in turn (then repeats the last)."""
    seen = {}

    def evaluate(text):
        i = seen[text] = seen.get(text, -1) + 1
        scores = table[text]
        return scores[min(i, len(scores) - 1)]

    return evaluate


def test_combine_averages_samples_and_skips_failures():
    a = Evaluation(score=0.2, instance_scores={"x": 0.0, "y": 0.4}, feedback="a")
    b = Evaluation(score=0.6, instance_scores={"x": 1.0, "y": 0.2}, feedback="b")
    out = combine([a, Evaluation(error="timeout"), b])
    assert out.score == pytest.approx(0.4) and out.feedback == "b"
    assert out.instance_scores == pytest.approx({"x": 0.5, "y": 0.3})
    assert out.metrics["samples"] == 2 and out.metrics["score_sd"] > 0
    assert combine([a]) is a and not combine([Evaluation(error="x")]).ok
    with pytest.raises(ValueError):
        combine([])


def test_confirm_rejects_a_lucky_first_score():
    table = {"seed": [0.5], "lucky": [0.9, 0.1, 0.1], "good": [0.7, 0.7, 0.7]}
    plain = evolve("seed", noisy(table), ListProposer(["lucky"]), max_evaluations=2)
    assert plain.best.text == "lucky"

    res = evolve(
        "seed",
        noisy(table),
        ListProposer(["lucky", "good"]),
        max_evaluations=20,
        confirm=2,
        max_stalls=1,
    )
    by_text = {c.text: c for c in res.archive.candidates}
    assert by_text["lucky"].n_samples == 3 and by_text["seed"].n_samples == 3
    assert by_text["lucky"].score == pytest.approx(1.1 / 3)
    assert res.best.text == "good" and res.best.n_samples == 3
    # seed 3 + lucky 3 + good 3 samples; the run stops when proposals run out.
    assert res.evaluations == 9 and res.stopped == "stalled"
    assert sum(e["event"] == "sample" for e in res.events) == 6


def test_confirm_reserves_budget_to_finish_each_comparison():
    table = {"seed": [0.5], "lucky": [0.9, 0.1]}
    # After the seed, a proposal needs 3 samples of it and 2 more of the seed.
    short = evolve(
        "seed", noisy(table), ListProposer(["lucky"]), max_evaluations=5, confirm=2
    )
    assert short.evaluations == 1 and short.stopped == "budget"
    res = evolve(
        "seed", noisy(table), ListProposer(["lucky"]), max_evaluations=6, confirm=2
    )
    assert res.evaluations == 6 and res.best.text == "seed"
    assert [c.n_samples for c in res.archive.candidates] == [3, 3]


def test_islands_rotate_and_migrate():
    texts = [f"t{i}" for i in range(6)]
    table = {"seed": [0.1], **{t: [0.2 + 0.1 * i] for i, t in enumerate(texts)}}
    proposer = ListProposer(texts)
    res = evolve(
        "seed",
        noisy(table),
        proposer,
        max_evaluations=7,
        selection=BEST,
        islands=2,
        migrate_every=4,
    )
    assert [c.island for c in res.archive.candidates[1:]] == [0, 1, 0, 1, 0, 1]
    parents = [p for p, _ in proposer.calls]
    # Island 0 builds on c1, c3, ...; island 1 on c2, c4, ...; both start from c0.
    assert parents == ["c0", "c0", "c1", "c2", "c3", "c4"]
    migrations = [e for e in res.events if e["event"] == "migration"]
    assert [(m["from"], m["to"], m["id"]) for m in migrations] == [(0, 1, "c3")]
    assert proposer.calls[3] == ("c2", "c3")
    assert res.best.text == "t5"


def test_map_elites_keeps_and_selects_every_cell():
    with pytest.raises(ValueError, match="descriptor"):
        evolve("seed", noisy({"seed": [0.1]}), ListProposer([]), selection=MAP_ELITES)

    archive = Archive()
    for cid, score, cell in [
        ("c0", 0.1, ("a",)),
        ("c1", 0.9, ("a",)),
        ("c2", 0.3, ("b",)),
        ("c3", 0.2, ("b",)),
        ("c4", 0.95, None),
    ]:
        c = Candidate(id=cid, text=cid, evaluation=Evaluation(score=score))
        c.cell = cell
        archive.add(c)
    elites = archive.elites()
    assert {k: v.id for k, v in elites.items()} == {("a",): "c1", ("b",): "c2"}
    rng = Random(0)
    picks = {archive.select_parent(rng, MAP_ELITES).id for _ in range(30)}
    assert picks == {"c1", "c2"}  # the weaker cell's elite is still improved
    assert (
        archive.select_inspiration(Random(0), elites[("a",)], strategy=MAP_ELITES).id
        == "c2"
    )


def test_descriptor_sets_cells_and_tolerates_errors():
    def by_length(c):
        if c.text == "boom":
            raise RuntimeError("bad descriptor")
        return (len(c.text) // 3,)

    table = {"seed": [0.1], "abcdef": [0.5], "boom": [0.2]}
    res = evolve(
        "seed",
        noisy(table),
        ListProposer(["abcdef", "boom"]),
        max_evaluations=3,
        selection=MAP_ELITES,
        descriptor=by_length,
    )
    assert [c.cell for c in res.archive.candidates] == [(1,), (2,), None]
