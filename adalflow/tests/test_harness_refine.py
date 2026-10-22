"""Deterministic tests for label-free harness refinement.

Scenario (after the Harness² local demo): the agent totals a ledger that contains
credits. The base harness says nothing about signs, so the agent drops credit rows.
The scripted editor fixes that; the scripted selector checks each deliverable against
the source rows, never against an answer key.
"""

import json
import re

import pytest

from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput
from adalflow.optim.harness import (
    PARALLEL,
    SEQUENTIAL,
    BlindSelector,
    Harness,
    HarnessEditor,
    HarnessSkill,
    ProbeResult,
    refine_harness,
)

ROWS = [120.0, -30.5, 40.0, -24.25]
TASK = "Total the ledger amounts: " + ", ".join(str(r) for r in ROWS)
BASE = Harness(system_prompt="You total ledgers.")
FIXED = {
    "system_prompt": "You total ledgers.",
    "guardrails": "Keep signed amounts: credits are negative rows, never drop them.",
    "skills": {
        "signed-totals": {
            "description": "Totals over ledgers that contain credits",
            "body": "Add every row with its sign, then re-add to check.",
        }
    },
}


class ScriptedClient(ModelClient):
    """A ModelClient whose reply is a function of the rendered prompt."""

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


class LedgerExecutor:
    """Runs a 'harness': credits are kept only if the harness says to keep signs."""

    def __init__(self, fail=False):
        self.fail = fail
        self.labels = []

    def execute(self, harness, task, label):
        self.labels.append(label)
        if self.fail:
            return ProbeResult(label=label, harness=harness, error="model unavailable")
        signed = "signed" in harness.render_role_desc().lower()
        rows = ROWS if signed else [r for r in ROWS if r > 0]
        total = sum(rows)
        steps = "\n".join(f"[ACTION] add({r})" for r in rows)
        return ProbeResult(
            label=label,
            harness=harness,
            answer=f"Total: {total:.2f}",
            trajectory=f"{steps}\n[ANSWER] Total: {total:.2f}",
            steps=len(rows),
        )


def fixing_editor_reply(prompt, n):
    return json.dumps(
        {"harness": FIXED, "notes": "keep credits -> base runs drop -30.5"}
    )


def checking_selector_reply(prompt, n):
    """Picks the deliverable whose total matches a re-add of the task's rows."""
    expected = f"{sum(ROWS):.2f}"
    for ident, body in re.findall(
        r"## Deliverable (\w)\n(.*?)(?=\n## |\Z)", prompt, re.S
    ):
        if expected in body:
            return json.dumps(
                {"why": "matches a re-add of the source rows", "choice": ident}
            )
    return json.dumps({"why": "none match", "choice": "A"})


def _roles(editor_reply=fixing_editor_reply, selector_reply=checking_selector_reply):
    ed_client, sel_client = ScriptedClient(editor_reply), ScriptedClient(selector_reply)
    editor = HarnessEditor(model_client=ed_client, model_kwargs={"model": "scripted"})
    selector = BlindSelector(
        model_client=sel_client, model_kwargs={"model": "scripted"}
    )
    return editor, selector, ed_client, sel_client


@pytest.mark.parametrize("mode", [PARALLEL, SEQUENTIAL])
def test_refinement_fixes_the_harness_without_labels(mode):
    editor, selector, ed_client, sel_client = _roles()
    executor = LedgerExecutor()
    res = refine_harness(TASK, BASE, executor, editor, selector, k=2, mode=mode)

    assert res.status == "ok"
    assert "Total: 60.25" not in [r.answer for r in res.base_runs]  # base drops credits
    assert res.selected.answer == f"Total: {sum(ROWS):.2f}"
    assert "signed-totals" in res.best_harness.skills
    # Budget identity: R = 4k rollouts, P = 3k editor sessions, no duplicates.
    assert res.ledger.rollouts == 8 and res.ledger.sessions == 6
    final_labels = ["agg1", "agg2"] if mode == PARALLEL else ["c1", "c2"]
    assert [f.label for f in res.finals] == final_labels
    assert len(ed_client.prompts) == 6


def test_parallel_order_and_evidence():
    editor, selector, ed_client, _ = _roles()
    executor = LedgerExecutor()
    res = refine_harness(TASK, BASE, executor, editor, selector, k=2, mode=PARALLEL)
    assert executor.labels == [
        "base1",
        "base2",
        "inv1",
        "dom1",
        "inv2",
        "dom2",
        "agg1",
        "agg2",
    ]
    assert [s.role for s in res.stages[:2]] == ["procedure", "domain"]
    # Proposals see the base runs; compositions also see every candidate.
    assert "### Run base2" in ed_client.prompts[0]
    assert "### Candidate" not in ed_client.prompts[0]
    compose_prompt = ed_client.prompts[4]
    for label in ("inv1", "dom1", "inv2", "dom2"):
        assert f"### Candidate {label}" in compose_prompt
    assert (
        "Improve HOW" in ed_client.prompts[0] and "Improve WHAT" in ed_client.prompts[1]
    )


def test_sequential_carries_the_checkpoint_forward():
    def reply(prompt, n):
        # Round 1 composes the fix; round 2's composition fails twice and falls back.
        if "compose a refined harness" in prompt and "### Candidate c1" in prompt:
            return "not json"
        return fixing_editor_reply(prompt, n)

    editor, selector, ed_client, _ = _roles(editor_reply=reply)
    res = refine_harness(
        TASK, BASE, LedgerExecutor(), editor, selector, k=2, mode=SEQUENTIAL
    )
    c2 = next(s for s in res.stages if s.label == "c2")
    assert c2.fell_back and c2.attempts == 2
    # The fallback keeps the current checkpoint (c1), not the original base.
    assert res.finals[1].harness == res.finals[0].harness
    assert res.finals[1].harness != BASE
    assert "Earlier rounds" in ed_client.prompts[-1]


def test_invalid_output_is_retried_with_feedback():
    def reply(prompt, n):
        if n == 1:
            return json.dumps(
                {"harness": {"skills": {"Bad Name": {"description": "d", "body": "b"}}}}
            )
        return fixing_editor_reply(prompt, n)

    editor, _, ed_client, _ = _roles(editor_reply=reply)
    res = refine_harness(TASK, BASE, LedgerExecutor(), editor, k=1)
    assert res.stages[0].attempts == 2 and not res.stages[0].fell_back
    assert "rejected" in ed_client.prompts[1] and "Bad Name" in ed_client.prompts[1]
    assert res.ledger.to_dict()["session_attempts"] == 4


def test_keep_reply_keeps_the_current_harness():
    editor, _, _, _ = _roles(
        editor_reply=lambda p, n: '{"keep": true, "notes": "base is fine"}'
    )
    res = refine_harness(TASK, BASE, LedgerExecutor(), editor, k=1)
    assert all(s.kept_base and not s.fell_back for s in res.stages)
    assert res.best_harness == BASE
    assert res.selection.trivial  # k=1: one delivered final, no selector call


def test_selector_sees_only_anonymous_deliverables_and_falls_back():
    editor, selector, _, sel_client = _roles(
        selector_reply=lambda p, n: '{"choice": "Z"}'
    )
    res = refine_harness(TASK, BASE, LedgerExecutor(), editor, selector, k=2)
    prompt = sel_client.prompts[0]
    assert "## Deliverable A" in prompt and "## Deliverable B" in prompt
    assert "Trajectory" not in prompt and "signed-totals" not in prompt
    assert "agg1" not in prompt
    assert len(sel_client.prompts) == 2  # one retry after the invalid choice
    assert res.selection.fallback and res.selected.label == "agg1"


def test_base_failure_stops_before_any_editing():
    editor, selector, ed_client, _ = _roles()
    res = refine_harness(TASK, BASE, LedgerExecutor(fail=True), editor, selector, k=2)
    assert res.status == "base-failed"
    assert ed_client.prompts == [] and res.best_harness == BASE


def test_argument_checks():
    editor, _, _, _ = _roles()
    with pytest.raises(ValueError, match="selector"):
        refine_harness(TASK, BASE, LedgerExecutor(), editor, k=2)
    with pytest.raises(ValueError, match="mode"):
        refine_harness(TASK, BASE, LedgerExecutor(), editor, mode="greedy")
    bad_base = Harness(skills={"x": HarnessSkill("", "")})
    with pytest.raises(ValueError):
        refine_harness(TASK, bad_base, LedgerExecutor(), editor)
