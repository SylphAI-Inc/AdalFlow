import json

import pytest

from adalflow.optim.harness import (
    PARALLEL,
    SEQUENTIAL,
    Experience,
    ExperienceLibrary,
    ExperienceReflector,
    Harness,
    LibraryEntry,
    RoleOutputError,
    run_domain,
)
from tests.test_harness_refine import (
    BASE,
    LedgerExecutor,
    ScriptedClient,
    _roles,
)

LESSON = {
    "cue": "ledger totals with negative rows",
    "lesson": "Keep signed amounts; state it in guardrails and a signed-totals skill.",
    "evidence": "guardrails row: helped (agg1 kept -30.5; base1 dropped it)",
    "confidence": "medium",
}
EDITS = [
    {
        "component": "guardrails",
        "candidates": "inv1, agg1",
        "change": "keep signed amounts",
        "observed": "credits kept",
        "verdict": "helped",
    }
]


def _reflector(reply):
    client = ScriptedClient(reply)
    return ExperienceReflector(model_client=client, model_kwargs={"model": "s"}), client


def _good_reflection(prompt, n):
    return json.dumps({"edits": EDITS, "experience": LESSON})


def test_reflector_parses_and_validates():
    reflector, _ = _reflector(_good_reflection)
    out = reflector.reflect("evidence")
    assert out.edits[0].verdict == "helped" and out.experience.is_supported

    none, _ = _reflector(lambda p, n: '{"edits": [], "experience": null}')
    assert none.reflect("e").experience is None

    bad, _ = _reflector(
        lambda p, n: json.dumps({"edits": [dict(EDITS[0], verdict="great")]})
    )
    with pytest.raises(RoleOutputError, match="verdict"):
        bad.reflect("e")


def _entry(task_id, **exp):
    return LibraryEntry(
        task_id=task_id,
        domain="accounting",
        harness=BASE,
        experience=Experience(**dict(LESSON, **exp)),
    )


def test_library_stores_only_supported_lessons_in_order(tmp_path):
    lib = ExperienceLibrary(tmp_path, mode=PARALLEL)
    assert lib.render("accounting") == ""
    assert not lib.add(_entry("t0", evidence="  "))
    assert lib.add(_entry("t2")) and lib.add(_entry("t1"))
    assert [e.task_id for e in lib.entries("accounting")] == ["t2", "t1"]
    assert [e.task_id for e in lib.entries("accounting", only=["t1"])] == ["t1"]
    assert "### From task t2" in lib.render("accounting")

    # Libraries are separate per mode.
    assert ExperienceLibrary(tmp_path, mode=SEQUENTIAL).entries("accounting") == []


def test_run_domain_shows_only_earlier_lessons(tmp_path):
    editor, selector, ed_client, _ = _roles()
    reflector, refl_client = _reflector(_good_reflection)
    lib = ExperienceLibrary(tmp_path)
    tasks = [("ledger-1", "Total ledger one"), ("ledger-2", "Total ledger two")]
    executor = LedgerExecutor()

    runs = run_domain(
        tasks, "accounting", BASE, executor, editor, reflector, lib, selector, k=1
    )

    assert [r.stored for r in runs] == [True, True]
    first_task_prompts, second_task_prompts = (
        ed_client.prompts[:3],
        ed_client.prompts[3:],
    )
    assert all("Experience from earlier tasks" not in p for p in first_task_prompts)
    assert all("From task ledger-1" in p for p in second_task_prompts)
    assert all("From task ledger-2" not in p for p in second_task_prompts)
    # Every task starts from the base harness.
    base_runs = [r.result.base_runs[0].harness for r in runs]
    assert base_runs == [BASE, BASE]
    # The reflector sees the adoption and every candidate.
    assert "Adopted final: agg1" in refl_client.prompts[0]
    assert "### Candidate inv1" in refl_client.prompts[0]
    stored = lib.entries("accounting")[0]
    assert stored.chosen_label == "agg1" and stored.harness != BASE


def test_include_existing_controls_visibility_of_older_entries(tmp_path):
    lib = ExperienceLibrary(tmp_path)
    lib.add(_entry("old-task"))
    for include, expected in ((False, False), (True, True)):
        editor, _, ed_client, _ = _roles()
        reflector, _ = _reflector(_good_reflection)
        run_domain(
            [("new", "Total it")],
            "accounting",
            BASE,
            LedgerExecutor(),
            editor,
            reflector,
            lib,
            include_existing=include,
        )
        assert ("From task old-task" in ed_client.prompts[0]) is expected


def test_failed_reflection_stores_nothing(tmp_path):
    editor, _, _, _ = _roles()
    reflector, refl_client = _reflector(lambda p, n: "no json here")
    lib = ExperienceLibrary(tmp_path)
    (run,) = run_domain(
        [("t", "Total it")],
        "accounting",
        BASE,
        LedgerExecutor(),
        editor,
        reflector,
        lib,
    )
    assert run.result.status == "ok"
    assert not run.stored and "no JSON" in run.reflect_error
    assert len(refl_client.prompts) == 2
    assert lib.entries("accounting") == []


def test_no_reflection_when_refinement_selects_nothing(tmp_path):
    editor, _, _, _ = _roles()
    reflector, refl_client = _reflector(_good_reflection)
    (run,) = run_domain(
        [("t", "Total it")],
        "accounting",
        BASE,
        LedgerExecutor(fail=True),
        editor,
        reflector,
        ExperienceLibrary(tmp_path),
    )
    assert run.result.status == "base-failed" and refl_client.prompts == []
    assert isinstance(run.result.best_harness, Harness)
