import pytest

from adalflow.core.model_client import ModelClient
from adalflow.core.types import Function, FunctionOutput, GeneratorOutput
from adalflow.optim.harness import (
    BudgetError,
    Harness,
    HarnessBudgetLedger,
    HarnessSkill,
    HarnessValidationError,
    RunnerExecutor,
    render_trajectory,
)


class NoCallClient(ModelClient):
    def call(self, api_kwargs, model_type):
        raise RuntimeError("tests make no model calls")


class FinalFunction(Function):
    def __init__(self, answer):
        super().__init__(name="finish", kwargs={})
        self._is_answer_final = True
        self._answer = answer


class ScriptedPlanner:
    """Returns scripted planner outputs, like ``tests/test_runner.py``'s FakePlanner."""

    def __init__(self, outputs):
        self._outputs = [GeneratorOutput(data=o) for o in outputs]

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        return self._outputs.pop(0)

    def get_prompt(self, **kwargs):
        return ""


class EchoToolManager:
    def execute_func(self, func):
        return FunctionOutput(name=func.name, input=func, output=f"ran {func.name}")


class ScriptedAgent:
    def __init__(self, outputs):
        self.planner = ScriptedPlanner(outputs)
        self.tool_manager = EchoToolManager()
        self.max_steps = 5
        self.answer_data_type = str


def _harness(**kwargs):
    base = dict(
        system_prompt="You audit ledgers.",
        guardrails="Never drop credit rows.",
        skills={
            "signed-totals": HarnessSkill("Totals with credits", "Sum signed amounts.")
        },
    )
    base.update(kwargs)
    return Harness(**base)


def test_render_role_desc_omits_empty_sections_and_indexes_skills():
    text = _harness().render_role_desc()
    assert text.startswith("You audit ledgers.")
    assert "## Guardrails\nNever drop credit rows." in text
    assert "## Memory" not in text and "## Tool guidance" not in text
    assert "- signed-totals: Totals with credits" in text
    assert "Sum signed amounts." not in text  # skill bodies load on demand
    assert Harness().render_role_desc() == ""


def test_validate_reports_every_problem():
    bad = _harness(
        skills={
            "Bad Name": HarnessSkill("ok", "body"),
            "no-desc": HarnessSkill("", "body"),
            "empty": HarnessSkill("d", "  "),
        }
    )
    with pytest.raises(HarnessValidationError) as info:
        bad.validate()
    assert len(info.value.errors) == 3

    with pytest.raises(HarnessValidationError, match="cap"):
        _harness(system_prompt="x" * 50).validate(always_on_cap=40)
    _harness().validate()


def test_from_dict_rejects_malformed_skills_and_ignores_unknown_keys():
    with pytest.raises(HarnessValidationError):
        Harness.from_dict({"skills": ["not", "a", "dict"]})
    with pytest.raises(HarnessValidationError):
        Harness.from_dict({"skills": {"a": "just text"}})
    h = Harness.from_dict({"system_prompt": "hi", "scripts": {"x.py": "print()"}})
    assert h.system_prompt == "hi" and not h.skills


def test_diff_and_roundtrip(tmp_path):
    base = _harness()
    new = base.copy()
    new.memory = "Credits are negative amounts in this workspace."
    new.guardrails = "Never drop credit rows. Recompute totals before answering."
    del new.skills["signed-totals"]
    new.skills["csv-check"] = HarnessSkill("Check CSV headers", "Read the header row.")

    diff = base.diff(new)
    assert diff.added == ["memory", "skills/csv-check"]
    assert diff.removed == ["skills/signed-totals"]
    assert diff.modified == ["guardrails"]
    assert base.diff(base.copy()).is_empty

    path = new.save(tmp_path / "h.json")
    assert Harness.load(path) == new


def test_to_agent_renders_prompt_and_registers_load_skill():
    agent = _harness(system_prompt="Literal {{ not_a_var }} stays.").to_agent(
        model_client=NoCallClient(), model_kwargs={"model": "none"}
    )
    prompt = agent.planner.get_prompt(input_str="sum the ledger")
    assert "Literal {{ not_a_var }} stays." in prompt
    assert "signed-totals" in prompt

    names = [t.definition.func_name for t in agent.tool_manager.tools]
    assert "load_skill" in names
    out = agent.tool_manager.execute_func(
        Function(name="load_skill", kwargs={"name": "signed-totals"})
    )
    assert out.output == "Sum signed amounts."


def test_render_trajectory_formats_and_truncates():
    class Step:
        def __init__(self, function, observation):
            self.function, self.observation = function, observation

    fn = Function(name="read_csv", kwargs={"path": "ledger.csv"}, thought="look first")
    text = render_trajectory([Step(fn, "x" * 50)], answer="105.25", obs_trunc=10)
    lines = text.splitlines()
    assert lines[0] == "[THOUGHT 1] look first"
    assert lines[1].startswith(
        '[ACTION 1] read_csv({"args": [], "path": "ledger.csv"})'
    )
    assert lines[2].startswith("[OBSERVATION 1] xxxxxxxxxx ... [40 chars truncated]")
    assert lines[3] == "[ANSWER] 105.25"
    assert render_trajectory([]) == "(no steps recorded)"


def test_runner_executor_records_steps_and_answer():
    outputs = [Function(name="read_csv", kwargs={"path": "a.csv"}), FinalFunction("42")]
    executor = RunnerExecutor(agent_factory=lambda h: ScriptedAgent(list(outputs)))
    result = executor.execute(_harness(), "sum it", "base1")
    assert result.ok and result.delivered
    assert result.answer == "42"
    assert result.steps == 1
    assert "[ACTION 1] read_csv" in result.trajectory
    assert "[OBSERVATION 1] ran read_csv" in result.trajectory


def test_runner_executor_never_raises():
    def broken_factory(harness):
        raise RuntimeError("model unavailable")

    result = RunnerExecutor(agent_factory=broken_factory).execute(
        _harness(), "t", "inv1"
    )
    assert not result.ok and not result.delivered
    assert result.error == "RuntimeError: model unavailable"
    assert "[ERROR] RuntimeError: model unavailable" in result.trajectory


def _full_ledger(k=2):
    ledger = HarnessBudgetLedger(mode="parallel", task_id="t1", k=k)
    for i in range(1, k + 1):
        ledger.rollout(f"base{i}")
    for i in range(1, k + 1):
        for dim in ("inv", "dom"):
            ledger.session(f"{dim}{i}")
            ledger.rollout(f"{dim}{i}")
    for i in range(1, k + 1):
        ledger.session(f"agg{i}", attempts=2, fell_back=i == 2)
        ledger.rollout(f"agg{i}")
    return ledger


def test_ledger_identity_holds_for_a_complete_step():
    ledger = _full_ledger()
    ledger.assert_identity()
    d = ledger.to_dict()
    assert (d["R"], d["P"], d["session_attempts"]) == (8, 6, 8)
    assert d["fell_back"] == ["agg2"]


def test_ledger_catches_missing_and_duplicate_stages():
    missing = _full_ledger()
    missing.entries.pop()
    with pytest.raises(BudgetError, match="R=7, expected 8"):
        missing.assert_identity()

    # Swapping one label for a duplicate keeps the counts right but is still caught.
    dup = _full_ledger()
    dup.entries[-1].label = "agg1"
    with pytest.raises(BudgetError, match="duplicate rollout labels"):
        dup.assert_identity()
