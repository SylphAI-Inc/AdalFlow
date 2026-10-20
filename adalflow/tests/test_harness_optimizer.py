import pytest

from adalflow.components.agent import Agent
from adalflow.optim.harness import (
    PARALLEL,
    SEQUENTIAL,
    ExperienceLibrary,
    Harness,
    HarnessOptimizer,
    HarnessSkill,
)
from tests.test_harness_library import _good_reflection
from tests.test_harness_refine import (
    ROWS,
    TASK,
    LedgerExecutor,
    ScriptedClient,
    checking_selector_reply,
    fixing_editor_reply,
)

FULL = Harness(
    system_prompt="You answer billing questions.",
    guardrails="Never quote a refund amount you did not look up.",
    tool_guidance="Use search before answering.",
)


def search(query: str) -> str:
    """Search the billing knowledge base."""
    return "no results"


def _one_model(prompt, n):
    """One scripted model for every role, dispatched on the role's prompt."""
    if "## Deliverable A" in prompt:
        return checking_selector_reply(prompt, n)
    if "harness-refinement step" in prompt:
        return _good_reflection(prompt, n)
    return fixing_editor_reply(prompt, n)


def _optimizer(**kw):
    client = ScriptedClient(_one_model)
    opt = HarnessOptimizer(
        editor_client=client,
        editor_model_kwargs={"model": "scripted"},
        executor=LedgerExecutor(),
        base="You total ledgers.",
        **kw,
    )
    return opt, client


def _tool_names(agent):
    return [t.definition.func_name for t in agent.tool_manager.tools]


def test_from_prompt_inverts_render_role_desc():
    assert Harness.from_prompt(FULL.render_role_desc()) == FULL
    assert Harness.from_prompt("Just be helpful.") == Harness(
        system_prompt="Just be helpful."
    )
    with_skill = Harness(skills={"refunds": HarnessSkill("Refund rules", "Body")})
    with pytest.raises(ValueError, match="Skills"):
        Harness.from_prompt(with_skill.render_role_desc())


def test_from_agent_keeps_model_tools_and_settings():
    client = ScriptedClient(lambda p, n: "")
    agent = FULL.to_agent(
        tools=[search], model_client=client, model_kwargs={"model": "m"}, max_steps=4
    )
    opt = HarnessOptimizer.from_agent(agent, k=2)

    assert opt.base == FULL and opt.k == 2
    assert opt.model_client is client and opt.model_kwargs == {"model": "m"}
    improved = FULL.copy()
    improved.skills["refunds"] = HarnessSkill("Refund rules", "Look up, then quote.")
    rebuilt = opt.build_agent(improved)
    assert _tool_names(rebuilt) == ["search", "load_skill"]
    assert rebuilt.max_steps == 4 and "refunds" in rebuilt.role_desc
    # Rollouts must be independent samples, not planner cache hits.
    assert rebuilt.planner._use_cache is False


def test_from_agent_rejects_template_role_desc():
    from adalflow.core.prompt_builder import Prompt

    agent = Agent(
        name="a",
        model_client=ScriptedClient(lambda p, n: ""),
        model_kwargs={"model": "m"},
        role_desc=Prompt(template="You help {{ who }}."),
    )
    with pytest.raises(TypeError, match="from_prompt"):
        HarnessOptimizer.from_agent(agent)


@pytest.mark.parametrize("mode", [PARALLEL, SEQUENTIAL])
def test_refine_runs_with_one_configuration(mode):
    opt, client = _optimizer(k=2, mode=mode)
    res = opt.refine(TASK, task_id="ledger")
    assert res.status == "ok" and res.mode == mode
    assert res.selected.answer == f"Total: {sum(ROWS):.2f}"
    assert res.ledger.rollouts == 8 and res.ledger.sessions == 6
    assert len(client.prompts) == 7  # 6 editor sessions + 1 selection


def test_run_domain_accepts_a_path_and_checks_the_mode(tmp_path):
    opt, _ = _optimizer()
    runs = opt.run_domain([("t1", TASK), ("t2", TASK)], "accounting", tmp_path)
    assert [r.stored for r in runs] == [True, True]
    stored = ExperienceLibrary(tmp_path, mode=PARALLEL).entries("accounting")
    assert [e.task_id for e in stored] == ["t1", "t2"]
    with pytest.raises(ValueError, match="mode"):
        opt.run_domain([], "accounting", ExperienceLibrary(tmp_path, mode=SEQUENTIAL))


def test_estimate_and_argument_checks():
    opt, _ = _optimizer(k=2, agent_kwargs={"max_steps": 5})
    est = opt.estimate(3, reflect=True)
    assert (est.rollouts, est.agent_calls_max, est.editor_calls_max) == (24, 120, 36)
    assert (est.selector_calls_max, est.reflector_calls_max) == (6, 6)
    assert est.total_calls_max == 168
    assert _optimizer(k=1)[0].estimate().selector_calls_max == 0

    with pytest.raises(ValueError, match="model_client"):
        HarnessOptimizer()
    with pytest.raises(ValueError, match="executor"):
        HarnessOptimizer(editor_client=ScriptedClient(_one_model))
