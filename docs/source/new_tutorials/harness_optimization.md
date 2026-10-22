<div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
    <a href="https://github.com/SylphAI-Inc/AdalFlow/tree/main/use_cases/harness_optimization" target="_blank" style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
        <span style="vertical-align: middle;"> Open Source Code (Harness examples)</span>
    </a>
</div>

# Optimizing Agent Harnesses

An agent's *harness* is everything around the model that makes it act as an agent:
the system prompt, guardrails, memory, tool guidance and on-demand skills.
`adalflow.optim.harness` improves the harness of an AdalFlow `Agent` with the model
fixed and **without labels**, following the propose-probe-compose-select recursion
of Harness² [[Xu et al., 2026]](https://arxiv.org/abs/2610.14224) (independent
implementation).

On one task, refinement:

1. runs the agent with the current harness (probe);
2. has an editor model propose harness changes from the trajectories (propose);
3. runs the agent with each proposal, and merges the proposals (compose);
4. shows the final deliverables, anonymized, to a selector model that picks one
   without an answer key (select).

With `k` parallel branches this costs R = 4k agent rollouts and P = 3k editor
sessions per task, which `HarnessOptimizer.estimate` turns into a call budget before
you run anything.

## Quick start: refine an agent you already have

`use_cases/harness_optimization/refine_existing_agent.py` starts from an agent with a
stale system prompt (it says a paginated tool returns everything in one call):

```python
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client import OpenAIClient
from adalflow.optim.harness import HarnessOptimizer

agent = Agent(
    name="accounts",
    role_desc="You answer questions about customer accounts using the tools. "
    "list_transactions returns all of an account's transactions in one call. "
    "End with 'Answer: <number>'.",
    tools=[list_transactions, calculator],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o-mini"},
    max_steps=8,
)

# Keeps the agent's model, tools and settings; a stronger model edits and selects.
opt = HarnessOptimizer.from_agent(
    agent,
    editor_client=OpenAIClient(),
    editor_model_kwargs={"model": "gpt-4o"},
    k=1,
)
print(opt.estimate(1).total_calls_max)  # upper bound on model calls for one task

res = opt.refine("What is the total of all transactions in account acct-1?")
best = res.best_harness
print(res.status, res.selected.answer)
print(opt.base.diff(best).summary())  # e.g. "added: tool_guidance; modified: system_prompt"
best.save("runs/accounts_harness.json")

deployed = opt.build_agent(best)  # same model and tools, refined harness
Runner(agent=deployed).call(prompt_kwargs={"input_str": "..."})
```

One run with `claude-haiku-4-5` as the agent and `claude-haiku-5-5` as the editor
(2026-10-21) rewrote the stale sentence from what the trajectories showed (a
non-null `next_cursor`):

```
at most 38 model calls for one task
status: ok, rollouts R=4, sessions P=3
changes: added: tool_guidance; modified: system_prompt
refined system prompt:
You answer questions about customer accounts using the tools. list_transactions
returns an account's transactions one page at a time. If the response includes a
non-null next_cursor, call list_transactions again with that cursor until
next_cursor is null. ...
held-out accuracy: base 0.89 -> refined 1.00
```

The held-out numbers come from 9 rollouts per harness, so treat the gap as an
example, not a measurement (see "Measuring on held-out tasks" below).

## What a harness is

```python
from adalflow.optim.harness import Harness, HarnessSkill

h = Harness(
    system_prompt="You answer billing questions.",
    guardrails="Never quote a refund amount you did not look up.",
    memory="Refunds over $500 need a manager.",
    tool_guidance="Search the knowledge base before answering.",
    skills={
        "refund-policy": HarnessSkill(
            description="Rules for partial refunds",
            body="1. Look up the order. 2. ...",
        )
    },
)
print(h.render_role_desc())  # the always-on text the Agent receives as role_desc
```

Skills are not always on: the agent sees their names and descriptions and gets a
`load_skill` tool to read a skill's body when it applies. `Harness.from_prompt(text)`
and `Harness.from_agent(agent)` build a harness from an existing prompt (an agent
whose `role_desc` is a `Prompt` object is rejected; pass its text instead).
`to_dict`, `save` and `load` round-trip it as JSON, and `diff` lists changed
components.

## Reading a refinement result

`refine` returns a `RefinementResult`:

| field | meaning |
|---|---|
| `status` | `"ok"`, `"base-failed"` (no base run worked) or `"no-delivery"` |
| `base_runs`, `candidates`, `finals` | `ProbeResult`s with `answer`, `trajectory`, `harness` |
| `selected`, `best_harness` | the blind pick (the base harness if nothing was picked) |
| `selection.why`, `selection.fallback` | the selector's reason; whether it fell back |
| `stages` | each editor session: attempts, whether it kept the base, its changes |
| `ledger` | rollouts and sessions actually used (`R`, `P`) |

Pass `mode="sequential"` to have each editor session build on the previous one
instead of branching from the base. Set `logging.getLogger("adalflow.optim.harness")`
to `INFO` to follow the stages as they run.

## Many tasks: carry lessons forward

`run_domain` refines tasks in order and, after each one, has a reflector write an
evidence-backed lesson into an experience library on disk. Later tasks' editors see
the lessons for their domain.

```python
runs = opt.run_domain(
    [("acct-1", "What is the total ..."), ("acct-2", "How many transactions ...")],
    domain="accounts",
    library="runs/library",  # a directory; one library per mode
)
for run in runs:
    print(run.task_id, run.result.status, "lesson stored:", run.stored)
    if run.reflection and run.reflection.experience:
        print("  ", run.reflection.experience.lesson)
```

## Measuring on held-out tasks

Refinement never sees labels, so check the result separately. `evaluate` runs a
harness on labeled tasks and returns per-task scores:

```python
def score(task_id, probe):  # your metric; 1.0 / 0.0 for exact answers
    return float(probe.answer is not None and expected[task_id] in probe.answer)

held_out = [("acct-11", "What is ..."), ("acct-12", "What is ...")]
before = opt.evaluate(opt.base, held_out, score, repeats=4, workers=4)
after = opt.evaluate(best, held_out, score, repeats=4, workers=4)
print(before.mean, after.mean, after.task_means())
```

Agent runs vary, so use several repeats and several tasks. In
`benchmarks/harness_optimization/` (3 trials, 4 held-out tasks x 4 rollouts), the
refined harness and the stale base were within noise of each other (1.000 vs 0.938),
because the agent usually noticed the pagination on its own. A one-shot prompt
rewrite that never ran the agent did worse than both in 2 of 3 trials (0.31, 0.38):
it made the stale claim stronger. In our checks stronger agents worked around flawed
harnesses by themselves, so expect the largest gains with weaker or cheaper agents
and with flaws an agent cannot see from inside one run.

## Cost and practical notes

- `estimate(n_tasks, reflect=...)` gives upper bounds per role (agent steps,
  editor, selector, reflector). Actual use is usually lower.
- Rollouts are built with `use_cache=False`, so repeated runs are independent
  samples. Pass `agent_kwargs={"use_cache": True}` for deployment if you want caching.
- In AdalFlow, `max_tokens` in a `Generator`'s `model_kwargs` limits the *prompt*
  (it truncates by characters and is not sent to the API). Do not pass it as an
  output limit for the editor or reflector, or their evidence gets cut.
- To try the mechanics without an API key: `python use_cases/harness_optimization/ledger_demo.py`
  runs every role with deterministic stand-ins.
