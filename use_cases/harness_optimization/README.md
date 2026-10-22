# Harness optimization

`ledger_demo.py` runs `adalflow.optim.harness.run_domain` on two ledger-totaling tasks
whose base harness drops credit rows.

```bash
python use_cases/harness_optimization/ledger_demo.py --k 2 --mode parallel
python use_cases/harness_optimization/ledger_demo.py --k 2 --mode sequential
# real model for every role (requires OPENAI_API_KEY):
python use_cases/harness_optimization/ledger_demo.py --provider openai --model gpt-4o-mini
```

The default `--provider scripted` replaces the agent, editor, selector and reflector
with deterministic stand-ins, so it demonstrates the mechanics only: base runs, 2k
candidates, k compositions, blind selection, and a stored lesson that the second
task's editor sees. It says nothing about how much a real model improves.

`refine_existing_agent.py` is the end-to-end path with real models: start from an
`Agent` whose system prompt is stale, `HarnessOptimizer.from_agent`, `estimate`,
`refine` (no labels), save the refined harness, score base vs refined on held-out
tasks with `evaluate`, and deploy with `build_agent`.

```bash
python use_cases/harness_optimization/refine_existing_agent.py \
    --provider openai --agent-model gpt-4o-mini --editor-model gpt-4o
```

Tutorial: `docs/source/new_tutorials/harness_optimization.md`. Held-out benchmark:
`benchmarks/harness_optimization/`.

Using it with your own agent:

```python
from adalflow.components.model_client import OpenAIClient
from adalflow.optim.harness import HarnessOptimizer

opt = HarnessOptimizer(
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o-mini"},
    base="You answer billing questions.",   # or a Harness
    tools=[search],
    k=2,                                     # R = 4k rollouts, P = 3k editor sessions
    # editor_client=..., editor_model_kwargs=...  (a stronger model for the editor)
)
print(opt.estimate(1).total_calls_max)       # upper bound on model calls
result = opt.refine("...")
result.best_harness.save("harness.json")
agent = opt.build_agent(result.best_harness)
```

Start from an agent you already have with `HarnessOptimizer.from_agent(agent)`: it
takes the agent's `role_desc` (`## Guardrails` / `## Memory` / `## Tool guidance`
sections are split out), model, tools and `max_steps`. `opt.run_domain(tasks, domain,
library_dir)` refines several tasks in order and stores lessons for later tasks.
Rollout agents are built with `use_cache=False` so repeated runs are independent; set
`agent_kwargs={"use_cache": True}` only for deployment. Progress is logged at `INFO`
by `adalflow.optim.harness.refine`.

`refine_harness`, `run_domain`, `HarnessEditor`, `BlindSelector`, `ExperienceReflector`
and `RunnerExecutor` stay available for custom setups (e.g. a non-Runner executor).
