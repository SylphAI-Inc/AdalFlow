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

Using it with your own agent:

```python
from adalflow.components.model_client import OpenAIClient
from adalflow.optim.harness import (
    BlindSelector, Harness, HarnessEditor, RunnerExecutor, refine_harness,
)

client, kw = OpenAIClient(), {"model": "gpt-4o-mini"}
executor = RunnerExecutor(
    agent_factory=lambda h: h.to_agent(model_client=client, model_kwargs=kw, tools=[...])
)
result = refine_harness(
    task="...",
    base=Harness(system_prompt="..."),
    executor=executor,
    editor=HarnessEditor(model_client=client, model_kwargs=kw),
    selector=BlindSelector(model_client=client, model_kwargs=kw),
    k=2,
    mode="parallel",
)
print(result.best_harness.render_role_desc())
```
