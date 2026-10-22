# LLM-guided evolution

`evolve_function_demo.py` evolves a buggy `median` function. Each candidate runs in a
separate Python process against real test cases (`PythonEvaluator`); the proposer reads
the failing cases and tracebacks and rewrites the function.

```bash
python use_cases/evolution/evolve_function_demo.py
python use_cases/evolution/evolve_function_demo.py --log runs/median.jsonl
# real model (requires OPENAI_API_KEY):
python use_cases/evolution/evolve_function_demo.py --provider openai --model gpt-4o-mini
```

With the default `--provider scripted`, a stand-in model fixes exactly what the
feedback reports, so the run shows the mechanics only: feedback reaching the proposer,
lineage, the stop rule and the event log. The sandbox and the tests are real.

`evolve_prompt.py` evolves a prompt instead of a program: the instruction of a
support-ticket router, scored on labeled tickets with held-out tickets for the final
check. One `claude-haiku-5-5` run went from 0.69 to 1.0 on the training tickets in
three rewrites, and from 0.50 to 0.92 on the held-out ones.

```bash
python use_cases/evolution/evolve_prompt.py --provider openai --model gpt-4o-mini
```

Tutorial: `docs/source/new_tutorials/evolution.md`. Strategy and noise benchmarks:
`benchmarks/evolution/`.

Your own task:

```python
from adalflow.components.model_client import OpenAIClient
from adalflow.optim.evolution import LLMProposer, PythonEvaluator, evolve

EVALUATOR = '''
CASES = {"small": ((2,), 4)}       # name -> (args, expected)

def evaluate(candidate):          # candidate is the imported candidate.py module
    scores, notes = {}, []
    for name, (args, want) in CASES.items():
        try:
            got = candidate.solve(*args)
        except Exception as exc:
            got = repr(exc)
        scores[name] = float(got == want)
        if got != want:
            notes.append(f"{name}: expected {want}, got {got}")
    return {"instance_scores": scores, "feedback": "\\n".join(notes)}
'''

result = evolve(
    seed=open("solve.py").read(),
    evaluate=PythonEvaluator(EVALUATOR, timeout=30),
    proposer=LLMProposer("solve() must ...", model_client=OpenAIClient(),
                         model_kwargs={"model": "gpt-4o-mini"}),
    max_evaluations=30,
    stop_score=1.0,
    log_path="runs/solve.jsonl",
)
print(result.best.text)
```

`evaluate` can be any callable `text -> float | dict | Evaluation`, so prompts or
configs can be evolved with an LLM-judged or programmatic metric instead of
`PythonEvaluator`.

Notes:

- `PythonEvaluator` gives process isolation (fresh temp dir, no inherited environment
  variables such as API keys, timeout that kills the process group, CPU and memory
  rlimits on POSIX). It is not a security sandbox: candidates can still use the
  filesystem and network. Run LLM-written code in a container when that matters.
- The evaluation script runs in the candidate's process, so a candidate can tamper
  with its own score. Read the best candidates before trusting their scores.
- With noisy evaluators (LLM judges, sampled benchmarks), pass `confirm=2` (or more)
  so the reported best is a mean over several evaluations rather than one lucky
  score. It costs budget: with a deterministic evaluator leave it at 0.
