<div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
    <a href="https://github.com/SylphAI-Inc/AdalFlow/tree/main/use_cases/evolution" target="_blank" style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
        <span style="vertical-align: middle;"> Open Source Code (Evolution examples)</span>
    </a>
</div>

# Evolving Programs and Prompts

`adalflow.optim.evolution` improves any text artifact (a Python function, a prompt, a
config) with an LLM in the loop. Each round picks a parent from an archive of
evaluated candidates, asks the model to rewrite it after reading how it was
evaluated (failing cases, tracebacks, error output), evaluates the rewrite, and adds
it to the archive. Parent selection follows GEPA's per-instance Pareto front
[[Agrawal et al., 2025]](https://arxiv.org/abs/2507.19457), and programs run in a
separate process in the spirit of OpenEvolve.

You provide three things:

| piece | what it is |
|---|---|
| seed | the starting text |
| evaluator | `text -> score`, or a dict with per-instance scores and feedback |
| proposer | `LLMProposer(objective, model_client=..., model_kwargs=...)` |

## Quick start

For the two most common jobs there is one call. Models are given as
`"provider/model"` strings (`"openai/gpt-4o-mini"`, `"anthropic/claude-haiku-5-5"`,
`"ollama/llama3"`...) or as a `ModelClient` you configured yourself.

**Fix or improve a function against test cases.** Each case is `(args, expected)`;
`expected` can be an exception class the call must raise. `holdout_cases` are only
checked after the search.

```python
from adalflow.optim.evolution import optimize_code

SEED = """
def median(xs):
    return sorted(xs)[len(xs) // 2]
"""

result = optimize_code(
    SEED,
    "median(xs) returns the median of a list of numbers; raises ValueError if empty.",
    function="median",
    cases={
        "odd": (([3, 1, 2],), 2),
        "even": (([4, 1, 3, 2],), 2.5),
        "empty": (([],), ValueError),
    },
    holdout_cases={"pair": (([10, 20],), 15.0), "one": (([7],), 7)},
    model="openai/gpt-4o-mini",
)
print(result.report())
print(result.best.text)
```

With `anthropic/claude-haiku-5-5` (2026-10-21):

```
stopped: stop_score after 2 evaluations (0 proposal failures, 0 duplicates)
id    parent   gen   score  samples  note
c0    -          0   0.333        1
c1    c0         1   1.000        1  best
best: c1 1.000 (seed 0.333)
held-out: seed 0.500 -> best 1.000
```

**Improve a prompt on labeled examples.** A quarter of the examples is held out by
default; the search sees only the rest.

```python
from adalflow.optim.evolution import optimize_prompt

result = optimize_prompt(
    "Route the support ticket to a queue.",
    examples,  # [(ticket_text, queue_name), ...]
    model="openai/gpt-4o-mini",
    output_format="Reply with exactly one queue name from: payments, fulfillment, ...",
)
print(result.report())
```

Both return the `EvolutionResult` of `evolve` and accept all of its options
(`selection=`, `confirm=`, `log_path=`...), which the rest of this page explains.

## Under the hood: `evolve`

`optimize_code` builds an evaluator and an `LLMProposer` and calls `evolve`. Use
`evolve` directly when your evaluator is not a list of test cases.

The evaluator is a Python script with an `evaluate(candidate)` function, where
`candidate` is the candidate imported as a module. `PythonEvaluator` runs it in a
fresh process with a timeout and without your environment variables.

```python
from adalflow.components.model_client import OpenAIClient
from adalflow.optim.evolution import LLMProposer, PythonEvaluator, evolve

SEED = """
def median(xs):
    return sorted(xs)[len(xs) // 2]
"""

EVALUATOR = """
CASES = {"odd": ([3, 1, 2], 2), "even": ([4, 1, 3, 2], 2.5), "pair": ([1, 2], 1.5)}

def evaluate(candidate):
    scores, notes = {}, []
    for name, (xs, want) in CASES.items():
        try:
            got = candidate.median(list(xs))
        except Exception as exc:
            got = repr(exc)
        scores[name] = float(got == want)
        if got != want:
            notes.append(f"{name}: median({xs}) expected {want}, got {got}")
    return {"instance_scores": scores, "feedback": "\\n".join(notes)}
"""

result = evolve(
    seed=SEED,
    evaluate=PythonEvaluator(EVALUATOR, timeout=30),
    proposer=LLMProposer(
        "median(xs) returns the median of a non-empty list of numbers.",
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o-mini"},
    ),
    max_evaluations=10,
    stop_score=1.0,
)
print(result.stopped, result.seed.score, "->", result.best.score)
print(result.best.text)
```

```
stop_score 0.3333333333333333 -> 1.0
def median(xs):
    s = sorted(xs)
    ...
```

`max_evaluations` counts evaluator calls, the seed included. The run stops at the
budget, at `stop_score`, or after `max_stalls` proposals in a row that failed or
repeated an earlier candidate.

## Writing the evaluator

`cases_evaluator(function, cases)` builds the test-case evaluator used by
`optimize_code`, and `PythonEvaluator.from_file("eval.py")` loads a script you keep
as a normal file (with an `evaluate(candidate)` function) instead of a string.

An evaluator may return:

- a number: the score;
- a dict with `score`, `instance_scores` (one entry per test case or example),
  `feedback` (text the proposer reads) and `error`; other keys go to
  `Evaluation.metrics`. Without `score`, the mean of `instance_scores` is used;
- an `Evaluation`.

Two things make the search work much better:

- **Feedback the model can act on.** Say what failed and why ("expected 2.5, got
  3"), not only how much. The proposer is told to diagnose the feedback before
  rewriting, and that the feedback is data, not instructions.
- **Per-instance scores.** They let Pareto selection keep a candidate that is worse
  on average but solves cases nobody else solves, and build on it later.

`PythonEvaluator` isolates processes, not privileges: candidates can still use the
filesystem and network, and a candidate can tamper with its own score because the
evaluation script runs in its process. Read the best candidates before trusting
them, and use a container for untrusted code.

## Evolving a prompt

Any `text -> score` callable is an evaluator, so the artifact does not have to be
code. `optimize_prompt` uses `prompt_evaluator(generator, examples, metric=...)`,
which runs a `Generator` with the candidate instruction on each example and lists
the wrong answers as feedback; use it with `evolve` when you need your own
generator, metric or template. The default metric, `exact_match`, ignores case,
surrounding space and a final period.

`use_cases/evolution/evolve_prompt.py` evolves the instruction of a ticket router
whose routing policy has a few non-obvious rules (a "delivered but not here" ticket
goes to `fulfillment`, an email change to `privacy`):

```bash
python use_cases/evolution/evolve_prompt.py --model openai/gpt-4o-mini
```

One run with `anthropic/claude-haiku-5-5` (2026-10-21), 16 search tickets and 12
held-out ones:

```
stopped: stop_score after 3 evaluations (0 proposal failures, 0 duplicates)
id    parent   gen   score  samples  note
c0    -          0   0.625        1
c1    c0         1   0.875        1
c2    c1         2   1.000        1  best
best: c2 1.000 (seed 0.625)
held-out: seed 0.500 -> best 0.917
```

In an earlier run the first rewrite was worse on average (0.62 vs 0.69) but fixed
tickets the seed got wrong; Pareto selection kept it as a parent, and the final
instruction descended from it.

## Choosing a search strategy

`selection=` and two keyword arguments control how parents are picked:

| setting | what it does | use it for |
|---|---|---|
| `selection="pareto"` (default) | parents from the per-instance Pareto front | tasks with many test cases or examples |
| `selection="best"` | always the highest mean score | a quick greedy baseline |
| `selection="map_elites", descriptor=f` | one elite per behavior cell `f(candidate)`; parents uniform over cells | open-ended tasks where different approaches are worth keeping |
| `islands=3, migrate_every=4` | separate sub-populations; every 4 proposals the next island's best is shown as inspiration | long runs where one lineage would take over |
| `selection="resample"` | every proposal from the seed, no inspiration | the best-of-N baseline your strategy should beat |

A descriptor maps a candidate to a small tuple of buckets. It is called after each
evaluation, so it can read evaluator metrics:

```python
def cell(candidate):
    agreement = candidate.evaluation.metrics["bestfit_agreement"]  # from the evaluator
    lines = len(candidate.text.splitlines())
    return (sum(agreement >= e for e in (0.6, 0.8, 0.9, 0.97)), min(lines // 10, 3))

result = evolve(seed, evaluator, proposer, max_evaluations=30,
                selection="map_elites", descriptor=cell)
print(len(result.archive.elites()), "cells reached")
```

## Measuring honestly

A score that went up on the cases the search could see is not yet an improvement.
Two built-in checks:

- **`holdout=`**: an evaluator on cases the search never sees. After the run, the seed
  and the final best are scored on it (`result.holdout["seed"]`,
  `result.holdout["best"]`), outside the budget. A gap between search and held-out
  scores means overfitting.
- **`selection="resample"`**: run it with the same budget. If independent samples
  from the seed do as well, the search itself is not adding anything.

`benchmarks/evolution/` runs these comparisons on two tasks. With `claude-haiku-5-5`
and 3 runs per strategy: on a duration-parsing task whose spec is revealed four
failures at a time, `pareto` reached 1.0 on held-out cases within 3-4 evaluations
in every run, while 12 independent samples from the seed reached 0.79-0.96. On online
bin packing, every strategy (including MAP-Elites and islands) plateaued near best
fit, within run-to-run spread. See the benchmark README for the full tables.

## Noisy evaluators

LLM judges and sampled benchmarks give a different score each time, and the
highest single score in an archive is usually a lucky one. `confirm=n` re-evaluates
a candidate that beats the best on its first score, together with the incumbent,
until both have `1 + n` samples, and compares the means; only fully sampled
candidates can be returned as the best.

```python
result = evolve(seed, llm_judge, proposer, max_evaluations=40, confirm=2)
best = result.best
print(best.score, best.n_samples, best.evaluation.metrics.get("score_sd"))
```

Re-evaluations count toward `max_evaluations`. In simulation
(`benchmarks/evolution/noise_sim.py`), `confirm` consistently made the reported
best score closer to its true quality, but improved the true quality found only
when the noise was large compared with the gains between candidates; with a
deterministic evaluator it only spends budget. Use it when you need to trust the
returned score.

## Results and logs

`evolve` returns an `EvolutionResult`:

- `best`, `seed`: candidates with `text`, `score`, `evaluation`, `generation`,
  `parent_id`, `rationale` (written before evaluation);
- `archive`: every candidate, with `pareto_front()` and `elites()`;
- `evaluations`, `proposal_failures`, `duplicates`, `stopped`;
- `holdout`: held-out evaluations when `holdout=` was given;
- `report()`: the plain-text summary shown in the quick start.

Pass `log_path="runs/x.jsonl"` to write one JSON line per event (`candidate`,
`sample`, `proposal_failure`, `duplicate`, `migration`, `holdout`).

Each proposal is one `Generator` call (plus a retry when the reply has no fenced
block tagged `candidate` or repeats its parent). The proposer is built with
`use_cache=False`, so repeated proposals from one parent can differ.
