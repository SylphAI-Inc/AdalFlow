# Evolution benchmark: search strategies vs. a best-of-N baseline

`run.py` runs `adalflow.optim.evolution.evolve` on the tasks in `tasks.py` with each
selection strategy at the same evaluation budget, and scores the seed and the final
best on held-out instances the search never sees (`evolve(..., holdout=...)`).

- `pareto` / `best`: build on earlier candidates (per-instance Pareto or greedy).
- `resample`: every proposal is an independent sample from the seed (best-of-N).

```bash
python benchmarks/evolution/run.py --check        # seed scores, no model calls
python benchmarks/evolution/run.py --provider anthropic --model claude-haiku-5-5 \
    --repeats 3 --budget 12 --out runs/evolution_bench
```

Each run writes its event log (`<task>-<strategy>-<repeat>.jsonl`) and the per-run
numbers (`summary.json`, `summary.md`) to `--out`.

## Tasks

- `duration`: parse duration strings (`"1h 30m"`, `"PT1H30M"`, `"01:30:00"`,
  `"2 hours"`, invalid strings raise `ValueError`). The objective does not list the
  formats, and the evaluator reports at most 4 failing inputs per run under opaque
  case names, so the spec has to be recovered over several rounds. 24 search cases,
  24 held-out cases with other values and combinations.
- `binpack`: an online bin-packing heuristic (the FunSearch task), scored as lower
  bound / bins used on 300-item instances. 5 search instances, 10 held-out ones.
  There is no perfect score.

## Reproducing the results

From the repo root, with `adalflow` installed (`pip install -e adalflow` or
`poetry install` in `adalflow/`) and `ANTHROPIC_API_KEY` set:

```bash
python benchmarks/evolution/run.py --check     # no model: seed scores, search / held-out (duration 0.458 / 0.417, binpack 0.946 / 0.943)

# duration rows: 3 strategies x 3 runs, budget 12 (about 3 min with the default 4 workers)
python benchmarks/evolution/run.py --provider anthropic --model claude-haiku-5-5 \
    --tasks duration --strategies pareto best resample --repeats 3 --budget 12 \
    --out runs/evolution_bench/duration

# binpack rows: 2 strategies x 3 runs, budget 30 (about 12 min)
python benchmarks/evolution/run.py --provider anthropic --model claude-haiku-5-5 \
    --tasks binpack --strategies pareto resample --repeats 3 --budget 30 \
    --out runs/evolution_bench/binpack
```

Each command prints and writes the aggregate table (`summary.md`), the per-run
numbers (`summary.json`) and one event log per run. Model sampling is not seeded, so
individual numbers vary between reruns; a rerun on 2026-10-23 gave the same picture:

| task | strategy | held-out best (this rerun) | held-out best (table below) |
|---|---|---|---|
| duration | pareto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| duration | best | 1.000 [1.000, 1.000] | 0.986 [0.958, 1.000] |
| duration | resample | 0.847 [0.833, 0.875] | 0.889 [0.792, 0.958] |
| binpack | pareto | 0.950 [0.947, 0.953] | 0.950 [0.949, 0.951] |
| binpack | resample | 0.947 [0.947, 0.947] | 0.947 [0.947, 0.947] |

`--provider openai --model gpt-4o-mini` (with `OPENAI_API_KEY`) runs the same
benchmark on OpenAI; the numbers below are only for `claude-haiku-5-5`.

## Results (2026-10-21, claude-haiku-5-5, 3 runs per cell)

One set of runs on small tasks; mean [min, max] over the 3 runs.

| task | strategy | budget | search best | held-out best | evals used |
|---|---|---|---|---|---|
| duration | pareto | 12 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 3.3 |
| duration | best | 12 | 0.986 [0.958, 1.000] | 0.986 [0.958, 1.000] | 3.3 |
| duration | resample | 12 | 0.903 [0.833, 0.958] | 0.889 [0.792, 0.958] | 12 |
| binpack | pareto | 30 | 0.951 [0.951, 0.952] | 0.950 [0.949, 0.951] | 30 |
| binpack | resample | 30 | 0.951 [0.951, 0.951] | 0.947 [0.947, 0.947] | 23.7 |

Seeds: `duration` 0.458 search / 0.417 held-out; `binpack` 0.946 / 0.943 (first fit;
best fit scores 0.951 / 0.947).

What they show:

- On `duration`, building on earlier candidates matters: `pareto` reached 1.0 on the
  held-out cases in all 3 runs within 3-4 evaluations, while 12 independent samples
  from the seed never did. One `best` run stalled at 0.958 after 12 proposals that
  repeated earlier candidates.
- On `binpack`, every strategy found best fit in the first proposal or two and then
  barely moved (held-out +0.003 for `pareto` over `resample`). The run with the
  highest search score had the lowest held-out score of the three `pareto` runs, a
  small instance of overfitting the search set.
- Proposals are not very diverse: `resample` produced 100+ duplicate candidates per
  `binpack` run, nearly always best fit. Islands or quality-diversity archives
  (MAP-Elites) are the natural next step for open-ended tasks like this one.
