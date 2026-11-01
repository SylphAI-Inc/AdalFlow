# Evolution benchmark: search strategies vs. a best-of-N baseline

`run.py` runs `adalflow.optim.evolution.evolve` on the tasks in `tasks.py` with each
selection strategy at the same evaluation budget, and scores the seed and the final
best on held-out instances the search never sees (`evolve(..., holdout=...)`).

- `pareto` / `best`: build on earlier candidates (per-instance Pareto or greedy).
- `resample`: every proposal is an independent sample from the seed (best-of-N).
- `map_elites`: one elite per behavior cell (tasks with a descriptor); `islands`:
  `pareto` on 3 islands with migration.

```bash
python benchmarks/evolution/run.py --check        # seed scores, no model calls
python benchmarks/evolution/noise_sim.py          # confirm= under simulated noise, no model
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
- `size`: parse data-size strings (`"10KB"` decimal, `"4MiB"`/`"2Gi"`/`"4K"` binary,
  `"10 kilobytes"`, `"1,024"`; invalid strings raise `ValueError`), built like
  `duration` with the same evaluator. 24 search cases, 24 held-out cases.
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

## Diversity: MAP-Elites and islands (2026-10-21, claude-haiku-5-5, 3 runs, budget 30)

`map_elites` uses the `binpack` descriptor `binpack_cell`: how often a candidate's
decisions agree with best fit (5 bands) x code length (4 buckets). `islands` is
`pareto` on 3 islands, migrating every 4 proposals.

| strategy | search best | held-out best | duplicates | cells reached |
|---|---|---|---|---|
| pareto | 0.954 [0.952, 0.959] | 0.948 [0.947, 0.949] | 0 | 4.3 |
| map_elites | 0.953 [0.951, 0.955] | 0.950 [0.947, 0.953] | 3.7 | 5.0 |
| islands | 0.952 [0.952, 0.952] | 0.950 [0.949, 0.951] | 1.7 | 4.3 |

The differences are within run-to-run spread. The best single held-out result so
far came from `map_elites` (0.953), and `pareto` again overfit the most (its highest
search score, 0.9585, had its lowest held-out score). The 30 candidates of a run
fill only 4-6 of 20 cells, so the proposer's narrow output, not parent selection,
limits diversity here; more varied proposals (temperature, prompts that ask for a
different approach) are the next lever.

## Noise: `confirm` in simulation

`noise_sim.py` (no model) evolves a scalar "true quality" with mutations
N(-0.01, 0.03) and an evaluator that adds Gaussian noise; budget 60, greedy
selection, 200 runs per cell (± is the standard error).

| noise sd | confirm | true quality of returned best | reported minus true |
|---|---|---|---|
| 0 | 0 | 0.953 ± 0.008 | +0.000 |
| 0 | 2 | 0.749 ± 0.004 | +0.000 |
| 0.02 | 0 | 0.742 ± 0.008 | +0.028 |
| 0.02 | 2 | 0.692 ± 0.005 | +0.009 |
| 0.05 | 0 | 0.557 ± 0.005 | +0.107 |
| 0.05 | 1 | 0.578 ± 0.005 | +0.066 |
| 0.1 | 0 | 0.511 ± 0.004 | +0.228 |
| 0.1 | 4 | 0.526 ± 0.003 | +0.078 |

`confirm` always shrinks the winner's curse (the reported best overstates less), but
it spends budget on re-evaluation, so the true quality found goes down when noise is
small relative to the mutation steps and goes up only a little when noise is large.
Use it when an honest score for the returned candidate matters, not as a way to
search better. Full grid: `python benchmarks/evolution/noise_sim.py`.

## Which failures the evaluator shows (2026-10-26, 5 runs per cell, budget 12)

`factorial.py` crosses lineage (`seed`: every proposal rewrites the seed, as
`resample`; `parent`: rewrites the greedy best) with what the proposer sees about
the candidate (`none`, `score`, `full`). `anchoring.py` keeps `full` with 4 shown
failures and changes which 4: `first` (case order), `random`, `iso` (duration: first
3 plus an ISO 8601 case) or `diverse` (the label-free selection that
`cases_evaluator` now uses by default).

```bash
python benchmarks/evolution/factorial.py --provider anthropic --model claude-haiku-5-5 \
    --tasks duration --repeats 5 --budget 12 --out runs/factorial/duration
python benchmarks/evolution/anchoring.py --provider anthropic --model claude-haiku-5-5 \
    --tasks size --cells seed-none seed-score seed-full-first seed-full-diverse \
    parent-none parent-score parent-full-first parent-full-diverse --out runs/factorial/size
python benchmarks/evolution/anchoring.py --report --baseline runs/factorial/{duration,anchoring,diverse,size}/summary.json
```

Held-out best (mean [bootstrap 95% interval]):

| model, task | seed, none | seed, full, first | seed, full, diverse | parent, full, first | parent, full, diverse |
|---|---|---|---|---|---|
| haiku, duration | 0.942 [0.925, 0.958] | 0.842 [0.833, 0.858] | 0.958 [0.933, 0.983] | 1.000 | 1.000 |
| haiku, size | 0.658 [0.642, 0.667] | 0.817 [0.783, 0.833] | 0.900 [0.883, 0.917] | 1.000 | 1.000 |
| sonnet, duration | 0.975 [0.958, 0.992] | 0.983 [0.967, 1.000] | 0.983 [0.967, 1.000] | | |

Shown failures act as a spec. On duration the first four failures are all unit
suffixes (`'90'`, `'1h30m'`, `'1h 30m 15s'`, `'3d'`); with them, no haiku proposal
(0/55) handled ISO 8601, against 27/55 with no feedback at all, and swapping one
shown failure for an ISO case gave 55/55. On size, separators (`"1,024"`) were
covered by 55/55 proposals when shown and 0/55 when not. The diverse sample keeps
the first failure and adds the least similar ones, which fixed most of it without
labels. Sonnet covered every format with or without feedback, so the effect needs a
proposer whose prior does not already cover the spec. With parent lineage every run
solved the search set either way (median 6–8k proposer tokens), since later rounds
reveal the failures the first sample missed.

### Follow-ups (2026-10-27, claude-haiku-5-5, 5 runs per cell)

`minibatch` is GEPA-style: four cases drawn at random from all cases (passing ones
shown as passed) in each evaluation. A `+spec` cell appends the full format list
to the objective. `prompt_anchoring.py` runs the same comparison on a prompt task
that is not a parser: label what kind of value a string is (`ipv4`, `hex_color`,
`us_date`..., 12 labels, examples grouped by label), with haiku as proposer and
task model.

```bash
python benchmarks/evolution/anchoring.py --provider anthropic --model claude-haiku-5-5 \
    --tasks duration size --cells seed-full-minibatch parent-full-minibatch \
    seed-none+spec seed-full-first+spec seed-full-diverse+spec --out runs/factorial/minibatch_spec
python benchmarks/evolution/prompt_anchoring.py --provider anthropic \
    --model claude-haiku-5-5 --budget 8 --out runs/factorial/prompt
```

| task | seed, none | seed, full, first | seed, full, minibatch | seed, full, diverse |
|---|---|---|---|---|
| duration | 0.942 | 0.842 | 0.933 [0.883, 0.975] | 0.958 |
| size | 0.658 | 0.817 | 0.733 [0.650, 0.817] | 0.900 |
| value labels (prompt) | 0.683 [0.650, 0.725] | 0.717 [0.633, 0.800] | 0.800 [0.708, 0.908] | 0.917 [0.842, 0.992] |
| size, `+spec` | 1.000 | 1.000 | | 1.000 |
| duration, `+spec` | 0.958 | 0.958 | | 0.958 |

- On the prompt task the first four wrong answers are all dates; the labels for
  `time`, `ipv4`, `ipv6` and `semver` were then almost never learned (0–2 of 35
  proposals). With the diverse sample `ipv4` and `ipv6` reached 28 of 32; `time`
  and `semver` stayed hard (3 and 5 of 32).
- A random minibatch shows fewer failures (passing cases take slots) and lands
  between the two, below `first` on size. With parent lineage it also solved
  every run, but needed a median of 13.5k (duration) and 20.8k (size) proposer
  tokens against 6–8k for `first` and `diverse`.
- With the spec in the objective the reveal no longer matters. The duration
  `+spec` runs used a spec that called two-field clock times `H:MM` while the
  evaluator wants `M:SS`, so the clock cases failed in all three cells alike; the
  script now says `M:SS`.

### Second proposer and GEPA itself (2026-10-27, gpt-4.1-mini, 5 runs per cell)

The same cells with `gpt-4.1-mini` as proposer (and task model on the prompt task),
plus GEPA 0.0.7 on the prompt task through `gepa_baseline.py`: same task-model call,
exact match and examples, `max_metric_calls=192` (what `evolve` spends at budget 8),
reflection and task model the same model.

```bash
python benchmarks/evolution/anchoring.py --provider openai --model gpt-4.1-mini \
    --tasks duration size --budget 12 --out runs/factorial/openai \
    --cells seed-none seed-full-first seed-full-diverse seed-full-minibatch parent-full-first
python benchmarks/evolution/prompt_anchoring.py --provider openai --model gpt-4.1-mini \
    --budget 8 --out runs/factorial/openai_prompt
python benchmarks/evolution/gepa_baseline.py --provider openai --model gpt-4.1-mini \
    --minibatch 3 --out runs/factorial/gepa_openai
```

Held-out best, seed lineage (resample from the seed):

| task (gpt-4.1-mini) | none | full, first | full, minibatch | full, diverse |
|---|---|---|---|---|
| duration | 0.633 [0.608, 0.658] | 0.592 [0.583, 0.608] | 0.533 [0.483, 0.608] | 0.717 [0.692, 0.742] |
| size | 0.742 [0.675, 0.808] | 0.717 [0.683, 0.750] | 0.817 [0.767, 0.867] | 0.733 [0.692, 0.775] |
| value labels (prompt) | 0.283 [0.242, 0.333] | 0.383 [0.325, 0.450] | 0.533 [0.458, 0.600] | 0.608 [0.567, 0.650] |

Value labels, building on parents, against GEPA (held-out best, tokens per run):

| method | gpt-4.1-mini | claude-haiku-5-5 |
|---|---|---|
| `evolve` parent, full, first | 0.933 [0.900, 0.958], 27k | 0.975 [0.942, 1.000], 67k |
| `evolve` parent, full, diverse | 0.942 [0.908, 0.975], 23k | 0.983 [0.967, 1.000], 60k |
| `evolve` parent, full, minibatch | 0.900 [0.858, 0.942], 42k | 0.967 [0.917, 1.000], 84k |
| GEPA, minibatch 3 | 0.908 [0.825, 0.983], 91k | 0.983 [0.967, 1.000], 221k |
| GEPA, minibatch 4 | 0.858 [0.758, 0.925], 98k | 0.992 [0.975, 1.000], 222k |

- Without lineage the diverse sample beats the first four on duration and on the
  prompt task with gpt-4.1-mini as well, but not on size: there it moves the
  crowding-out to other formats (`separator` 0 → 53 of 55 proposals, `letter` 47 →
  4) and the random minibatch does best. Which sample wins depends on what the
  proposer already knows.
- Building on parents matters more than which failures are shown; with it
  `evolve` and GEPA reach the same held-out score. The token gap is not search
  efficiency: proposer tokens are close (gpt-4.1-mini 2.6–5.1k against 4.6–4.9k).
  It comes from the task model, for two reasons. `evolve` stops once the search
  set is solved, after 3–5 of its 8 evaluations, while GEPA spends its whole
  budget. GEPA's best prompts are also 2–5 times longer (gpt-4.1-mini 1953 against
  about 360 characters, haiku 3165 against about 900), which makes each task-model
  call 2–3 times dearer, then and after deployment.

### Two more prompt tasks and tokens to solve (2026-10-27, 5 runs per cell)

`prompt_anchoring.py --task dates` rewrites dates in 12 formats as `YYYY-MM-DD` (the
slash/dot and two-digit-year conventions come only from feedback); `--task routing`
is the support-ticket router of `use_cases/evolution` (queue names come only from
feedback). `gepa_baseline.py --task` runs GEPA on the same tasks at 8 x (search
examples) metric calls. Tokens to solve are proposer plus task tokens when the
search set is first fully solved, median over the runs that solved it; held-out
scoring is metered apart.

```bash
for t in dates routing; do
  python benchmarks/evolution/prompt_anchoring.py --provider openai --model gpt-4.1-mini \
      --task $t --out runs/factorial/t_openai_$t
  python benchmarks/evolution/gepa_baseline.py --provider openai --model gpt-4.1-mini \
      --task $t --out runs/factorial/t_gepa_openai_$t
done
```

Building on parents against GEPA (held-out best; solved runs; tokens to solve;
characters in the best prompt):

| task, model | `evolve` parent, full, first | `evolve` parent, full, diverse | GEPA, minibatch 3 |
|---|---|---|---|
| dates, gpt-4.1-mini | 0.992; 5/5; 4.8k; 210 | 1.000; 5/5; 4.7k; 227 | 0.983; 5/5; 14.8k; 1541 |
| dates, haiku | 0.983; 5/5; 42k; 1532 | 0.975; 5/5; 40k; 1679 | 0.983; 5/5; 33k; 2075 |
| routing, gpt-4.1-mini | 0.850; 2/5; 19k; 1056 | 0.783; 3/5; 17k; 1207 | 0.783; 2/5; 32k, 74k; 2525 |
| routing, haiku | 0.967; 5/5; 33k; 1863 | 0.917; 5/5; 38k; 2230 | 0.833; 2/5; 121k, 142k; 3562 |
| value labels, gpt-4.1-mini | 0.933 | 0.942 | 0.958; 3/5; 67k; 2300 |
| value labels, haiku | 0.975 | 0.983 | 0.975; 5/5; 98k; 2650 |

(The value-label `evolve` runs predate the tokens-to-solve meter.)

- Quality is the same within the intervals everywhere except routing with haiku,
  where `evolve` building on parents solved all five runs and GEPA two.
- To the point of first solving the search set, GEPA spent more in 4 of the 5
  comparable settings (about 2–4 times); on dates with haiku it spent less. Its
  best prompts were longer in all six (1.2–7 times).
- Seed lineage on routing, gpt-4.1-mini / haiku: first 0.450 / 0.483, diverse
  0.433 / 0.417, minibatch 0.550 / 0.567. With six queue names the proposer cannot
  guess, there is no prior for the shown failures to crowd out, and the diverse
  sample does not help; the random minibatch, which also shows answers that were
  right, does best. Dates is at the ceiling for every cell.

### Where GEPA's tokens go (2026-10-27, traced reruns, 5 runs per cell)

`gepa_baseline.py` now logs every task-model batch. Share of each run's tokens,
gpt-4.1-mini unless marked (full = the whole search set after an accepted
candidate; minibatch = GEPA's 3-example checks; reflection = the proposer):

| task | total/run | full | minibatch | reflection | tokens to solve, median (solved) |
|---|---|---|---|---|---|
| dates | 58k | 17% | 82% | 1% | 11.9k (5/5) |
| dates, haiku | 160k | 33% (17% in the 4 one-reflection runs) | 64% (81%) | 3% | 26.3k (5/5) |
| routing | 66k | 47% | 41% | 12% | 35.6k (1/5) |
| value labels | 90k | 62% | 32% | 6% | 69.8k (3/5) |

- On a task one reflection solves (dates), GEPA keeps checking the solved prompt
  on minibatches until the budget runs out: about 48 of them, 64–82% of its tokens,
  spent after the search set already scores 1.0. `evolve` stops at that point.
- On harder tasks most tokens go to full search-set evaluations of accepted
  candidates. Per call, a minibatch costs more than a full-set call (the prompt is
  longer by then), and GEPA's prompts are 1–7k characters.
- The proposer is 1–12% of the tokens. The difference from `evolve` is
  evaluation, not reflection.

Value labels with the meter (`evolve`, building on parents, full feedback):

| model | first | diverse | GEPA, minibatch 3 |
|---|---|---|---|
| gpt-4.1-mini | 0.892; 3/5; 25.4k; 606 | 0.950; 5/5; 16.9k; 504 | 0.900; 3/5; 69.8k; 2258 |
| claude-haiku-5-5 | 0.942; 4/5; 49.2k; 1446 | 0.983; 5/5; 43.1k; 1387 | 0.983; 5/5; 98k; 2650 |

These replace the estimates above: GEPA needs about 4× (gpt-4.1-mini) and 2.3×
(haiku) the tokens `evolve` with the diverse sample needs to solve the search set.
The GEPA reruns of this section are a second set of 5 seeds; they differ from the
first set within its intervals (value labels 0.900 vs 0.958, dates 1.000 vs 0.983).

The held-out email that gpt-4.1-mini misses is a single example:
`root@localhost.dev` is answered `hostname` (once `url`) in all 10 `evolve` runs
and all 5 GEPA runs; `x_y-z@sub.example.org` is always right, and haiku gets both.
That is the task model reading `localhost` (the search set labels
`http://localhost:8080` as `url`), not a family the proposer left out.

### Free-text replies with an LLM judge (2026-10-27, `--task replies`, 5 runs per cell)

`replies_task.py`: support replies for a fictional company, 8 kinds of message with
house policies the objective does not state; each example's expected value is a
rubric, scored PASS/FAIL by a fixed judge (gpt-4.1-mini, temperature 0, cached).
Judge check: the all-policies prompt scores 0.969 / 1.000 (gpt-4.1-mini / haiku), the
seed 0.031 / 0.094; re-judging 128 pairs flipped none. Held-out best, mean [95% CI]:

| cell | gpt-4.1-mini | haiku |
|---|---|---|
| seed-none | 0.062 | 0.075 |
| seed-full-first | 0.287 [0.237, 0.338] | 0.338 [0.312, 0.388] |
| seed-full-diverse | 0.438 [0.400, 0.475] | 0.487 [0.438, 0.537] |
| seed-full-minibatch | 0.438 [0.388, 0.487] | 0.475 [0.412, 0.550] |
| parent-full-first | 0.950 [0.875, 1.000]; solved 4/5; 32.3k to solve; 2176 chars | 0.988 [0.963, 1.000]; 5/5; 98.5k; 4596 chars |
| parent-full-diverse | 0.900 [0.875, 0.950]; 2/5; 42.3k; 1923 chars | 0.812 [0.588, 0.963]; 4/5; 83.3k; 3776 chars |
| GEPA (minibatch 3) | 0.887 [0.812, 0.938]; 1/5; 77.6k; 4012 chars | 0.912 [0.850, 0.975]; 1/5; 268k; 5382 chars |

- Without lineage, diverse beats first on both models with disjoint CIs: first's
  rewrites fix refund and password only; diverse adds cancel. Random minibatch ties
  diverse but covers other kinds (bug, feature, delete). No reveal of 4 reaches all 8
  kinds from the seed.
- With lineage, first is at least as good as diverse here (haiku diverse has one run
  at 0.375; without it 0.922). This is the first task where diverse is worse under
  lineage.
- evolve-parent beats GEPA at equal task calls on both models, solves the search set
  more often (4/5 and 5/5 vs 1/5) and with fewer tokens; GEPA's prompts are longer.
- Judge cost per model: about 4.3k calls and 1.0-1.3M tokens for the evolve cells,
  0.7-0.8k calls and 0.15-0.21M for GEPA.

### Public benchmarks and carrying facts (2026-10-28, gpt-4.1-mini, 10 runs per cell)

Data (not checked in):

```bash
mkdir -p data/public/banking77 data/public/bbh
B=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data
curl -sfL -o data/public/banking77/train.csv $B/train.csv
curl -sfL -o data/public/banking77/test.csv $B/test.csv
curl -sfL -o data/public/bbh/salient_translation_error_detection.json \
  https://raw.githubusercontent.com/suzgun/BIG-Bench-Hard/main/bbh/salient_translation_error_detection.json
# ScoNe (--task scone): the six <condition>.csv test files from github.com/selenashe/ScoNe in data/scone
```

On Banking77 (12 intents) the intent names appear only in the feedback, and the
task model never sees the feedback. Building on the best parent alone reached 0.372
held-out; GEPA 0.0.7 (minibatch 3, same task-call budget) reached 0.664, because its
reflection prompt asks for domain facts to be written into the instruction. Adding
one sentence that asks for the same (`--carry-facts`, now the default of
`optimize_prompt`) closes and reverses the gap:

| arm | Banking77 held-out | intents named in best prompt (of 12) | tokens/run (median) |
|---|---|---|---|
| seed, first-k | 0.225 | 3.2 | |
| parent, first-k | 0.372 | 5.6 | 38.7k |
| GEPA 0.0.7 mb3 | 0.664 | 9.0 | 106.9k |
| seed + carry-facts | 0.317 | 4 | |
| parent + carry-facts | 0.856 | 11.7 | 45.8k |

Without lineage the sentence does little (the seed is rewritten from scratch each
time, so facts cannot accumulate). On BBH salient-translation-error detection, where
the six options are already in the seed prompt, the sentence cost 0.061 held-out,
so turn it off for fixed-option reasoning tasks.

```bash
bash benchmarks/evolution/reproduce_banking77.sh            # the three main rows, about $1
CHECK=1 bash benchmarks/evolution/reproduce_banking77.sh    # fake models, no API
python benchmarks/evolution/prompt_anchoring.py --provider openai --model gpt-4.1-mini \
  --task bbh_ste --repeats 10 --cells seed-full-first parent-full-first --carry-facts --out runs/bbh_carry
```
