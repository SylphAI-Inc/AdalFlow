# Harness² ROI import (`harness2_roi`)

Imports the outputs of [Harness²](https://github.com/google-research/harness2)
runs into `adalflow.optim.ModelROILedger`, so model/harness configurations can be
compared on measured token usage, prompt-cache hit rate, cost and judged quality.

Harness² itself is not a dependency: the importer only reads its run directories.

## What is read

Per namespace (`<bench>-<substrate>-<mode><k>[...]`) under the Harness² runs root:

| Source | Used for |
| --- | --- |
| `config.json` | executor / improver model names |
| `<task>/summary.json` | rollouts per mode, stage retries / fallbacks, mode ledger USD |
| `judged.json` | judged score (`n_passed / n_criteria`) of the run selected per task and mode |
| rollout `*.jsonl` streams | Codex `turn.completed` usage: input, cached input, output, reasoning tokens |

Rollout directories are located from the base rollouts' recorded `run_dir`, for both
the nested layout (LAB, local demo) and the flattened `__` layout (JobBench,
WorkBuddy; JobBench streams are read from the sibling `model_traj/` tree).

## How to use

The question it answers: *on my benchmark, which executor model (or harness setup)
gives the best judged quality per token and dollar?* Harness² records cost and
scores per run; the importer puts them into one `ModelROILedger` so the setups are
ranked side by side.

**1. Run Harness² once per setup**, each into its own runs directory, with the Codex
executor (`--substrate cx`; only Codex streams carry token usage). See the
[Harness² README](https://github.com/google-research/harness2#run-a-benchmark) for
installation and model settings (`HARNESS2_CODEX_MODEL`, `HARNESS2_RUNS_DIR`).

```bash
# in the harness2 checkout; repeat with another model and runs dir (e.g. runs-gemini)
export HARNESS2_CODEX_MODEL=<executor model> HARNESS2_RUNS_DIR=runs-gpt
harness2-setup lab --substrate cx   # records the Codex model; rerun after changing it
harness2 --bench lab --substrate cx --k 1 --mode parallel --max-tasks 1
harness2-judge --bench lab --substrate cx --k 1 --mode parallel
```

Judging is what produces the scores; tasks that were not judged are listed in the
report but not ranked.

**2. Import the runs directories** (from the AdalFlow repo root):

```bash
python benchmarks/harness2_roi/import_harness2_runs.py \
  --runs ../harness2/runs-gpt ../harness2/runs-gemini \
  --output artifacts/harness2_roi/report.json
```

**3. Read the report.** `namespaces` has one entry per Harness² namespace: tokens,
cache hit rate, USD per mode and mean judged score. `roi_by_task` ranks the
executor models for each `harness2/<benchmark>/<mode>`, best first. On the test
fixture (two synthetic runs with equal token usage, where one scores 4/4 and the
other 3/4), the ranking is:

| model | invocations | cache hit rate | mean score | roi_score |
|---|---|---|---|---|
| gemini-3.5-flash | 4 | 0.867 | 1.00 | 68.55 |
| gpt-5-codex | 4 | 0.867 | 0.75 | 51.41 |

The same functions can be used from Python, for example to add Harness² runs to a
ledger that already holds your own AdalFlow runs:

```python
from pathlib import Path
from adalflow.optim import ModelROILedger
from benchmarks.harness2_roi.import_harness2_runs import load_namespace, to_ledger

ledger = ModelROILedger()  # or an existing ledger
to_ledger([load_namespace(Path("../harness2/runs-gpt/lab-cx-agg1"))], ledger)
print(ledger.summarize_by_task()["harness2/lab/agg"])
```

Tests (no Harness² install needed): `pytest benchmarks/harness2_roi`.

## Caveats

- Token usage is only available for the **Codex** executor (`--substrate cx`).
  OpenCode rollouts import with zero tokens; their USD is still reported.
- Codex reports cached tokens as a subset of input tokens; the ledger records
  `input - cached` as uncached input.
- Harness² scores only the selected run per task and mode. That score is attributed
  to every rollout spent on the task in that mode; ungraded tasks are reported but
  left out of the ledger.
- Harness²'s own `cached` flag means a resumed (reused) stage, not an LLM
  prompt-cache hit; it is not used here.
- `harness2-demo` runs without a model, so importing its output yields zero tokens
  and no scores. It only checks the directory layout.
