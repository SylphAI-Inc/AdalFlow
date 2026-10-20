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

## Run

```bash
# after harness2 ... and harness2-judge ... with --substrate cx
python benchmarks/harness2_roi/import_harness2_runs.py \
  --runs /path/to/harness2/runs \
  --output artifacts/harness2_roi/report.json

pytest benchmarks/harness2_roi
```

The report has one entry per namespace (tokens, cache hit rate, USD per mode, mean
score) and `roi_by_task`, the ledger's ranking per `harness2/<benchmark>/<mode>`.

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
