# ARC-Bench-style Smoke Benchmark

This directory contains a lightweight ARC-Bench-style smoke benchmark for AdalFlow autonomous research-agent prototypes.

It is designed to validate **agent control-flow shape** rather than claim external benchmark superiority. The current benchmark is local, deterministic by default, and safe to run without API keys.

## Files

- `topics_smoke.jsonl` — 5 ARC-Bench-style smoke topics:
  - 2 machine learning topics
  - 1 statistics topic
  - 1 biology topic
  - 1 quantum topic
- `run_adalflow_smoke.py` — benchmark runner for AdalFlow research demos.
- `metrics.py` — normalized per-topic metrics.
- `report.py` — Markdown report generator.

## Supported modes

### 1. Simple mode

Runs the original linear autonomous-research prototype:

```text
topic → literature → hypothesis → HITL gate → code repair → report status
```

Command:

```bash
python benchmarks/arc_bench/run_adalflow_smoke.py \
  --mode simple \
  --topics benchmarks/arc_bench/topics_smoke.jsonl \
  --output artifacts/arc_bench/adalflow_simple_smoke
```

### 2. Evolution mode

Runs the OpenEvolve-style research loop:

```text
seed candidate
→ mutate
→ execute / repair
→ evaluate
→ reflect
→ archive
→ select best
→ repeat
→ final report
```

Command:

```bash
python benchmarks/arc_bench/run_adalflow_smoke.py \
  --mode evolution \
  --max-iterations 5 \
  --topics benchmarks/arc_bench/topics_smoke.jsonl \
  --output artifacts/arc_bench/adalflow_evolution_smoke
```

Generate a report:

```bash
python benchmarks/arc_bench/report.py \
  --input artifacts/arc_bench/adalflow_evolution_smoke \
  --output artifacts/arc_bench/adalflow_evolution_smoke_report.md
```

## Optional LLM-backed mode

Evolution mode can optionally replace the seed, mutation, reflection, and final report-writing steps with AdalFlow `Generator` components:

```bash
export OPENAI_API_KEY="..."

python benchmarks/arc_bench/run_adalflow_smoke.py \
  --mode evolution \
  --use-llm \
  --provider openai \
  --model gpt-4o-mini \
  --max-iterations 5 \
  --topics benchmarks/arc_bench/topics_smoke.jsonl \
  --output artifacts/arc_bench/adalflow_evolution_llm_smoke
```

Supported providers:

- `openai`
- `groq`
- `anthropic`

The executor and evaluator remain deterministic in this phase so that smoke benchmark metrics stay reproducible and safe. The executor is a mock and `eval_accuracy` is computed as `0.65 + 0.04 * generation` plus a hash-based term, so scores rise with generation by construction, with or without `--use-llm`; the metrics check control flow, not search quality.

## Metrics

The benchmark records one `metrics.json` per topic and one aggregate `summary.json`.

For evolution mode, key metrics include:

- `initial_score`
- `best_score`
- `score_improvement`
- `num_candidates`
- `retry_count`
- `convergence_iteration`
- `archive_size`
- `artifact_completeness`

Example output structure:

```text
artifacts/arc_bench/
├── adalflow_evolution_smoke/
│   ├── ml_001/
│   │   ├── metrics.json
│   │   └── result.json
│   ├── ...
│   └── summary.json
└── adalflow_evolution_smoke_report.md
```

## Important limitation

This is not a verified AdalFlow-vs-AutoResearchClaw benchmark.

A fair external comparison requires:

1. The same topic set.
2. The same LLM model and decoding configuration.
3. The same hardware and sandbox policy.
4. The same timeout and cost budget.
5. The same scoring rubric.
6. Raw artifacts for both systems.

The next step is to add an AutoResearchClaw runner that executes the same `topics_smoke.jsonl` file and emits the same normalized `metrics.json` schema.
