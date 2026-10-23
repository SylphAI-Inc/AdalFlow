# Benchmarks

Benchmarking is an integral development part of the project.

Contributors are encouraged to write benchmarks for their code, in addition to unit tests in the `tests/` directory.

## Available benchmark areas

- `optimize_anything/` — scaffolds for the `optimize_anything` API (heuristic string mutations, no LLM).
- `harness_optimization/` — label-free harness refinement on a tool-using domain, scored with labels on held-out tasks against prompt-only rewriting (real model).
- `evolution/` — `adalflow.optim.evolution` search strategies vs. a best-of-N baseline at equal budget, scored on held-out instances (real model).
- `arc_bench/` — ARC-Bench-style smoke benchmark harness for autonomous research-agent prototypes.
- `itinerary_roi_bench/` — Coordinator + worker itinerary planning smoke benchmark for `ModelROILedger` and `FrozenPrefixMemorySnapshot`.
- `harness2_roi/` — Imports [Harness²](https://github.com/google-research/harness2) run outputs (measured Codex token usage, judged scores) into `ModelROILedger`.

## ARC-Bench-style smoke benchmark

The `arc_bench/` benchmark is a lightweight, local smoke benchmark for testing AdalFlow research-agent orchestration patterns. It currently supports:

1. A simple linear autonomous-research demo.
2. An OpenEvolve-style evolutionary research loop demo.

Both demos test orchestration only. In the evolution demo the executor is a mock and
`eval_accuracy` is a formula that grows with the generation number
(`0.65 + 0.04 * generation + noise`), so rising scores are built in and do not show
that evolution works. For real candidate execution and LLM-guided proposals, see
`adalflow.optim.evolution` and `use_cases/evolution/`.

The evolutionary loop models:

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

Run the deterministic evolution smoke benchmark:

```bash
python benchmarks/arc_bench/run_adalflow_smoke.py \
  --mode evolution \
  --max-iterations 5 \
  --topics benchmarks/arc_bench/topics_smoke.jsonl \
  --output artifacts/arc_bench/adalflow_evolution_smoke
```

Generate a Markdown report:

```bash
python benchmarks/arc_bench/report.py \
  --input artifacts/arc_bench/adalflow_evolution_smoke \
  --output artifacts/arc_bench/adalflow_evolution_smoke_report.md
```

Optional LLM-backed mode replaces the seed, mutation, reflection, and final report-writing components with AdalFlow `Generator` components while keeping execution and evaluation deterministic:

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

> Note: this benchmark is not a verified comparison against AutoResearchClaw. A valid comparison requires running both systems on the same topics, model, hardware, timeout, budget, and scoring rubric.
