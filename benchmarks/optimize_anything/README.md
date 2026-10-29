# optimize_anything benchmark (GEPA-style API scaffold)

`optimize_anything` mirrors GEPA's API, but its mutations are fixed string operations
and no LLM is called, so these results do not measure GEPA-style reflective
optimization. For LLM-guided evolution see `adalflow.optim.evolution` and
`use_cases/evolution/`.

This benchmark provides a reproducible scaffold for `adalflow.optim.optimize_anything` across
three artifact categories:

1. text/prompt artifact
2. code artifact
3. config/SVG-like artifact

Each run reports:
- quality score (maximize)
- token_cost (minimize)
- latency_ms (minimize)

## Run

```bash
python benchmarks/optimize_anything/gepa_parity_benchmark.py
```

The script prints JSON containing baseline (seed) and optimized metrics for each case.