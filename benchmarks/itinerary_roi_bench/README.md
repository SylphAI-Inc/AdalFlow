# Itinerary ROI Benchmark (`itinerary_roi_bench`)

A deterministic smoke benchmark for the coordinator + worker pattern built on
`adalflow.optim.model_roi_ledger`:

- **`FrozenPrefixMemorySnapshot`** keeps the session-start memory index frozen in the
  system prompt and surfaces mid-session memory writes as a short `<system-reminder>`
  tail delta, so the cached prompt prefix is not invalidated.
- **`ModelROILedger`** records per-`(task_category, model)` tokens, cache hit rate,
  step success rate, constraint score and TTFT, and routes worker calls to the
  highest-ROI model that clears a quality floor.
- **`DecisionCard`** is a file-backed, non-expiring decision checkpoint a worker can
  park on instead of blocking on an interactive prompt.

The scenarios use the demo in
`use_cases/multi_agent_dag/itinerary_discovery_coordinator_demo.py`: 5 multi-stop
itineraries over a small merchant catalog, each checked for 3 scheduled stops within
budget and opening hours.

## Run

```bash
python benchmarks/itinerary_roi_bench/run_itinerary_roi_benchmark.py \
  --output artifacts/itinerary_roi_bench/report.json

python use_cases/multi_agent_dag/itinerary_discovery_coordinator_demo.py

cd adalflow && pytest tests/test_model_roi_ledger.py
```

## Example output

| Metric | Monolithic baseline | Coordinator + workers |
| --- | --- | --- |
| Scenarios completed | — | 5/5 |
| Prompt-cache hit rate | 13.24% | 90.72% |
| Weighted token cost | 160.82 | 22.04 (−86.3%) |

> Note: no LLM is called. Itinerary planning, ranking and constraint checks run for
> real, but the token counts recorded per turn are fixed, simulated values for both
> arms. The cache-hit and cost figures therefore show how the ledger accounts for the
> two architectures under those assumptions; they are not measured provider savings.
> Model names and tier weights are illustrative and can be overridden via
> `ModelROILedger(model_tier_weights=...)`.
