"""Deterministic smoke benchmark for coordinator + worker itinerary planning ROI.

Compares, over 5 itinerary scenarios:

1. A monolithic baseline: one frontier model for every turn, with memory writes
   rewriting the prompt prefix (so little of each prompt is served from cache).
2. Coordinator + workers using ``FrozenPrefixMemorySnapshot`` and
   ``ModelROILedger`` routing.

Itinerary planning, ranking and constraint checks run for real. Token counts are
simulated, fixed per turn type (no LLM is called), so the cache-hit-rate and cost
figures show how the ledger accounts for the two architectures, not measured
provider savings.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "adalflow"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adalflow.optim.model_roi_ledger import (  # noqa: E402
    InvocationTelemetry,
    ModelROILedger,
)
from use_cases.multi_agent_dag.itinerary_discovery_coordinator_demo import (  # noqa: E402
    COORDINATOR_TASK,
    WORKER_TASK,
    ItineraryDiscoveryCoordinator,
    default_sample_catalog,
)

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "scn-01-morning-artisan",
        "start_coords": (37.3900, -122.0810),
        "start_hour": 8.5,
        "end_hour": 14.5,
        "budget_usd": 95.0,
        "tags": [
            "artisan-coffee",
            "local-roaster",
            "vegetarian-friendly",
            "scenic-walk",
        ],
    },
    {
        "id": "scn-02-courtyard-lunch",
        "start_coords": (37.3750, -122.0500),
        "start_hour": 9.0,
        "end_hour": 15.5,
        "budget_usd": 110.0,
        "tags": ["botanical", "scenic-walk", "farm-to-table", "quiet-seating"],
    },
    {
        "id": "scn-03-tight-budget-local",
        "start_coords": (37.3820, -122.0750),
        "start_hour": 8.0,
        "end_hour": 14.0,
        "budget_usd": 88.0,
        "tags": ["single-origin", "local-artisan", "vegetarian-friendly"],
    },
    {
        "id": "scn-04-afternoon-corridor",
        "start_coords": (37.3600, -122.0200),
        "start_hour": 9.5,
        "end_hour": 16.0,
        "budget_usd": 105.0,
        "tags": ["artisan-coffee", "botanical", "local-merchant"],
    },
    {
        "id": "scn-05-full-day-discovery",
        "start_coords": (37.3880, -122.0800),
        "start_hour": 8.5,
        "end_hour": 16.5,
        "budget_usd": 120.0,
        "tags": [
            "local-roaster",
            "scenic-walk",
            "farm-to-table",
            "vegetarian-friendly",
        ],
    },
]

BASELINE_MODEL = "claude-opus-4-high"
BASELINE_TURNS_PER_SCENARIO = 4


def _seed_worker_calibration(ledger: ModelROILedger) -> None:
    """Seeds history so the router compares a frontier and a flash worker model."""
    for session_id, model, in_tok, cache_tok, quality, ttft in (
        ("seed-calib-opus", "claude-opus-4-high", 400, 2400, 0.98, 2.10),
        ("seed-calib-flash", "gemini-2.5-flash", 300, 2650, 0.99, 0.60),
    ):
        ledger.record(
            InvocationTelemetry(
                session_id=session_id,
                task_category=WORKER_TASK,
                role="worker",
                model=model,
                input_tokens=in_tok,
                output_tokens=200,
                cache_read_tokens=cache_tok,
                total_steps=3,
                constraint_score=quality,
                ttft_s=ttft,
            )
        )


def run_benchmark() -> Dict[str, Any]:
    catalog = default_sample_catalog()

    # 1. Baseline: every turn on the frontier model, prefix rewritten each turn.
    baseline_ledger = ModelROILedger()
    for scn in SCENARIOS:
        for _ in range(BASELINE_TURNS_PER_SCENARIO):
            baseline_ledger.record(
                InvocationTelemetry(
                    session_id=scn["id"],
                    task_category="monolithic_itinerary_planning",
                    role="coordinator",
                    model=BASELINE_MODEL,
                    input_tokens=2950,
                    output_tokens=220,
                    cache_read_tokens=450,
                    total_steps=3,
                    constraint_score=0.80,
                    ttft_s=2.40,
                )
            )
    base = baseline_ledger.summarize_by_task()["monolithic_itinerary_planning"][0]
    base_total_cost = round(base.avg_weighted_cost_per_turn * base.invocations, 2)

    # 2. Coordinator + workers with frozen memory prefix and ROI routing.
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ModelROILedger(ledger_path=str(Path(tmpdir) / "roi_ledger.json"))
        _seed_worker_calibration(ledger)
        coordinator = ItineraryDiscoveryCoordinator(ledger=ledger)
        episodes = [
            coordinator.run_episode(
                session_id=scn["id"],
                start_coords=scn["start_coords"],
                start_hour=scn["start_hour"],
                end_hour=scn["end_hour"],
                budget_usd=scn["budget_usd"],
                user_intent_tags=scn["tags"],
                catalog=catalog,
                decision_dir=tmpdir,
            )
            for scn in SCENARIOS
        ]

    # Exclude the seeded calibration rows so only episode traffic is counted.
    episode_ledger = ModelROILedger()
    for r in ledger.records:
        if not r.session_id.startswith("seed-calib-"):
            episode_ledger.record(r)
    summaries = episode_ledger.summarize_by_task()
    coord = summaries[COORDINATOR_TASK][0]
    worker = summaries[WORKER_TASK][0]

    prop_total_cost = round(
        coord.avg_weighted_cost_per_turn * coord.invocations
        + worker.avg_weighted_cost_per_turn * worker.invocations,
        2,
    )
    prop_hit_rate = ModelROILedger.compute_cache_hit_rate(
        coord.input_tokens + worker.input_tokens,
        coord.cache_read_tokens + worker.cache_read_tokens,
    )

    completed = sum(
        1 for e in episodes if e["stops_count"] == 3 and e["budget_satisfied"]
    )
    total_stops = sum(e["stops_count"] for e in episodes)
    long_tail_ratio = round(
        sum(e["long_tail_merchants_selected"] for e in episodes)
        / float(max(1, total_stops)),
        4,
    )
    cost_reduction_pct = round(
        (1.0 - prop_total_cost / max(0.01, base_total_cost)) * 100.0, 2
    )

    return {
        "benchmark": "itinerary_roi_bench",
        "token_counts": "simulated",
        "scenarios_evaluated": len(SCENARIOS),
        "scenarios_completed": f"{completed}/{len(SCENARIOS)}",
        "baseline_monolithic": {
            "model": base.model,
            "cache_hit_rate": base.cache_hit_rate,
            "total_weighted_cost": base_total_cost,
            "roi_score": base.roi_score,
        },
        "coordinator_worker_with_roi_ledger": {
            "coordinator_model": coord.model,
            "routed_worker_model": worker.model,
            "overall_cache_hit_rate": prop_hit_rate,
            "total_weighted_cost": prop_total_cost,
            "worker_roi_score": worker.roi_score,
            "long_tail_discovery_ratio": long_tail_ratio,
        },
        "deltas": {
            "cache_hit_rate_delta": round(prop_hit_rate - base.cache_hit_rate, 4),
            "weighted_cost_reduction_pct": cost_reduction_pct,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, help="Optional path to write the JSON report."
    )
    args = parser.parse_args()

    report = run_benchmark()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
