"""
Coordinator + worker itinerary planning and local-merchant discovery demo.

Shows how ``ModelROILedger``, ``FrozenPrefixMemorySnapshot`` and ``DecisionCard``
(``adalflow.optim.model_roi_ledger``) fit together in a multi-agent loop:

- A coordinator holds the user's memory as a frozen prompt prefix and dispatches
  scoped workers (merchant discovery, route/time-window solving).
- Worker models are picked per task category by ``ModelROILedger.recommend_model``.
- A long-tail vs. mainstream merchant trade-off is parked on a ``DecisionCard``;
  the resolved preference is written back to memory as a tail reminder, leaving
  the frozen prefix untouched.

This is a deterministic, local prototype: ranking, routing and constraint checks
run for real, but no LLM is called. The token counts recorded in the ledger are
fixed, simulated values that stand in for provider usage reports.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADALFLOW_PKG = PROJECT_ROOT / "adalflow"
if str(ADALFLOW_PKG) not in sys.path:
    sys.path.insert(0, str(ADALFLOW_PKG))

from adalflow.optim.model_roi_ledger import (  # noqa: E402
    DecisionCard,
    DecisionOption,
    FrozenPrefixMemorySnapshot,
    InvocationTelemetry,
    ModelROILedger,
)

WORKER_TASK = "itinerary_worker_constraint_check"
COORDINATOR_TASK = "itinerary_coordinator_planning"
URBAN_SPEED_KMH = 28.0
LONG_TAIL_REVIEW_THRESHOLD = 100


@dataclass
class MerchantCandidate:
    """A local commercial POI or specialty merchant."""

    merchant_id: str
    name: str
    category: str
    lat: float
    lon: float
    open_hour: int
    close_hour: int
    avg_spend_usd: float
    dwell_hours: float
    popularity_reviews: int
    entity_tags: List[str]
    semantic_relevance: float
    landmark_cue: str

    @property
    def is_long_tail(self) -> bool:
        return self.popularity_reviews < LONG_TAIL_REVIEW_THRESHOLD


@dataclass
class ItineraryStop:
    """A scheduled stop in a verified multi-stop itinerary."""

    order: int
    merchant: MerchantCandidate
    arrival_hour: float
    departure_hour: float
    travel_km_from_prev: float
    navigation_instruction: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two coordinates."""
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return round(2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 2)


def selective_long_tail_score(
    candidate: MerchantCandidate,
    user_tags: List[str],
    selective_regularization: bool = True,
) -> float:
    """Merchant utility with semantic entity grounding applied selectively.

    Long-tail merchants (few reviews) lean on semantic/tag relevance, while head
    merchants keep their popularity prior. With ``selective_regularization=False``
    both signals are blended uniformly for every merchant.
    """
    pop_prior = min(1.0, math.log10(max(10, candidate.popularity_reviews)) / 3.5)
    tag_overlap = len(set(candidate.entity_tags) & set(user_tags)) / float(
        max(1, len(user_tags))
    )
    sem_score = 0.65 * candidate.semantic_relevance + 0.35 * tag_overlap
    if not selective_regularization:
        return round(0.50 * pop_prior + 0.50 * sem_score, 4)
    if candidate.is_long_tail:
        return round(0.25 * pop_prior + 0.75 * sem_score, 4)
    return round(0.70 * pop_prior + 0.30 * sem_score, 4)


class ItineraryDiscoveryCoordinator:
    """Coordinator + worker planner with a frozen memory prefix and ROI routing."""

    def __init__(
        self,
        ledger: ModelROILedger,
        initial_memory: Optional[Dict[str, str]] = None,
        coordinator_model: str = "claude-sonnet-4-high",
    ) -> None:
        self.ledger = ledger
        self.memory = FrozenPrefixMemorySnapshot(
            initial_memory
            or {
                "dietary-preference": "artisan coffee, vegetarian-friendly local cuisine",
                "budget-cap-usd": "120.0",
                "mobility-mode": "urban driving with visual landmark cues",
            }
        )
        self.coordinator_model = coordinator_model

    def plan_stops(
        self,
        ranked: List[MerchantCandidate],
        start_coords: Tuple[float, float],
        start_hour: float,
        end_hour: float,
        budget_usd: float,
        max_stops: int = 3,
    ) -> List[ItineraryStop]:
        """Greedily schedules ranked merchants under budget and time windows."""
        stops: List[ItineraryStop] = []
        cur_lat, cur_lon = start_coords
        cur_hour = start_hour
        spent = 0.0
        for cand in ranked:
            if len(stops) >= max_stops:
                break
            if spent + cand.avg_spend_usd > budget_usd:
                continue
            dist_km = haversine_km(cur_lat, cur_lon, cand.lat, cand.lon)
            arr_h = max(cur_hour + dist_km / URBAN_SPEED_KMH, float(cand.open_hour))
            dep_h = round(arr_h + cand.dwell_hours, 2)
            if dep_h > min(end_hour, float(cand.close_hour)):
                continue
            stops.append(
                ItineraryStop(
                    order=len(stops) + 1,
                    merchant=cand,
                    arrival_hour=round(arr_h, 2),
                    departure_hour=dep_h,
                    travel_km_from_prev=dist_km,
                    navigation_instruction=(
                        f"Proceed {dist_km:.1f} km; turn toward {cand.name} "
                        f"({cand.landmark_cue})."
                    ),
                )
            )
            cur_lat, cur_lon = cand.lat, cand.lon
            cur_hour = dep_h
            spent = round(spent + cand.avg_spend_usd, 2)
        return stops

    def run_episode(
        self,
        session_id: str,
        start_coords: Tuple[float, float],
        start_hour: float,
        end_hour: float,
        budget_usd: float,
        user_intent_tags: List[str],
        catalog: List[MerchantCandidate],
        decision_dir: Optional[str] = None,
        auto_answer_key: str = "A",
    ) -> Dict[str, Any]:
        """Runs one coordinator + 2-worker episode and logs (simulated) telemetry."""
        worker_model = self.ledger.recommend_model(
            WORKER_TASK, min_quality=0.90, fallback_model="gemini-2.5-flash"
        )

        # Turn 1 (coordinator): frozen memory prefix, dispatch workers.
        self.ledger.record(
            InvocationTelemetry(
                session_id=session_id,
                task_category=COORDINATOR_TASK,
                role="coordinator",
                model=self.coordinator_model,
                input_tokens=420,
                output_tokens=180,
                cache_read_tokens=3400,
                total_steps=2,
                constraint_score=1.0,
                ttft_s=1.45,
            )
        )

        # Worker 1: long-tail-aware merchant discovery.
        ranked = sorted(
            catalog,
            key=lambda c: selective_long_tail_score(c, user_intent_tags),
            reverse=True,
        )
        self.ledger.record(
            InvocationTelemetry(
                session_id=f"{session_id}-merchant-worker",
                task_category=WORKER_TASK,
                role="worker",
                model=worker_model,
                input_tokens=310,
                output_tokens=210,
                cache_read_tokens=2650,
                total_steps=3,
                constraint_score=1.0,
                ttft_s=0.62,
            )
        )

        # Worker 2: route geometry and time-window solving.
        stops = self.plan_stops(ranked, start_coords, start_hour, end_hour, budget_usd)
        spent_usd = round(sum(s.merchant.avg_spend_usd for s in stops), 2)
        self.ledger.record(
            InvocationTelemetry(
                session_id=f"{session_id}-route-worker",
                task_category=WORKER_TASK,
                role="worker",
                model=worker_model,
                input_tokens=290,
                output_tokens=195,
                cache_read_tokens=2650,
                total_steps=3,
                constraint_score=1.0 if len(stops) == 3 else 0.75,
                ttft_s=0.58,
            )
        )

        # Park the long-tail vs. mainstream trade-off on a non-expiring card.
        decision_card = DecisionCard(
            decision_id=f"dec-{session_id[-6:]}",
            thread_name="cold-start-merchant-worker",
            question=(
                "Include specialty long-tail merchants (few reviews, high entity "
                "match) or mainstream chains only?"
            ),
            options=[
                DecisionOption(
                    key="A",
                    label="Include specialty local merchants",
                    detail="Stays within budget and favors long-tail discovery.",
                    recommended=True,
                ),
                DecisionOption(
                    key="B",
                    label="Use high-review chain stores only",
                    detail="Lower novelty, higher popularity prior.",
                ),
            ],
            context=f"Total planned spend ${spent_usd:.2f} / ${budget_usd:.2f}",
        )
        decision_path = None
        if decision_dir:
            decision_path = os.path.join(
                decision_dir, f"{decision_card.decision_id}.json"
            )
            decision_card.save(decision_path)
        # A real deployment would resume once a human answers the card; the demo
        # answers it immediately so the episode stays deterministic.
        decision_card.answer(auto_answer_key, path=decision_path)

        # Record the preference without mutating the frozen prefix.
        self.memory.upsert_memory(
            "long-tail-merchant-policy",
            f"Selected option {decision_card.selected_key}: "
            "prioritize entity-matched local merchants.",
        )
        tail_delta = self.memory.build_tail_delta_reminder()

        # Turn 2 (coordinator): verify deliverables and write the recap.
        self.ledger.record(
            InvocationTelemetry(
                session_id=session_id,
                task_category=COORDINATOR_TASK,
                role="coordinator",
                model=self.coordinator_model,
                input_tokens=260,
                output_tokens=150,
                cache_read_tokens=3820,
                total_steps=1,
                constraint_score=1.0,
                ttft_s=1.32,
            )
        )

        total_km = round(sum(s.travel_km_from_prev for s in stops), 2)
        long_tail_included = sum(1 for s in stops if s.merchant.is_long_tail)
        return {
            "session_id": session_id,
            "coordinator_model": self.coordinator_model,
            "worker_model": worker_model,
            "stops_count": len(stops),
            "long_tail_merchants_selected": long_tail_included,
            "total_distance_km": total_km,
            "total_spend_usd": spent_usd,
            "budget_satisfied": spent_usd <= budget_usd,
            "frozen_prefix_sha": self.memory.frozen_sha,
            "tail_delta_reminder": tail_delta,
            "itinerary": [
                {
                    "order": s.order,
                    "name": s.merchant.name,
                    "category": s.merchant.category,
                    "is_long_tail": s.merchant.is_long_tail,
                    "arrival_hour": s.arrival_hour,
                    "departure_hour": s.departure_hour,
                    "spend_usd": s.merchant.avg_spend_usd,
                    "navigation_instruction": s.navigation_instruction,
                }
                for s in stops
            ],
            "recap": (
                f"Recap: {len(stops)}-stop itinerary "
                f"(${spent_usd:.2f}/${budget_usd:.2f}, {total_km:.1f} km, "
                f"{long_tail_included} long-tail merchants) using worker={worker_model}."
            ),
        }


def default_sample_catalog() -> List[MerchantCandidate]:
    """A small deterministic catalog mixing head and long-tail local merchants."""
    return [
        MerchantCandidate(
            merchant_id="m1",
            name="Kindred Craft Coffee Roasters",
            category="coffee_roaster",
            lat=37.3861,
            lon=-122.0839,
            open_hour=8,
            close_hour=16,
            avg_spend_usd=14.5,
            dwell_hours=0.75,
            popularity_reviews=28,
            entity_tags=[
                "artisan-coffee",
                "single-origin",
                "local-roaster",
                "quiet-seating",
            ],
            semantic_relevance=0.96,
            landmark_cue="just past the red-brick clocktower on Castro St",
        ),
        MerchantCandidate(
            merchant_id="m2",
            name="Botanical Courtyard & Tea",
            category="cultural_garden",
            lat=37.3688,
            lon=-122.0363,
            open_hour=9,
            close_hour=18,
            avg_spend_usd=22.0,
            dwell_hours=1.5,
            popularity_reviews=45,
            entity_tags=["botanical", "scenic-walk", "local-artisan", "quiet-seating"],
            semantic_relevance=0.92,
            landmark_cue="turn right after the glass conservatory archway",
        ),
        MerchantCandidate(
            merchant_id="m3",
            name="Verde Kitchen & Seasonal Farm Table",
            category="vegetarian_bistro",
            lat=37.3541,
            lon=-121.9552,
            open_hour=11,
            close_hour=21,
            avg_spend_usd=48.0,
            dwell_hours=1.25,
            popularity_reviews=64,
            entity_tags=["vegetarian-friendly", "farm-to-table", "local-merchant"],
            semantic_relevance=0.95,
            landmark_cue="opposite the historic mission bell plaza",
        ),
        MerchantCandidate(
            merchant_id="m4",
            name="Highway Chain Diner #402",
            category="fast_casual",
            lat=37.3910,
            lon=-122.0700,
            open_hour=6,
            close_hour=23,
            avg_spend_usd=25.0,
            dwell_hours=1.0,
            popularity_reviews=2400,
            entity_tags=["drive-thru", "standard-menu"],
            semantic_relevance=0.42,
            landmark_cue="at the highway exit neon sign",
        ),
    ]


if __name__ == "__main__":
    coordinator = ItineraryDiscoveryCoordinator(ledger=ModelROILedger())
    result = coordinator.run_episode(
        session_id="demo-itinerary-001",
        start_coords=(37.3900, -122.0810),
        start_hour=9.0,
        end_hour=15.0,
        budget_usd=100.0,
        user_intent_tags=[
            "artisan-coffee",
            "local-roaster",
            "vegetarian-friendly",
            "scenic-walk",
        ],
        catalog=default_sample_catalog(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
