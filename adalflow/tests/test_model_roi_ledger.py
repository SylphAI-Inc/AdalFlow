from adalflow.optim.model_roi_ledger import (
    DecisionCard,
    DecisionOption,
    FrozenPrefixMemorySnapshot,
    InvocationTelemetry,
    ModelROILedger,
)

import pytest


def _telemetry(model: str, **kwargs) -> InvocationTelemetry:
    defaults = dict(
        session_id="s",
        task_category="worker_routing",
        role="worker",
        model=model,
        total_steps=4,
        constraint_score=0.98,
    )
    defaults.update(kwargs)
    return InvocationTelemetry(**defaults)


def test_frozen_prefix_survives_memory_writes():
    snap = FrozenPrefixMemorySnapshot({"route-mode": "scenic", "budget": "100"})
    orig_prefix, orig_sha = snap.frozen_prefix, snap.frozen_sha
    assert snap.build_tail_delta_reminder() is None

    snap.upsert_memory("Dietary", "vegetarian")
    assert snap.frozen_prefix == orig_prefix
    assert snap.frozen_sha == orig_sha
    assert snap.entries["dietary"] == "vegetarian"

    delta = snap.build_tail_delta_reminder()
    assert delta is not None and "dietary='vegetarian'" in delta
    assert snap.build_tail_delta_reminder() is None


def test_tail_reminder_reports_only_new_changes():
    snap = FrozenPrefixMemorySnapshot()
    snap.upsert_memory("a", "1")
    assert "a='1'" in snap.build_tail_delta_reminder()

    snap.upsert_memory("b", "2")
    delta = snap.build_tail_delta_reminder()
    assert "b='2'" in delta
    assert "a='1'" not in delta


def test_decision_card_roundtrip_and_answer(tmp_path):
    path = str(tmp_path / "decision.json")
    card = DecisionCard(
        decision_id="dec-01",
        thread_name="route-worker",
        question="Pick route option?",
        options=[
            DecisionOption(key="A", label="Option A", recommended=True),
            DecisionOption(key="B", label="Option B"),
        ],
    )
    card.save(path)
    loaded = DecisionCard.load(path)
    assert not loaded.is_answered

    with pytest.raises(ValueError):
        loaded.answer("Z")
    loaded.answer("A", path=path)

    reloaded = DecisionCard.load(path)
    assert reloaded.selected_key == "A"
    assert reloaded.is_answered


def test_cache_hit_rate_and_weighted_cost():
    assert ModelROILedger.compute_cache_hit_rate(0, 0) == 0.0
    assert ModelROILedger.compute_cache_hit_rate(250, 750) == 0.75

    ledger = ModelROILedger(model_tier_weights={"m": 2.0})
    # (1000 + 0.1 * 1000 + 3 * 100) / 1000 * 2.0
    assert ledger.compute_weighted_cost("m", 1000, 100, 1000) == 2.8
    assert ledger.compute_weighted_cost("unknown", 1000, 0, 0) == 1.0


def test_ledger_recommends_high_roi_model_and_persists(tmp_path):
    path = str(tmp_path / "ledger.json")
    ledger = ModelROILedger(ledger_path=path)
    ledger.record(
        _telemetry(
            "claude-opus-4-high",
            input_tokens=1500,
            output_tokens=300,
            cache_read_tokens=1500,
            ttft_s=2.5,
        )
    )
    ledger.record(
        _telemetry(
            "gemini-2.5-flash",
            input_tokens=300,
            output_tokens=200,
            cache_read_tokens=2700,
            ttft_s=0.5,
        )
    )
    assert (
        ledger.recommend_model("worker_routing", min_quality=0.95) == "gemini-2.5-flash"
    )

    reloaded = ModelROILedger(ledger_path=path)
    assert len(reloaded.records) == 2
    assert (
        reloaded.recommend_model("worker_routing", min_quality=0.95)
        == "gemini-2.5-flash"
    )


def test_recommend_model_quality_floor_and_fallback():
    ledger = ModelROILedger()
    assert ledger.recommend_model("unseen", fallback_model="fb") == "fb"

    ledger.record(
        _telemetry("cheap", input_tokens=100, constraint_score=0.5, ttft_s=0.1)
    )
    ledger.record(_telemetry("good", input_tokens=5000, output_tokens=2000, ttft_s=3.0))
    # "cheap" has the higher ROI but fails the quality floor.
    assert ledger.summarize_by_task()["worker_routing"][0].model == "cheap"
    assert ledger.recommend_model("worker_routing", min_quality=0.9) == "good"
