"""Tests for the Harness² run importer, on a fixture mirroring Harness²'s layout.

Run from the repo root: ``pytest benchmarks/harness2_roi``.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "adalflow"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmarks.harness2_roi.import_harness2_runs import (  # noqa: E402
    build_report,
    load_namespace,
    read_codex_usage,
    to_ledger,
)

K = 1
NS = "lab-cx-agg1"


def _codex_stream(path: Path, turns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["not json: codex banner", json.dumps({"type": "thread.started"})]
    for i, c, o, r in turns:
        lines.append(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": i,
                        "cached_input_tokens": c,
                        "output_tokens": o,
                        "reasoning_output_tokens": r,
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_runs(tmp_path: Path, score=(3, 4)) -> Path:
    runs, execs = tmp_path / "runs", tmp_path / "executions"
    ns_dir = runs / NS
    task = "t1"
    base_dir = execs / NS / task / "base1"
    _codex_stream(
        base_dir / "codex_session.jsonl", [(1000, 800, 50, 10), (500, 400, 20, 0)]
    )
    for label in ("inv1", "dom1", "agg1"):
        _codex_stream(
            execs / NS / task / label / "codex_session.jsonl", [(1000, 900, 30, 0)]
        )

    (ns_dir / task).mkdir(parents=True)
    (ns_dir / "config.json").write_text(
        json.dumps({"executor": "gpt-5-codex", "improver": "vertex/gemini-3.5-flash"})
    )
    (ns_dir / task / "summary.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "k": K,
                "base": [
                    {
                        "label": "base1",
                        "run_id": f"{NS}/{task}/base1",
                        "run_dir": str(base_dir),
                    }
                ],
                "modes": {
                    "agg": {
                        "stages": [
                            {"label": "inv1", "usd": 0.01, "attempts": 1},
                            {
                                "label": "dom1",
                                "usd": 0.02,
                                "attempts": 2,
                                "fell_back": True,
                            },
                            {"label": "agg1", "usd": 0.03, "attempts": 1},
                        ],
                        "ledger": {
                            "rollouts": ["base1", "inv1", "dom1", "agg1"],
                            "usd": 0.5,
                        },
                    },
                    "seq": {
                        "stages": [],
                        "ledger": {"rollouts": ["base1"], "usd": 0.25},
                    },
                },
            }
        )
    )
    if score is not None:
        (ns_dir / "judged.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": task,
                        "mode": "agg",
                        "k": K,
                        "label": "agg1",
                        "n_passed": score[0],
                        "n_criteria": score[1],
                        "frac": score[0] / score[1],
                    }
                ]
            )
        )
    # A reflection-only mode namespace must not show up as a config.
    (runs / "lab-cx-seq1" / task / "reflection").mkdir(parents=True)
    return runs


def test_read_codex_usage_sums_turns_and_skips_junk(tmp_path):
    _codex_stream(tmp_path / "a.jsonl", [(100, 60, 5, 1), (50, 50, 5, 0)])
    usage = read_codex_usage(tmp_path)
    assert (usage.input_tokens, usage.cached_input_tokens) == (150, 110)
    assert (usage.output_tokens, usage.reasoning_output_tokens, usage.turns) == (
        10,
        1,
        2,
    )
    assert usage.uncached_input_tokens == 40


def test_load_namespace_resolves_execution_dirs(tmp_path):
    rep = load_namespace(_make_runs(tmp_path) / NS)
    assert rep.executor_model == "gpt-5-codex"
    assert [r.label for r in rep.rollouts] == ["base1", "inv1", "dom1", "agg1", "base1"]
    assert all(r.usage.turns for r in rep.rollouts)
    assert rep.scores == {("t1", "agg"): 0.75}
    dom = rep.rollouts[2]
    assert dom.failed and dom.attempts == 2 and dom.session_usd == 0.02


def test_ledger_uses_uncached_input_and_task_score(tmp_path):
    ledger = to_ledger([load_namespace(_make_runs(tmp_path) / NS)])
    (summary,) = ledger.summarize_by_task()["harness2/lab/agg"]
    assert summary.model == "gpt-5-codex"
    assert summary.invocations == 4
    # uncached = (1500 - 1200) + 3 * (1000 - 900); cached = 1200 + 3 * 900
    assert summary.input_tokens == 600
    assert summary.cache_read_tokens == 3900
    assert summary.mean_constraint_score == 0.75


def test_report_skips_ungraded_tasks_and_reflection_only_namespaces(tmp_path):
    report = build_report(_make_runs(tmp_path, score=None))
    (ns,) = report["namespaces"]
    assert ns["namespace"] == NS
    assert ns["graded"] == 0 and ns["ungraded"] == 2
    # base1 appears in both modes but is counted once.
    assert ns["tokens"]["cached_input"] == 3900
    assert ns["usd_by_mode"] == {"agg": 0.5, "seq": 0.25}
    assert report["roi_by_task"] == {}


def test_jobbench_flattened_layout_reads_model_traj(tmp_path):
    ns, task = "jb-cx-agg1", "accountant/task3"
    store = tmp_path / "jb" / "dataset" / task
    flat = lambda label: f"{ns}/{task}/{label}".replace("/", "__")  # noqa: E731
    for label in ("base1", "inv1"):
        (store / "model_output" / flat(label)).mkdir(parents=True)
        _codex_stream(
            store / "model_traj" / flat(label) / "s.jsonl", [(200, 150, 10, 0)]
        )

    ns_dir = tmp_path / "runs" / ns / task
    ns_dir.mkdir(parents=True)
    (ns_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "base": [
                    {
                        "label": "base1",
                        "run_id": f"{ns}/{task}/base1",
                        "run_dir": str(store / "model_output" / flat("base1")),
                    }
                ],
                "modes": {
                    "agg": {"stages": [], "ledger": {"rollouts": ["base1", "inv1"]}}
                },
            }
        )
    )
    rep = load_namespace(tmp_path / "runs" / ns)
    assert [
        (r.task_id, r.label, r.usage.cached_input_tokens) for r in rep.rollouts
    ] == [
        (task, "base1", 150),
        (task, "inv1", 150),
    ]
