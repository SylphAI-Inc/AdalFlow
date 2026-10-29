"""Lineage x feedback factorial: where does evolution's edge over best-of-N come from?

Two factors, each run with ``evolve`` at the same evaluation budget:

- lineage: ``seed`` (every proposal rewrites the seed, ``selection="resample"``) or
  ``parent`` (rewrites the greedy best so far, ``selection="best"``).
- feedback: what the proposer sees about the candidate it rewrites. ``none`` (only
  the candidate), ``score`` (its aggregate score) or ``full`` (score, per-instance
  results and the evaluator's failing cases, the default ``LLMProposer`` view).

``--show`` sets how many failing cases the ``duration`` evaluator reveals per
evaluation (the dose; ``0`` reveals all). Inspiration candidates are switched off in
every cell, so ``parent`` differs from ``seed`` only in which candidate is rewritten.

    python benchmarks/evolution/factorial.py --check          # fake proposer, no model
    python benchmarks/evolution/factorial.py --provider anthropic --model claude-haiku-5-5 \
        --tasks duration --repeats 5 --budget 12 --out runs/factorial

Each run logs its events to ``<out>/<cell>-<repeat>.jsonl`` and its candidates, with
the cumulative proposer tokens at each one, to ``summary.json``. ``analyze`` (run at
the end, or alone with ``--analyze <out>``) reports held-out best per cell at equal
evaluations and at several equal-token cutoffs, tokens to reach the stop score, the lineage effect within each feedback level and
the lineage x feedback interaction, with bootstrap 95% intervals over runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adalflow"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adalflow.optim.evolution import (  # noqa: E402
    Evaluation,
    LLMProposer,
    Proposal,
    PythonEvaluator,
    evolve,
    to_evaluation,
)
from tasks import TASKS, Task  # noqa: E402

LINEAGE = {"seed": "resample", "parent": "best"}
FEEDBACK = ("none", "score", "full")


def with_show(task: Task, show: int) -> Task:
    """``duration`` revealing ``show`` failing cases per evaluation (0: all of them)."""
    if task.name != "duration" or show == 4:
        return task
    marker = "SHOW = 4"
    assert marker in task.search_evaluator, "duration evaluator changed"
    source = task.search_evaluator.replace(marker, f"SHOW = {show or 10_000}")
    return dataclasses.replace(task, search_evaluator=source)


def masked(candidate, feedback: str):
    """The candidate as the proposer should see it under ``feedback``."""
    if feedback == "full" or candidate.evaluation is None:
        return candidate
    ev = candidate.evaluation
    if feedback == "score":
        view = Evaluation(score=ev.score, error="failed" if ev.error else None)
    else:
        view = Evaluation()
    return dataclasses.replace(candidate, evaluation=view, samples=[])


class TokenMeter:
    """Adds up prompt and completion tokens over a generator's calls."""

    def __init__(self, generator: Any) -> None:
        self.generator, self.total, self.estimated = generator, 0, False
        self._lock = threading.Lock()

    def call(self, prompt_kwargs: Dict[str, Any], **kwargs: Any) -> Any:
        out = self.generator.call(prompt_kwargs=prompt_kwargs, **kwargs)
        usage = getattr(out, "usage", None)
        used = getattr(usage, "total_tokens", None) or (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0)
        )
        if not used:  # provider reported nothing: estimate from characters
            prompt = self.generator.get_prompt(prompt_kwargs=prompt_kwargs)
            used = (len(prompt) + len(str(getattr(out, "data", "") or ""))) // 4
            self.estimated = True
        with self._lock:
            self.total += used
        return out


CLUSTER_PROMPT = (
    "Below are all the wrong answers a system prompt produced on its examples, one "
    "per line. Group them into 3 to 7 failure patterns (fewer if there are fewer "
    "wrong answers). For each pattern write one line: what goes wrong, how many of "
    "the wrong answers it covers, and one representative wrong answer copied "
    "verbatim. Reply with the patterns only.\n\n{failures}"
)
# Stage-10 sensitivity arm: same, but the representative keeps its input and
# expected answer (ESPO's error set is (input, gold, prediction)).
CLUSTER_GOLD_PROMPT = CLUSTER_PROMPT.replace(
    "one representative wrong answer copied verbatim",
    "one representative failure copied verbatim with its input, the wrong answer "
    "and the expected answer in full",
)


def summarize(generator: Any, failures: str, gold: bool = False) -> str:
    """ESPO-style diagnosis: every failure in, 3 to 7 patterns out."""
    if not failures:
        return failures
    prompt = CLUSTER_GOLD_PROMPT if gold else CLUSTER_PROMPT
    out = generator.call(prompt_kwargs={"input_str": prompt.format(failures=failures)})
    return str(out.data if out.data is not None else "")


class FactorialProposer:
    """Wraps a proposer: masks the parent's feedback and drops the inspiration.

    With ``clusterer``, the parent's feedback (all its failures) is replaced by
    ``summarize``'s patterns, once per candidate.
    """

    def __init__(
        self,
        inner: Any,
        feedback: str,
        meter: Optional[TokenMeter],
        clusterer: Any = None,
        gold: bool = False,
    ) -> None:
        self.inner, self.feedback, self.meter = inner, feedback, meter
        self.clusterer, self._summaries, self.gold = clusterer, {}, gold

    def propose(self, parent, inspiration=None) -> Proposal:
        view = masked(parent, self.feedback)
        if self.clusterer is not None and view.evaluation is not None:
            if parent.id not in self._summaries:
                self._summaries[parent.id] = summarize(
                    self.clusterer, view.evaluation.feedback, self.gold
                )
            ev = dataclasses.replace(
                view.evaluation, feedback=self._summaries[parent.id]
            )
            view = dataclasses.replace(view, evaluation=ev)
        return self.inner.propose(view, None)


class FakeProposer:
    """No model: appends a numbered comment, so plumbing can be checked for free."""

    def __init__(self) -> None:
        self.count = 0
        self.prompt_chars: List[int] = []

    def propose(self, parent, inspiration=None) -> Proposal:
        from adalflow.optim.evolution.proposer import render_feedback

        self.count += 1
        self.prompt_chars.append(len(parent.text) + len(render_feedback(parent)))
        return Proposal(text=parent.text.rstrip("\n") + f"\n# variant {self.count}\n")


def make_client(provider: str):
    if provider == "anthropic":
        from adalflow.components.model_client import AnthropicAPIClient

        return AnthropicAPIClient()
    from adalflow.components.model_client import OpenAIClient

    return OpenAIClient()


def cells(feedback: List[str], shows: List[int]) -> List[Tuple[str, str, int]]:
    """(lineage, feedback, show): the 2 x 3 grid at show 4, plus the dose at full."""
    grid = [(lin, fb, 4) for lin in LINEAGE for fb in feedback]
    dose = [(lin, "full", s) for lin in LINEAGE for s in shows if s != 4]
    return grid + (dose if "full" in feedback else [])


def cell_name(task: str, lineage: str, feedback: str, show: int) -> str:
    return f"{task}-{lineage}-{feedback}-k{show if show else 'all'}"


class FakeClusterer:
    def call(self, prompt_kwargs, **kwargs):
        from adalflow.core.types import GeneratorOutput

        return GeneratorOutput(data="pattern: everything (1 wrong answer)")


def one_run(
    args,
    task_name: str,
    cell: Tuple[str, str, int],
    repeat: int,
    cluster=False,
    gold=False,
):
    lineage, feedback, show = cell
    task = with_show(TASKS[task_name], show)
    meter = clusterer = None
    if args.check:
        inner = FakeProposer()
        clusterer = FakeClusterer() if cluster else None
    else:
        inner = LLMProposer(
            task.objective,
            model_client=make_client(args.provider),
            model_kwargs={"model": args.model},
        )
        meter = inner.generator = TokenMeter(inner.generator)
        if cluster:
            from adalflow.core.generator import Generator

            clusterer = TokenMeter(
                Generator(
                    model_client=make_client(args.provider),
                    model_kwargs={"model": args.model},
                    use_cache=False,
                )
            )
    tokens_at: Dict[str, int] = {}

    def total() -> int:
        return (meter.total if meter else 0) + getattr(clusterer, "total", 0)

    class Metered(FactorialProposer):
        def propose(self, parent, inspiration=None) -> Proposal:
            try:
                return super().propose(parent, inspiration)
            finally:
                # tokens spent up to and including this proposal
                tokens_at[f"after-{len(tokens_at) + 1}"] = total()

    name = cell_name(task_name, lineage, feedback, show)
    log_path = args.out / f"{name}-{repeat}.jsonl"
    log_path.unlink(missing_ok=True)
    res = evolve(
        task.seed,
        PythonEvaluator(task.search_evaluator, timeout=60),
        Metered(inner, feedback, meter, clusterer, gold),
        max_evaluations=args.budget,
        selection=LINEAGE[lineage],
        random_seed=repeat,
        stop_score=task.stop_score,
        max_stalls=args.budget,
        log_path=log_path,
        holdout=PythonEvaluator(task.holdout_evaluator, timeout=60),
    )
    # proposal i (1-based) produced candidate c<j> or a duplicate/failure; candidates
    # appear in archive order, so map them to the tokens spent when they were made.
    attempts = [
        e
        for e in res.events
        if e["event"] in ("candidate", "duplicate", "proposal_failure")
    ]
    spent, n, candidates = 0, 0, []
    for e in attempts:
        if e["event"] == "candidate" and e.get("origin") == "seed":
            candidates.append({"id": e["id"], "tokens": 0})
            continue
        n += 1
        spent = tokens_at.get(f"after-{n}", spent)
        if e["event"] == "candidate":
            candidates.append({"id": e["id"], "tokens": spent})
    by_id = {c.id: c for c in res.archive.candidates}
    for c in candidates:
        cand = by_id[c["id"]]
        c.update(score=cand.score, text=cand.text)
    return {
        "task": task_name,
        "lineage": lineage,
        "feedback": feedback,
        "show": show,
        "cell": name,
        "repeat": repeat,
        "evaluations": res.evaluations,
        "duplicates": res.duplicates,
        "proposal_failures": res.proposal_failures,
        "stopped": res.stopped,
        "tokens": total(),
        "cluster_tokens": getattr(clusterer, "total", 0),
        "tokens_estimated": bool(meter and meter.estimated),
        "prompt_chars": getattr(inner, "prompt_chars", None),
        "holdout_seed": res.holdout["seed"].score,
        "holdout_best": res.holdout["best"].score if "best" in res.holdout else None,
        "candidates": candidates,
    }


# ------------------------------------------------------------------ analysis
def _boot(
    values: List[float], rng: random.Random, n: int = 2000
) -> Tuple[float, float]:
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _fmt(values: List[float], rng: random.Random) -> str:
    if not values:
        return "-"
    lo, hi = _boot(values, rng)
    return f"{statistics.mean(values):.3f} [{lo:.3f}, {hi:.3f}]"


def solved(row: Dict[str, Any]) -> bool:
    """The run reached its task's ``stop_score`` on the search set (and stopped)."""
    stop = TASKS[row["task"]].stop_score
    scores = [c["score"] for c in row["candidates"] if c["score"] is not None]
    return stop is not None and bool(scores) and max(scores) >= stop


def tokens_to_solve(row: Dict[str, Any]) -> Optional[int]:
    stop = TASKS[row["task"]].stop_score
    for c in row["candidates"]:
        if stop is not None and c["score"] is not None and c["score"] >= stop:
            return c["tokens"]
    return None


def holdout_at_tokens(
    rows: List[Dict[str, Any]], budget: int, holdout_cache: Dict
) -> None:
    """Adds ``holdout@<budget>``: held-out score of the search-best made within
    ``budget`` tokens. A run that stopped at ``stop_score`` keeps its best beyond
    its own spend; a run that ran out of evaluations before ``budget`` gets None,
    since with more evaluations it would have kept searching."""
    for r in rows:
        key_name = f"holdout@{budget}"
        made = [
            c
            for c in r["candidates"]
            if c["tokens"] <= budget and c["score"] is not None
        ]
        if not made or (r["tokens"] < budget and not solved(r)):
            r[key_name] = None
            continue
        best = max(made, key=lambda c: c["score"])
        key = (r["task"], best["text"])
        if key not in holdout_cache:
            ev = PythonEvaluator(TASKS[r["task"]].holdout_evaluator, timeout=60)(
                best["text"]
            )
            holdout_cache[key] = to_evaluation(ev).score
        r[key_name] = holdout_cache[key]


def _diff(a: List[float], b: List[float], rng: random.Random, n: int = 2000) -> str:
    if not a or not b:
        return "-"
    d = sorted(
        statistics.mean(rng.choices(a, k=len(a)))
        - statistics.mean(rng.choices(b, k=len(b)))
        for _ in range(n)
    )
    point = statistics.mean(a) - statistics.mean(b)
    return f"{point:+.3f} [{d[int(0.025 * n)]:+.3f}, {d[int(0.975 * n) - 1]:+.3f}]"


def _interaction(
    cells_: Dict, f1: str, f2: str, rng: random.Random, n: int = 2000
) -> str:
    """(parent - seed) under f1 minus (parent - seed) under f2."""
    parts = [
        cells_.get(k, [])
        for k in (("parent", f1), ("seed", f1), ("parent", f2), ("seed", f2))
    ]
    if not all(parts):
        return "-"
    m = lambda v: statistics.mean(rng.choices(v, k=len(v)))  # noqa: E731
    d = sorted(m(parts[0]) - m(parts[1]) - m(parts[2]) + m(parts[3]) for _ in range(n))
    point = (statistics.mean(parts[0]) - statistics.mean(parts[1])) - (
        statistics.mean(parts[2]) - statistics.mean(parts[3])
    )
    return f"{point:+.3f} [{d[int(0.025 * n)]:+.3f}, {d[int(0.975 * n) - 1]:+.3f}]"


def _fmt_n(values: List[Optional[float]], rng: random.Random) -> str:
    """Mean [interval] (n) over the runs that have a value."""
    kept = [v for v in values if v is not None]
    return f"{_fmt(kept, rng)} ({len(kept)})" if kept else "-"


def analyze(
    rows: List[Dict[str, Any]], seed: int = 0, cutoffs: Optional[List[int]] = None
) -> str:
    rng = random.Random(seed)
    out: List[str] = []
    cache: Dict = {}
    cutoffs = cutoffs or [10_000, 20_000, 30_000]
    for task in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == task]
        names = sorted({r["cell"] for r in tr})
        spend = {
            c: statistics.mean(r["tokens"] for r in tr if r["cell"] == c) for c in names
        }
        for t in cutoffs:
            holdout_at_tokens(tr, t, cache)
        heads = " | ".join(f"held-out @ {t // 1000}k tok" for t in cutoffs)
        out += [
            f"## {task}",
            "",
            "Equal tokens: held-out of the search-best made within T proposer tokens. "
            "(n) counts runs with a value at T: runs that ran out of evaluations "
            "before T are left out; runs that stopped at the stop score keep their best.",
            "",
            f"| cell | runs | tokens / run | duplicates | solved | tokens to solve (median) "
            f"| held-out best (equal evals) | {heads} |",
            "|---|---|---|---|---|---|---|" + "---|" * len(cutoffs),
        ]
        for c in names:
            g = [r for r in tr if r["cell"] == c]
            tts = [v for v in (tokens_to_solve(r) for r in g) if v is not None]
            at = " | ".join(
                _fmt_n([r[f"holdout@{t}"] for r in g], rng) for t in cutoffs
            )
            out.append(
                f"| {c} | {len(g)} | {spend[c]:.0f} "
                f"| {statistics.mean(r['duplicates'] for r in g):.1f} "
                f"| {len(tts)}/{len(g)} "
                f"| {statistics.median(tts) if tts else '-'} "
                f"| {_fmt_n([r['holdout_best'] for r in g], rng)} | {at} |"
            )
        metrics = [("holdout_best", "equal evals")] + [
            (f"holdout@{t}", f"equal tokens, T={t // 1000}k") for t in cutoffs
        ]
        for metric, label in metrics:
            at4 = {(r["lineage"], r["feedback"]): [] for r in tr if r["show"] == 4}
            for r in tr:
                if r["show"] == 4 and r[metric] is not None:
                    at4[(r["lineage"], r["feedback"])].append(r[metric])
            out += ["", f"Lineage effect (parent - seed), {label}, k=4:", ""]
            for fb in FEEDBACK:
                out.append(
                    f"- {fb}: {_diff(at4.get(('parent', fb), []), at4.get(('seed', fb), []), rng)}"
                )
            out.append(
                f"- interaction I = Δ(full) - Δ(score): {_interaction(at4, 'full', 'score', rng)}"
            )
            dose = {}
            for r in tr:
                if r["feedback"] == "full" and r[metric] is not None:
                    dose.setdefault((r["lineage"], r["show"]), []).append(r[metric])
            shows = sorted({s for _, s in dose}, key=lambda s: s or 10_000)
            if len(shows) > 1:
                out += [
                    "",
                    f"Dose (full feedback), lineage effect by failures shown, {label}:",
                    "",
                ]
                for s in shows:
                    out.append(
                        f"- k={s or 'all'}: {_diff(dose.get(('parent', s), []), dose.get(('seed', s), []), rng)}"
                    )
        out.append("")
    out.append("Mean [bootstrap 95% interval over runs].")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--tasks", nargs="+", default=["duration"], choices=TASKS)
    parser.add_argument(
        "--feedback", nargs="+", default=list(FEEDBACK), choices=FEEDBACK
    )
    parser.add_argument(
        "--shows",
        type=int,
        nargs="+",
        default=[1, 4, 0],
        help="failing cases the duration evaluator reveals (0: all); 4 is the grid",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=int, default=12, help="evaluations per run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/factorial"))
    parser.add_argument("--check", action="store_true", help="fake proposer, no model")
    parser.add_argument("--analyze", type=Path, help="only analyze <dir>/summary.json")
    args = parser.parse_args()
    if args.analyze:
        rows = json.loads((args.analyze / "summary.json").read_text())
        return print(analyze(rows))
    if not args.check and not (args.provider and args.model):
        parser.error("--provider and --model are required (or pass --check)")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for t in args.tasks:
        shows = args.shows if t == "duration" else [4]
        jobs += [
            (t, c, r) for c in cells(args.feedback, shows) for r in range(args.repeats)
        ]
    print(f"{len(jobs)} runs x up to {args.budget - 1} proposals", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: one_run(args, *job), jobs))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    report = analyze(rows)
    (args.out / "summary.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
