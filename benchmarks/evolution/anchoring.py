"""Does the sample of failing cases the evaluator reveals anchor the proposer?

In ``factorial.py`` the ``seed-full-k4`` cell never attempted ISO 8601 durations
(0/55 proposals) while ``seed-none-k4`` did in 28/55: the four failures it was shown
were all unit-suffix forms. This script keeps four revealed failures and changes
which four are revealed (``--cells`` entries are ``lineage-feedback[-reveal]``):

- ``first``: the evaluator's order (what ``factorial.py`` ran).
- ``random``: four failures drawn independently in each evaluation.
- ``iso``: the first three plus the first failing ISO 8601 case (duration only).
- ``minibatch``: GEPA-style, four cases drawn at random from all cases (passing
  ones shown as passed) in each evaluation.
- ``diverse``: the first failure, then greedily the failure whose input has the
  largest edit distance to those already picked, after mapping digits to ``9`` and
  letter runs to ``a``/``A``. Label-free: it needs only the failing inputs.

    python benchmarks/evolution/anchoring.py --check --tasks duration size
    python benchmarks/evolution/anchoring.py --provider anthropic --model claude-haiku-5-5 \
        --tasks duration --cells seed-full-random seed-full-iso --out runs/factorial/anchoring

A ``+spec`` suffix (``seed-full+spec``) appends the full format list to the
objective, to test whether anchoring needs a prior that does not cover the spec.

Runs reuse ``factorial.one_run``; each reveal is registered as its own task
(``duration@diverse``) so runs of different reveals can share a thread pool.
``--baseline`` adds rows from earlier ``summary.json`` files (``factorial.py`` or this
script) to the report. The report gives held-out best per cell and, per category of
held-out input, the share of proposals and of run-best candidates that pass every
held-out case of that category.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import factorial  # noqa: E402
from factorial import LINEAGE, PythonEvaluator, Task, to_evaluation  # noqa: E402
from tasks import DURATION_HOLDOUT, SIZE_HOLDOUT, TASKS  # noqa: E402

FIRST = "    shown = failures[:SHOW]\n"
REVEAL = {
    "first": FIRST,
    # all failures; the proposer sees factorial.summarize's patterns instead
    "cluster": "    shown = failures\n",
    "clustergold": "    shown = failures\n",
    "random": "    shown = __import__('random').sample(failures, min(SHOW, len(failures)))\n",
    "iso": (
        '    iso = [f for f in failures if "(\'P" in f]\n'
        "    shown = failures[: SHOW - 1]\n"
        "    shown += iso[:1] if iso and iso[0] not in shown else failures[SHOW - 1 : SHOW]\n"
    ),
    "diverse": r"""    re = __import__("re")

    def shape(failure):
        m = re.search(r"\((.*)\) (?:returned|raised)", failure)
        s = re.sub(r"\d+(?:\.\d+)?", "9", m.group(1) if m else failure)
        return re.sub(r"[a-z]+", "a", re.sub(r"[A-Z]+", "A", s))

    def dist(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, row[0] = row[0], i
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
        return row[-1]

    shapes = [shape(f) for f in failures]
    picked = [0] if failures else []
    while len(picked) < min(SHOW, len(failures)):
        rest = [i for i in range(len(failures)) if i not in picked]
        picked.append(
            max(rest, key=lambda i: (min(dist(shapes[i], shapes[j]) for j in picked), -i))
        )
    shown = [failures[i] for i in sorted(picked)]
""",
    "minibatch": (
        "    idx = sorted(__import__('random').sample(range(len(CASES)), SHOW))\n"
        "    shown = []\n"
        "    for i in idx:\n"
        "        hit = [f for f in failures if f.startswith(f'case-{i:02d}:')]\n"
        "        shown.append(hit[0] if hit else f'case-{i:02d}: {CASES[i][0]!r} passed')\n"
    ),
}
# Objectives that spell the spec out (the ``+spec`` cell suffix): the proposer's
# prior then covers every format, so a narrow failure sample has nothing to crowd out.
SPEC = {
    "duration": (
        " Accepted formats: a bare number of seconds; numbers (decimals allowed) "
        "with the unit suffixes s, m, h, d, w, combined with or without spaces; unit "
        "words (second, minute, hour, day, week, singular or plural); clock times "
        "M:SS and H:MM:SS; ISO 8601 durations such as P1DT2H. Invalid: empty "
        "strings, unknown units, negative values, a unit used twice, clock fields "
        "out of range, malformed numbers."
    ),
    "size": (
        " Accepted formats: a bare number of bytes, thousands separators allowed; "
        "the single letters K, M, G, T in any case as powers of 1024; the SI units "
        "kB, MB, GB, TB in any case as powers of 1000; the IEC units KiB, MiB, GiB, "
        "TiB and Ki, Mi, Gi, Ti as powers of 1024; unit words (byte, kilobyte = "
        "1000, megabyte, gigabyte, kibibyte = 1024, mebibyte, gibibyte, singular or "
        "plural); decimals and a space before the unit allowed. Invalid: empty "
        "strings, a unit without a number, negative values, unknown units, a unit "
        "used twice, malformed numbers, a number after the unit."
    ),
}
HOLDOUT = {"duration": DURATION_HOLDOUT, "size": SIZE_HOLDOUT}


def category(task: str, text: str, want: Any) -> str:
    """Format family of a held-out input, for the coverage table."""
    t = text.strip()
    if want is None:
        return "invalid"
    if task == "duration":
        if t.startswith("P"):
            return "iso"
        if ":" in t:
            return "clock"
        if re.search(r"[a-z]{3,}", t):
            return "words"
        if "w" in t or "d" in t:
            return "days/weeks"
        return "units" if re.search(r"[a-z]", t) else "number"
    if task == "size":
        if "," in t:
            return "separator"
        if re.search(r"[a-z]{4,}", t):
            return "words"
        if re.search(r"i(b)?$", t, re.I):
            return "iec"
        if re.search(r"[kmgt]b$", t, re.I):
            return "si"
        if re.search(r"[kmgt]$", t, re.I):
            return "letter"
        return "bytes"
    return "all"


def reveal_task(task: Task, reveal: str, spec: bool = False) -> str:
    """Registers ``<task>@<reveal>[+spec]`` in ``TASKS`` and returns its name."""
    name = f"{task.name}@{reveal}" + ("+spec" if spec else "")
    if name not in TASKS:
        assert FIRST in task.search_evaluator, "evaluator changed"
        source = task.search_evaluator.replace(FIRST, REVEAL[reveal])
        objective = task.objective + (SPEC[task.name] if spec else "")
        TASKS[name] = dataclasses.replace(
            task, search_evaluator=source, objective=objective
        )
    return name


def parse_cell(cell: str):
    """``lineage-feedback[-reveal][+spec]`` -> (lineage, feedback, reveal, spec)."""
    cell, spec = cell.removesuffix("+spec"), cell.endswith("+spec")
    parts = cell.split("-")
    lineage, feedback = parts[0], parts[1]
    reveal = parts[2] if len(parts) > 2 else "first"
    assert lineage in LINEAGE and feedback in factorial.FEEDBACK and reveal in REVEAL
    return lineage, feedback, "first" if feedback != "full" else reveal, spec


def run(args, task: str, spec: str, repeat: int) -> Dict[str, Any]:
    lineage, feedback, reveal, with_spec = parse_cell(spec)
    name = reveal_task(TASKS[task], reveal, with_spec)
    row = factorial.one_run(
        args,
        name,
        (lineage, feedback, 4),
        repeat,
        cluster=reveal in ("cluster", "clustergold"),
        gold=reveal == "clustergold",
    )
    log = args.out / f"{row['cell']}-{repeat}.jsonl"
    first = json.loads(log.read_text().splitlines()[0])
    key = key_of(lineage, feedback, reveal, with_spec)
    row.update(task=task, reveal=reveal, spec=with_spec, key=key)
    row["shown"] = (first.get("evaluation") or {}).get("feedback")
    return row


def key_of(lineage: str, feedback: str, reveal: str, spec: bool = False) -> str:
    key = f"{lineage}-{feedback}" + (f"-{reveal}" if feedback == "full" else "")
    return key + ("+spec" if spec else "")


def load_baseline(path: Path) -> List[Dict[str, Any]]:
    """Rows of an earlier summary at four revealed failures, keyed like this script."""
    rows = []
    for r in json.loads(path.read_text()):
        if r.get("show", 4) != 4:
            continue
        r = dict(r, task=r["task"].split("@")[0].split("+")[0])
        r.setdefault("reveal", "first")
        r["key"] = key_of(
            r["lineage"], r["feedback"], r["reveal"], r.get("spec", False)
        )
        rows.append(r)
    return rows


def report(rows: List[Dict[str, Any]], seed: int = 0) -> str:
    rng = random.Random(seed)
    out: List[str] = []
    cache: Dict = {}
    for task in sorted({r["task"] for r in rows}):
        cases = HOLDOUT[task]
        cats = sorted({category(task, t, w) for t, w in cases})
        members = {
            c: [
                f"case-{i:02d}"
                for i, (t, w) in enumerate(cases)
                if category(task, t, w) == c
            ]
            for c in cats
        }
        holdout = PythonEvaluator(TASKS[task].holdout_evaluator, timeout=60)

        def passes(text: str) -> Dict[str, float]:
            if (task, text) not in cache:
                cache[(task, text)] = to_evaluation(holdout(text)).instance_scores or {}
            return cache[(task, text)]

        def covered(text: str, c: str) -> bool:
            scores = passes(text)
            return all(scores.get(m) == 1.0 for m in members[c])

        out += [
            f"## {task}",
            "",
            "Coverage: share of proposals (first number) and of run-best candidates "
            "(second) that pass every held-out case of the category.",
            "",
            "| cell | runs | tokens / run | solved | tokens to solve (median) | held-out best | "
            + " | ".join(cats)
            + " |",
            "|---|---|---|---|---|---|" + "---|" * len(cats),
        ]
        tr = [r for r in rows if r["task"] == task]
        for key in sorted({r["key"] for r in tr}):
            g = [r for r in tr if r["key"] == key]
            props = [
                c for r in g for c in r["candidates"][1:] if c["score"] is not None
            ]
            bests = [
                max(
                    (c for c in r["candidates"] if c["score"] is not None),
                    key=lambda c: c["score"],
                )
                for r in g
            ]
            tts = [t for t in (tokens_to_solve(r) for r in g) if t is not None]
            hb = [r["holdout_best"] for r in g if r["holdout_best"] is not None]
            cov = " | ".join(
                f"{sum(covered(c['text'], k) for c in props)}/{len(props)}, "
                f"{sum(covered(b['text'], k) for b in bests)}/{len(bests)}"
                for k in cats
            )
            out.append(
                f"| {key} | {len(g)} | {statistics.mean(r['tokens'] for r in g):.0f} "
                f"| {len(tts)}/{len(g)} | {statistics.median(tts) if tts else '-'} "
                f"| {factorial._fmt(hb, rng)} | {cov} |"
            )
        shown = [r for r in tr if r.get("shown") and r["feedback"] == "full"]
        if shown:
            out += ["", "Failures shown to the seed's first rewrite:", ""]
            seen = set()
            for r in sorted(shown, key=lambda r: (r["key"], r["repeat"])):
                names = re.findall(r"\((\'[^']*\'|\"[^\"]*\")\)", r["shown"])
                line = f"- {r['key']}: {', '.join(names)}"
                if r["reveal"] in ("random", "minibatch") or line not in seen:
                    seen.add(line)
                    out.append(
                        line
                        + (
                            f" (run {r['repeat']})"
                            if r["reveal"] in ("random", "minibatch")
                            else ""
                        )
                    )
        out.append("")
    out.append("Mean [bootstrap 95% interval over runs].")
    return "\n".join(out)


def tokens_to_solve(row: Dict[str, Any]) -> Optional[int]:
    for c in row["candidates"]:
        if c["score"] is not None and c["score"] >= 1.0:
            return c["tokens"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    parser.add_argument(
        "--tasks", nargs="+", default=["duration"], choices=sorted(HOLDOUT)
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["seed-full-random", "seed-full-iso"],
        help="lineage-feedback[-reveal], e.g. seed-none parent-full-diverse",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/factorial/anchoring"))
    parser.add_argument("--baseline", type=Path, nargs="*", default=[])
    parser.add_argument("--check", action="store_true", help="fake proposer, no model")
    parser.add_argument("--report", action="store_true", help="only report --baseline")
    args = parser.parse_args()
    rows = [r for p in args.baseline for r in load_baseline(p)]
    if args.report:
        return print(report(rows))
    if not args.check and not (args.provider and args.model):
        parser.error("--provider and --model are required (or pass --check)")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (t, c, r)
        for t in args.tasks
        for c in args.cells
        if not (parse_cell(c)[2] == "iso" and t != "duration")
        for r in range(args.repeats)
    ]
    for t, c, _ in jobs:  # register before the pool starts
        reveal_task(TASKS[t], *parse_cell(c)[2:])
    print(f"{len(jobs)} runs", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        new = list(pool.map(lambda job: run(args, *job), jobs))
    (args.out / "summary.json").write_text(json.dumps(new, indent=2))
    text = report(rows + new)
    (args.out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
