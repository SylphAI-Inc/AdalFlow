"""Failure-sample anchoring on prompt tasks (not parsers).

``--task labels`` (default): label the kind of value. The task model reads a string (``'192.168.1.10'``, ``'#FFF'``, ``'03/15/2024'``...)
and must answer its label (``ipv4``, ``hex_color``, ``us_date``...), 12 labels with 2
search and 2 held-out strings each. The objective tells the proposer the naming
convention but not the labels, so the labels it never sees in feedback have to come
from its prior. Search examples are grouped by label, as labeled data often is.

``--task dates``: rewrite a date in one of 12 formats as ``YYYY-MM-DD`` (or
``invalid``). Conventions the objective does not state (slashes are month first, dots
day first, two-digit years from 50 are 19xx) come only from feedback.

``--task routing``: the support-ticket router of ``use_cases/evolution``; the queue
names are only visible in feedback.

``--task replies``: free-text support replies scored by an LLM judge against a
rubric per example (``replies_task.py``); the house policies are only in feedback.

``--task banking77``: Banking77 intent classification (Casanueva et al. 2020,
CC BY 4.0), 12 of the 77 intents drawn with a fixed seed, 2 search queries per
intent from ``train.csv`` and 3 held-out ones from ``test.csv``; like routing, the
intent names are only visible in feedback. Read from ``$PUBLIC_DATA_DIR/banking77``.

``--task bbh_ste``: BIG-Bench Hard ``salient_translation_error_detection``
(Suzgun et al. 2023), the six error types are the groups; 3 search and 6 held-out
examples per type, drawn with a fixed seed from ``$PUBLIC_DATA_DIR/bbh``. The task
text and option list every input shares become the seed prompt, so an input is its
source and translation; inputs longer than the 300 characters feedback shows are
left out (198 of 250 remain).

``--task scone``: ScoNe negation-scope NLI (She et al. 2023, CC0), entailment or
neutral; the six negation conditions are the groups. Read from ``$SCONE_DIR``
(the six ``<condition>.csv`` test files); 6 search and 12 held-out pairs per
condition, label-balanced, drawn with a fixed seed.

Cells are ``lineage-feedback[-reveal]`` as in ``anchoring.py``; reveals:

- ``first``: the first 4 wrong answers in example order.
- ``diverse``: ``prompt_evaluator(show_order="diverse")``, the same farthest-point
  sample ``cases_evaluator`` uses.
- ``minibatch``: GEPA-style: 4 examples drawn at random from all search examples
  in each evaluation, shown right or wrong (selection still uses the full score).
- ``cluster``: ESPO-style: all wrong answers go to one extra call of the task's
  proposer's model, which groups them into 3 to 7 patterns with a representative
  each; the proposer sees that summary. Its tokens count toward ``proposer_tokens``.
- ``clustergold``: ``cluster``, but each representative keeps its input and
  expected answer (stage-10 sensitivity arm).

    python benchmarks/evolution/prompt_anchoring.py --check
    python benchmarks/evolution/prompt_anchoring.py --provider anthropic \
        --model claude-haiku-5-5 --out runs/factorial/prompt

The task model is the proposer's model unless ``--proposer-model`` names another
one (same provider), which then also writes the cluster summaries.
``--carry-facts`` appends ``CARRY_FACTS`` to the objective: the proposer is asked to
write task-specific facts seen in the feedback (such as the label names) into the
prompt, as GEPA's reflection prompt does (stage-10 round-5 arm).
Held-out scoring has its own meter (``holdout_tokens``); ``tokens_to_solve`` is
proposer plus search-side task tokens of runs that solve the search set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import factorial  # noqa: E402
from factorial import LINEAGE, FactorialProposer, TokenMeter, make_client  # noqa: E402

from adalflow.optim.evolution import LLMProposer, Proposal, evolve  # noqa: E402
from adalflow.optim.evolution.presets import (  # noqa: E402
    PROMPT_TEMPLATE,
    exact_match,
    prompt_evaluator,
)

CARRY_FACTS = (
    " The task model sees only this prompt, never the feedback: copy every task-specific"
    " fact the feedback reveals (for example the exact expected labels or formats) into"
    " the prompt."
)

FAMILIES = {  # label: (search strings, held-out strings)
    "iso_date": (["2024-03-15", "1999-12-31"], ["2020-02-29", "2011-07-04"]),
    "us_date": (["03/15/2024", "12/31/1999"], ["07/04/2011", "11/23/2018"]),
    "eu_date": (["15.03.2024", "31.12.1999"], ["29.02.2020", "23.11.2018"]),
    "time": (["14:05", "23:59:59"], ["07:30", "18:45:10"]),
    "ipv4": (["192.168.1.10", "8.8.8.8"], ["10.0.0.255", "172.16.4.1"]),
    "ipv6": (
        ["2001:db8::1", "fe80::1ff:fe23:4567:890a"],
        ["::1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
    ),
    "mac_address": (
        ["00:1A:2B:3C:4D:5E", "a4:83:e7:0b:9c:12"],
        ["3C-52-82-11-AB-CD", "00:00:5e:00:53:af"],
    ),
    "uuid": (
        [
            "123e4567-e89b-12d3-a456-426614174000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ],
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "9b2d6f1e-3c4a-4e8b-9f7d-2a1b0c3d4e5f",
        ],
    ),
    "email": (
        ["ana@example.com", "j.smith+news@mail.co.uk"],
        ["root@localhost.dev", "x_y-z@sub.example.org"],
    ),
    "url": (
        ["https://example.com/a?b=1", "http://localhost:8080"],
        ["ftp://files.example.net/pub", "https://sub.domain.io/path#frag"],
    ),
    "hex_color": (["#1a2b3c", "#FFF"], ["#00ff7f", "#abc"]),
    "semver": (["1.4.2", "2.0.0-rc.1"], ["0.9.13", "10.2.3+build.7"]),
}
SEARCH = [(s, label) for label, (ss, _) in FAMILIES.items() for s in ss]
HOLDOUT = [(s, label) for label, (_, hs) in FAMILIES.items() for s in hs]
SEED = "Say what kind of value the input is."
OUTPUT_FORMAT = "Reply with the label only."
OBJECTIVE = (
    "A system prompt that makes the model label what kind of value the input string "
    "is, for example email. Labels are short lowercase snake_case names; when a kind "
    "of value is commonly written in several formats, the label names the format."
)
SHOW = 4

DATES = {  # format: ([(search input, answer)], [(held-out input, answer)])
    "iso": (
        [("2024-03-05", "2024-03-05"), ("2019-11-30", "2019-11-30")],
        [("2021-07-04", "2021-07-04"), ("2018-12-25", "2018-12-25")],
    ),
    "us_slash": (
        [("03/05/2024", "2024-03-05"), ("11/30/2019", "2019-11-30")],
        [("07/04/2021", "2021-07-04"), ("12/25/2018", "2018-12-25")],
    ),
    "eu_dot": (
        [("05.03.2024", "2024-03-05"), ("30.11.2019", "2019-11-30")],
        [("04.07.2021", "2021-07-04"), ("25.12.2018", "2018-12-25")],
    ),
    "long_us": (
        [("March 5, 2024", "2024-03-05"), ("November 30, 2019", "2019-11-30")],
        [("July 4, 2021", "2021-07-04"), ("December 25, 2018", "2018-12-25")],
    ),
    "long_eu": (
        [("5 March 2024", "2024-03-05"), ("30 Nov 2019", "2019-11-30")],
        [("4 Jul 2021", "2021-07-04"), ("25 December 2018", "2018-12-25")],
    ),
    "compact": (
        [("20240305", "2024-03-05"), ("20191130", "2019-11-30")],
        [("20210704", "2021-07-04"), ("20181225", "2018-12-25")],
    ),
    "slash_ymd": (
        [("2024/03/05", "2024-03-05"), ("2019/11/30", "2019-11-30")],
        [("2021/07/04", "2021-07-04"), ("2018/12/25", "2018-12-25")],
    ),
    "with_time": (
        [("2024-03-05T14:30:00Z", "2024-03-05"), ("2019-11-30 23:59", "2019-11-30")],
        [
            ("2021-07-04T00:00:00+02:00", "2021-07-04"),
            ("2018-12-25 08:15", "2018-12-25"),
        ],
    ),
    "ordinal": (
        [
            ("the 5th of March, 2024", "2024-03-05"),
            ("30th November 2019", "2019-11-30"),
        ],
        [
            ("4th of July 2021", "2021-07-04"),
            ("the 25th of December, 2018", "2018-12-25"),
        ],
    ),
    "short_year": (
        [("3/5/24", "2024-03-05"), ("11/30/99", "1999-11-30")],
        [("7/4/21", "2021-07-04"), ("12/25/98", "1998-12-25")],
    ),
    "weekday": (
        [
            ("Tue, 5 Mar 2024", "2024-03-05"),
            ("Saturday, November 30, 2019", "2019-11-30"),
        ],
        [
            ("Sun, 4 Jul 2021", "2021-07-04"),
            ("Tuesday, December 25, 2018", "2018-12-25"),
        ],
    ),
    "invalid": (
        [("2024-02-30", "invalid"), ("13/45/2020", "invalid")],
        [("2021-13-01", "invalid"), ("31.04.2019", "invalid")],
    ),
}


def _routing() -> Tuple[list, list]:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "use_cases" / "evolution")
    )
    import evolve_prompt  # noqa: E402

    return evolve_prompt.TRAIN, evolve_prompt.HOLDOUT


TASKS = ["labels", "dates", "routing", "replies", "scone", "banking77", "bbh_ste"]


def _public_dir() -> Path:
    import os

    return Path(os.environ.get("PUBLIC_DATA_DIR", "data/public"))


def _banking77(intents: int = 12, per_search: int = 2, per_holdout: int = 3):
    import csv

    root = _public_dir() / "banking77"
    rng = random.Random(0)
    split = {}
    for name in ("train", "test"):
        with open(root / f"{name}.csv", newline="") as f:
            for r in csv.DictReader(f):
                split.setdefault(name, {}).setdefault(r["category"], []).append(
                    r["text"].strip()
                )
    picked = sorted(rng.sample(sorted(split["test"]), intents))
    search, holdout = [], []
    for intent in picked:
        search += [(t, intent) for t in rng.sample(split["train"][intent], per_search)]
        holdout += [(t, intent) for t in rng.sample(split["test"][intent], per_holdout)]
    rng.shuffle(search)
    rng.shuffle(holdout)
    return search, holdout


def _bbh_ste(per_search: int = 3, per_holdout: int = 6):
    path = _public_dir() / "bbh" / "salient_translation_error_detection.json"
    rows = json.loads(path.read_text())  # the lukaemon/bbh test split, as JSON
    first = rows[0]["input"]
    head = first[: first.index("Source:")]
    tail = first[first.index("The translation contains") :]
    rng = random.Random(0)
    by_target = {}
    for r in rows:
        x = r["input"]
        assert x.startswith(head) and x.endswith(tail)
        core = x[len(head) : len(x) - len(tail)].strip()
        if len(core) <= 300:
            by_target.setdefault(r["target"], []).append(core)
    search, holdout = [], []
    for target in sorted(by_target):
        group = by_target[target]
        rng.shuffle(group)
        search += [(x, target) for x in group[:per_search]]
        holdout += [(x, target) for x in group[per_search : per_search + per_holdout]]
    rng.shuffle(search)
    rng.shuffle(holdout)
    seed = head.strip() + "\n" + tail[tail.index("Options:") :]
    return search, holdout, seed


@dataclass
class PromptTask:
    search: List[Tuple[str, str]]
    holdout: List[Tuple[str, str]]
    search_groups: List[str]  # coverage group of each example
    holdout_groups: List[str]
    seed: str
    output_format: str
    objective: str
    metric: Callable[[Any, Any], float] = exact_match


SCONE_CONDITIONS = (
    "no_negation",
    "one_scoped",
    "one_not_scoped",
    "two_scoped",
    "two_not_scoped",
    "one_scoped_one_not_scoped",
)


def _scone(per_search: int = 6, per_holdout: int = 12, seed: int = 0):
    import csv
    import os

    root = Path(os.environ.get("SCONE_DIR", "data/scone"))
    rng = random.Random(seed)
    search, holdout = [], []
    for cond in SCONE_CONDITIONS:
        with open(root / f"{cond}.csv", newline="") as f:
            raw = list(csv.DictReader(f))
        # Each condition's pair is the ``*_edited`` one (one_scoped has none: its
        # base pair is the condition); the label column is gold_label_edited if any.
        rows = [
            {
                "sentence1": r.get("sentence1_edited") or r["sentence1"],
                "sentence2": r.get("sentence2_edited") or r["sentence2"],
                "gold_label": r.get("gold_label_edited") or r["gold_label"],
            }
            for r in raw
        ]
        by_label = {}
        for r in rows:
            by_label.setdefault(r["gold_label"], []).append(r)
        picked_s, picked_h = [], []
        for label in sorted(by_label):
            group = by_label[label]
            rng.shuffle(group)
            a, b = per_search // len(by_label), per_holdout // len(by_label)
            picked_s += group[:a]
            picked_h += group[a : a + b]
        for picked, out in ((picked_s, search), (picked_h, holdout)):
            rng.shuffle(picked)
            out += [
                (
                    f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
                    r["gold_label"],
                    cond,
                )
                for r in picked
            ]
    return search, holdout


def make_task(name: str) -> PromptTask:
    if name == "labels":
        return PromptTask(
            SEARCH,
            HOLDOUT,
            [w for _, w in SEARCH],
            [w for _, w in HOLDOUT],
            SEED,
            OUTPUT_FORMAT,
            OBJECTIVE,
        )
    if name == "dates":
        return PromptTask(
            [ex for ss, _ in DATES.values() for ex in ss],
            [ex for _, hs in DATES.values() for ex in hs],
            [f for f, (ss, _) in DATES.items() for _ in ss],
            [f for f, (_, hs) in DATES.items() for _ in hs],
            "Rewrite the date.",
            "Reply with the answer only.",
            "A system prompt that makes the model rewrite a date written in any common "
            "format as an ISO 8601 calendar date (YYYY-MM-DD). Inputs that are not a real "
            "calendar date get the answer invalid.",
        )
    if name == "routing":
        train, holdout = _routing()
        return PromptTask(
            train,
            holdout,
            [w for _, w in train],
            [w for _, w in holdout],
            "Route the support ticket to a queue.",
            "Reply with the queue name only.",
            "A system prompt that makes the model route a customer support ticket to "
            "the right queue.",
        )
    if name == "scone":
        search, holdout = _scone()
        return PromptTask(
            [(x, y) for x, y, _ in search],
            [(x, y) for x, y, _ in holdout],
            [g for _, _, g in search],
            [g for _, _, g in holdout],
            "Decide whether sentence 2 follows from sentence 1.",
            "Reply with entailment or neutral only.",
            "A system prompt that makes the model decide whether sentence 2 is "
            "entailed by sentence 1 (answer entailment) or not (answer neutral).",
        )
    if name == "banking77":
        search, holdout = _banking77()
        return PromptTask(
            search,
            holdout,
            [w for _, w in search],
            [w for _, w in holdout],
            "Classify the banking customer query.",
            "Reply with the intent label only.",
            "A system prompt that makes the model classify an online-banking customer "
            "query into its intent. Intent labels are lower-case snake_case names.",
        )
    if name == "bbh_ste":
        search, holdout, seed = _bbh_ste()
        return PromptTask(
            search,
            holdout,
            [w for _, w in search],
            [w for _, w in holdout],
            seed,
            "Reply with the option letter in parentheses only, for example (A).",
            "A system prompt that makes the model find which type of error a "
            "German-to-English translation contains, answering with the option letter.",
        )
    if name == "replies":
        import replies_task as R

        (search, sg), (holdout, hg) = R.data(0), R.data(1)
        return PromptTask(
            search, holdout, sg, hg, R.SEED, R.OUTPUT_FORMAT, R.OBJECTIVE, R.judge
        )
    raise ValueError(f"unknown task {name!r}")


def cluster_evaluator(full, clusterer, gold=False):
    """``full`` shows every wrong answer; the proposer sees ``clusterer``'s summary."""

    def evaluate(instruction: str) -> Dict[str, Any]:
        out = full(instruction)
        out["feedback"] = factorial.summarize(clusterer, out["feedback"], gold)
        return out

    return evaluate


def search_evaluator(
    task: PromptTask, generator, reveal: str, rng: random.Random, clusterer=None
):
    """Scores every search example; the feedback depends on ``reveal``."""
    if reveal in ("first", "diverse"):
        return prompt_evaluator(
            generator,
            task.search,
            metric=task.metric,
            output_format=task.output_format,
            show_failures=SHOW,
            show_order=reveal,
        )
    full = prompt_evaluator(
        generator,
        task.search,
        metric=task.metric,
        output_format=task.output_format,
        show_failures=len(task.search),
        show_order="first",
    )
    if reveal in ("cluster", "clustergold"):
        return cluster_evaluator(full, clusterer, gold=reveal == "clustergold")
    assert reveal == "minibatch", reveal

    def evaluate(instruction: str) -> Dict[str, Any]:
        out = full(instruction)
        lines = out["feedback"].splitlines()
        shown = []
        for i in sorted(rng.sample(range(len(task.search)), SHOW)):
            item, want = task.search[i]
            key = f"input {item!r}: answered "
            wrong = [ln for ln in lines if ln.startswith(key)]
            if wrong:
                shown.append(wrong[0])
            elif task.metric is exact_match:
                shown.append(f"{key}{want!r}, correct")
            else:  # the expected value is a rubric, not an answer
                shown.append(f"input {item!r}: correct (expected {want!r})")
        out["feedback"] = "\n".join(shown)
        return out

    return evaluate


class FakeProposer:
    def __init__(self) -> None:
        self.count = 0

    def propose(self, parent, inspiration=None) -> Proposal:
        self.count += 1
        return Proposal(text=parent.text + f" Label {self.count}.")


class FakeGenerator:
    def call(self, prompt_kwargs, **kwargs):
        from adalflow.core.types import GeneratorOutput

        return GeneratorOutput(data="email")


REVEALS = ("first", "diverse", "minibatch", "cluster", "clustergold")


def parse_cell(spec: str):
    parts = spec.split("-")
    lineage, feedback = parts[0], parts[1]
    reveal = parts[2] if len(parts) > 2 else "first"
    assert lineage in LINEAGE and feedback in factorial.FEEDBACK
    assert reveal in REVEALS
    return lineage, feedback, reveal if feedback == "full" else "first"


def one_run(args, spec: str, repeat: int) -> Dict[str, Any]:
    from adalflow.core.generator import Generator

    lineage, feedback, reveal = parse_cell(spec)
    pt = make_task(args.task)
    if args.check:
        task, held, inner, meter = (
            FakeGenerator(),
            FakeGenerator(),
            FakeProposer(),
            None,
        )
        clusterer = FakeGenerator()
    else:
        client = make_client(args.provider)

        def generator():
            return Generator(
                model_client=client,
                model_kwargs={"model": args.model},
                template=PROMPT_TEMPLATE,
                use_cache=False,
            )

        task, held = TokenMeter(generator()), TokenMeter(generator())
        proposer_model = args.proposer_model or args.model
        inner = LLMProposer(
            pt.objective + (CARRY_FACTS if args.carry_facts else ""),
            kind="prompt",
            model_client=client,
            model_kwargs={"model": proposer_model},
        )
        meter = inner.generator = TokenMeter(inner.generator)
        clusterer = TokenMeter(
            Generator(
                model_client=client,
                model_kwargs={"model": proposer_model},
                use_cache=False,
            )
        )
    rng = random.Random(1000 + repeat)
    log = args.out / f"{spec}-{repeat}.jsonl"
    log.unlink(missing_ok=True)
    res = evolve(
        pt.seed,
        search_evaluator(pt, task, reveal, rng, clusterer),
        FactorialProposer(inner, feedback, meter),
        max_evaluations=args.budget,
        selection=LINEAGE[lineage],
        random_seed=repeat,
        stop_score=1.0,
        max_stalls=args.budget,
        log_path=log,
        holdout=prompt_evaluator(
            held,
            pt.holdout,
            metric=pt.metric,
            output_format=pt.output_format,
            show_failures=len(pt.holdout),
        ),
    )
    first = json.loads(log.read_text().splitlines()[0])
    cluster_tokens = clusterer.total if meter else 0
    spent = (meter.total + task.total + cluster_tokens) if meter else 0
    return {
        "key": spec if feedback == "full" else f"{lineage}-{feedback}",
        "repeat": repeat,
        "reveal": reveal,
        "evaluations": res.evaluations,
        "proposer_model": args.proposer_model or args.model,
        "proposer_tokens": (meter.total + cluster_tokens) if meter else 0,
        "cluster_tokens": cluster_tokens,
        "task_tokens": task.total if meter else 0,
        "holdout_tokens": held.total if meter else 0,
        "tokens_to_solve": spent if meter and res.best.score >= 1.0 else None,
        "best_chars": len(res.best.text),
        "holdout_seed": res.holdout["seed"].score,
        "holdout_best": res.holdout["best"].score if "best" in res.holdout else None,
        "holdout_best_feedback": (
            res.holdout["best"].feedback if "best" in res.holdout else None
        ),
        "holdout_best_scores": (
            res.holdout["best"].instance_scores if "best" in res.holdout else None
        ),
        "shown": (first.get("evaluation") or {}).get("feedback"),
        "candidates": [
            {
                "score": c.score,
                "text": c.text,
                "instance_scores": c.evaluation.instance_scores if c.evaluation else {},
            }
            for c in res.archive.candidates
        ],
    }


def labels_covered(scores: Dict[str, float], groups: List[str]) -> List[str]:
    """Groups whose examples all score 1 (``example-NN`` keys, in ``groups`` order)."""
    out = []
    for label in dict.fromkeys(groups):
        idx = [i for i, g in enumerate(groups) if g == label]
        if all(scores.get(f"example-{i:02d}") == 1.0 for i in idx):
            out.append(label)
    return out


def report(rows: List[Dict[str, Any]], pt: PromptTask, seed: int = 0) -> str:
    rng = random.Random(seed)
    labels = list(dict.fromkeys(pt.search_groups))
    out = [
        "Held-out best per run, and per label the share of proposals that get all "
        "search examples of the label right (first number) and the held-out best "
        "candidates that get all held-out examples right (second). Tokens exclude "
        "held-out scoring; to solve: median over runs that solved the search set.",
        "",
        "| cell | runs | tokens / run | solved | tokens to solve | best chars "
        "| held-out best | " + " | ".join(labels) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(labels),
    ]
    for key in sorted({r["key"] for r in rows}):
        g = [r for r in rows if r["key"] == key]
        props = [c for r in g for c in r["candidates"][1:] if c["score"] is not None]
        bests = [r["holdout_best_scores"] or {} for r in g]
        pc = [
            labels_covered(c["instance_scores"] or {}, pt.search_groups) for c in props
        ]
        bc = [labels_covered(b, pt.holdout_groups) for b in bests]
        cov = " | ".join(
            f"{sum(lab in x for x in pc)}/{len(pc)}, {sum(lab in x for x in bc)}/{len(bc)}"
            for lab in labels
        )
        tokens = statistics.mean(r["proposer_tokens"] + r["task_tokens"] for r in g)
        hb = [r["holdout_best"] for r in g if r["holdout_best"] is not None]
        solved = [r["tokens_to_solve"] for r in g if r.get("tokens_to_solve")]
        to_solve = f"{statistics.median(solved):.0f}" if solved else "-"
        chars = statistics.mean(r.get("best_chars", 0) for r in g)
        out.append(
            f"| {key} | {len(g)} | {tokens:.0f} | {len(solved)}/{len(g)} | {to_solve} "
            f"| {chars:.0f} | {factorial._fmt(hb, rng)} | {cov} |"
        )
    out += ["", "Wrong answers shown to the seed's first rewrite:", ""]
    for r in sorted(rows, key=lambda r: (r["key"], r["repeat"])):
        if r.get("shown") and "-full" in r["key"]:
            items = re.findall(r"^input (.*?): answered", r["shown"], re.M)
            out.append(f"- {r['key']} run {r['repeat']}: {', '.join(items)}")
    out += ["", "Mean [bootstrap 95% interval over runs]."]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--task", choices=TASKS, default="labels")
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--proposer-model", help="default: --model")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=[
            "seed-none",
            "seed-full-first",
            "seed-full-diverse",
            "seed-full-minibatch",
            "parent-full-first",
            "parent-full-diverse",
            "parent-full-minibatch",
        ],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/factorial/prompt"))
    parser.add_argument("--carry-facts", action="store_true", help="see CARRY_FACTS")
    parser.add_argument("--check", action="store_true", help="fake models, no API")
    parser.add_argument("--report", action="store_true", help="only report --out")
    args = parser.parse_args()
    if args.report:
        rows = json.loads((args.out / "summary.json").read_text())
        return print(report(rows, make_task(args.task)))
    if not args.check and not (args.provider and args.model):
        parser.error("--provider and --model are required (or pass --check)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check and args.task == "replies":
        import replies_task

        replies_task.judge.off = True
    jobs = [(c, r) for c in args.cells for r in range(args.repeats)]
    print(f"{len(jobs)} runs", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: one_run(args, *job), jobs))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    text = report(rows, make_task(args.task))
    if args.task == "replies":
        import replies_task

        j = replies_task.judge
        text += (
            f"\n\nJudge {replies_task.JUDGE_MODEL}: {j.calls} calls, {j.total} tokens."
        )
    (args.out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
