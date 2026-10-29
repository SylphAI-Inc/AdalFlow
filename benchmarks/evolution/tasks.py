"""Evolution benchmark tasks with separate search and held-out instances.

Each task has a seed program, a search evaluator (what ``evolve`` optimizes) and a
held-out evaluator on instances the search never sees. Both are ``PythonEvaluator``
scripts, so candidates run in a separate process.

- ``duration``: parse duration strings. The objective names the goal but not the
  formats; the evaluator reports at most 4 failing inputs per run under opaque case
  names, so the full spec has to be recovered over several rounds of feedback.
- ``size``: parse data-size strings (bytes, SI and IEC units, words), built like
  ``duration``: same evaluator, at most 4 failing inputs shown per run.
- ``binpack``: an online bin-packing heuristic (the classic FunSearch task), scored
  as lower bound / bins used on random instances. There is no perfect score; the
  held-out instances use other random seeds from the same distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Task", "TASKS", "binpack_cell"]


@dataclass
class Task:
    name: str
    objective: str
    seed: str
    search_evaluator: str
    holdout_evaluator: str
    stop_score: Optional[float] = None
    # candidate -> behavior cell, for selection="map_elites"
    descriptor: Optional[Callable[[Any], Tuple]] = None


# ------------------------------------------------------------------ duration
# (input, seconds) for valid strings, (input, None) for strings that must raise.
DurationCase = Tuple[str, object]

DURATION_SEARCH: List[DurationCase] = [
    ("90", 90),
    ("45s", 45),
    ("5m", 300),
    ("2h", 7200),
    ("1h30m", 5400),
    ("1h 30m 15s", 5415),
    ("3d", 259200),
    ("1w", 604800),
    ("1.5h", 5400),
    ("2 hours", 7200),
    ("1 minute 30 seconds", 90),
    ("01:30:00", 5400),
    ("2:05", 125),
    ("PT1H30M", 5400),
    ("P1DT2H", 93600),
    ("  10m ", 600),
    ("", None),
    ("abc", None),
    ("1x", None),
    ("h", None),
    ("-5m", None),
    ("1:75", None),
    ("1h1h", None),
    ("1.2.3h", None),
]

DURATION_HOLDOUT: List[DurationCase] = [
    ("120", 120),
    ("7s", 7),
    ("15m", 900),
    ("10h", 36000),
    ("2h15m", 8100),
    ("4m 20s", 260),
    ("2d 3h", 183600),
    ("2w", 1209600),
    ("0.5m", 30),
    ("3 minutes", 180),
    ("1 hour 1 second", 3601),
    ("1 day", 86400),
    ("00:00:45", 45),
    ("12:00", 720),
    ("PT45S", 45),
    ("P2D", 172800),
    ("PT2H5M10S", 7510),
    ("1h 1m", 3660),
    ("m5", None),
    ("1h-30m", None),
    ("10:60", None),
    ("2 parsecs", None),
    ("PT", None),
    ("5m5m", None),
]

_DURATION_EVALUATOR = r"""
import math

CASES = __CASES__
SHOW = 4


def evaluate(candidate):
    scores, failures = {}, []
    for i, (text, want) in enumerate(CASES):
        name = f"case-{i:02d}"
        try:
            got = candidate.parse_duration(text)
            if want is None:
                ok, what = False, f"returned {got!r}, expected ValueError"
            else:
                ok = isinstance(got, (int, float)) and not isinstance(got, bool) \
                    and math.isclose(got, want, abs_tol=1e-6)
                what = f"returned {got!r}, expected {want}"
        except ValueError as exc:
            ok = want is None
            what = f"raised ValueError({exc}), expected {want}"
        except Exception as exc:
            ok, what = False, f"raised {type(exc).__name__}: {exc}"
        scores[name] = float(ok)
        if not ok:
            failures.append(f"{name}: parse_duration({text!r}) {what}")
    shown = failures[:SHOW]
    if len(failures) > SHOW:
        shown.append(f"... {len(failures) - SHOW} more failing cases not shown")
    return {"instance_scores": scores, "feedback": "\n".join(shown)}
"""


def _duration_evaluator(cases: List[DurationCase]) -> str:
    return _DURATION_EVALUATOR.replace("__CASES__", repr(cases))


DURATION = Task(
    name="duration",
    objective=(
        "parse_duration(text) returns the length of a duration string in seconds, "
        "for the formats people and systems commonly write durations in, and raises "
        "ValueError for strings that are not valid durations."
    ),
    seed="""def parse_duration(text):
    units = {"s": 1, "m": 60, "h": 3600}
    text = text.strip()
    return float(text[:-1]) * units[text[-1]]
""",
    search_evaluator=_duration_evaluator(DURATION_SEARCH),
    holdout_evaluator=_duration_evaluator(DURATION_HOLDOUT),
    stop_score=1.0,
)

# ------------------------------------------------------------------ size
# Conventions: a single-letter unit (K, M, G, T, any case) is binary (1024), a unit
# ending in B (KB, MB, ...) is decimal, KiB/Ki style is binary, and words follow the
# same split (kilobyte = 1000, kibibyte = 1024).
SIZE_SEARCH: List[DurationCase] = [
    ("512", 512),
    ("100B", 100),
    ("4K", 4096),
    ("512M", 536870912),
    ("2G", 2147483648),
    ("10KB", 10000),
    ("1.5MB", 1500000),
    ("3 GB", 3000000000),
    ("1kB", 1000),
    ("1KiB", 1024),
    ("4MiB", 4194304),
    ("2Gi", 2147483648),
    ("256Mi", 268435456),
    ("1.5K", 1536),
    ("512m", 536870912),
    ("10 kilobytes", 10000),
    ("1 byte", 1),
    ("2 gibibytes", 2147483648),
    ("1,024", 1024),
    ("", None),
    ("MB", None),
    ("-1KB", None),
    ("10XB", None),
    ("5 MB MB", None),
]

SIZE_HOLDOUT: List[DurationCase] = [
    ("2048", 2048),
    ("7B", 7),
    ("8K", 8192),
    ("1M", 1048576),
    ("1T", 1099511627776),
    ("500KB", 500000),
    ("2.5GB", 2500000000),
    ("20 MB", 20000000),
    ("1mb", 1000000),
    ("8KiB", 8192),
    ("1.5GiB", 1610612736),
    ("64Ki", 65536),
    ("1Ti", 1099511627776),
    ("0.5M", 524288),
    ("2g", 2147483648),
    ("3 megabytes", 3000000),
    ("5 bytes", 5),
    ("1 kibibyte", 1024),
    ("10,000", 10000),
    ("GB", None),
    ("1..5MB", None),
    ("-2G", None),
    ("12 parsecs", None),
    ("KB10", None),
]


def _size_evaluator(cases: List[DurationCase]) -> str:
    return _duration_evaluator(cases).replace("parse_duration", "parse_size")


SIZE = Task(
    name="size",
    objective=(
        "parse_size(text) returns the number of bytes in a data-size string, for the "
        "formats people and systems commonly write sizes in, and raises ValueError for "
        "strings that are not valid sizes."
    ),
    seed="""def parse_size(text):
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    text = text.strip()
    return int(text[:-1]) * units[text[-1]]
""",
    search_evaluator=_size_evaluator(SIZE_SEARCH),
    holdout_evaluator=_size_evaluator(SIZE_HOLDOUT),
    stop_score=1.0,
)

# ------------------------------------------------------------------ binpack
_BINPACK_EVALUATOR = r"""
import math
import random

SEEDS = __SEEDS__
N_ITEMS, CAPACITY, LOW, HIGH = 300, 150, 20, 100


def best_fit(item, bins):
    fits = [(r, i) for i, r in enumerate(bins) if r >= item]
    return min(fits)[1] if fits else -1


def pack(choose_bin, items, agree):
    bins = []  # remaining capacity of each open bin
    for item in items:
        i = choose_bin(item, list(bins), CAPACITY)
        agree.append(i == best_fit(item, bins))
        if i == -1:
            bins.append(CAPACITY - item)
            continue
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(bins):
            raise ValueError(f"choose_bin returned {i!r} with {len(bins)} open bins")
        if bins[i] < item:
            raise ValueError(f"item {item} does not fit bin {i} (remaining {bins[i]})")
        bins[i] -= item
    return len(bins)


def evaluate(candidate):
    scores, notes, agree = {}, [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        items = [rng.randint(LOW, HIGH) for _ in range(N_ITEMS)]
        bound = math.ceil(sum(items) / CAPACITY)
        name = f"instance-{seed}"
        try:
            used = pack(candidate.choose_bin, items, agree)
            scores[name] = bound / used
            notes.append(f"{name}: {used} bins, lower bound {bound}")
        except Exception as exc:
            scores[name] = 0.0
            notes.append(f"{name}: invalid packing: {type(exc).__name__}: {exc}")
    return {
        "instance_scores": scores,
        "feedback": "\n".join(notes),
        # Behavior, not quality: how often a decision matches best fit.
        "bestfit_agreement": sum(agree) / len(agree) if agree else 0.0,
    }
"""


def binpack_cell(candidate) -> Tuple[int, int]:
    """(agreement-with-best-fit bucket, code length bucket)."""
    agreement = candidate.evaluation.metrics.get("bestfit_agreement", 0.0)
    band = sum(agreement >= edge for edge in (0.6, 0.8, 0.9, 0.97))
    lines = len([ln for ln in candidate.text.splitlines() if ln.strip()])
    length = 0 if lines <= 8 else 1 if lines <= 20 else 2 if lines <= 40 else 3
    return (band, length)


def _binpack_evaluator(seeds: List[int]) -> str:
    return _BINPACK_EVALUATOR.replace("__SEEDS__", repr(seeds))


BINPACK = Task(
    name="binpack",
    objective=(
        "choose_bin(item, bins, capacity) packs items online: bins lists the remaining "
        "capacity of each open bin; return the index of an open bin the item fits in, "
        "or -1 to open a new bin. Use as few bins as possible. Items are integers drawn "
        "uniformly from 20..100 and capacity is 150; each instance has 300 items."
    ),
    seed="""def choose_bin(item, bins, capacity):
    for i, remaining in enumerate(bins):
        if remaining >= item:
            return i
    return -1
""",
    search_evaluator=_binpack_evaluator([0, 1, 2, 3, 4]),
    holdout_evaluator=_binpack_evaluator(list(range(100, 110))),
    descriptor=binpack_cell,
)

TASKS: Dict[str, Task] = {t.name: t for t in (DURATION, SIZE, BINPACK)}
