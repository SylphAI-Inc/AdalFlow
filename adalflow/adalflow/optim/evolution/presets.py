"""Shortcuts for the common cases: test-case evaluators and one-call optimizers.

    res = optimize_code(SEED, "median(xs) returns the median ...", function="median",
                        cases={"odd": (([3, 1, 2],), 2), "empty": (([],), ValueError)},
                        model="openai/gpt-4o-mini")

    res = optimize_prompt("Route the support ticket to a queue.", tickets,
                          model="anthropic/claude-haiku-5-5")
    print(res.report())

Both return the ``EvolutionResult`` of ``evolve``; every ``evolve`` keyword argument
(``selection``, ``confirm``, ``log_path``...) passes through.
"""

from __future__ import annotations

import ast
import inspect
import re
from concurrent.futures import ThreadPoolExecutor
from random import Random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from adalflow.optim.evolution.loop import EvolutionResult, evolve
from adalflow.optim.evolution.proposer import LLMProposer
from adalflow.optim.evolution.sandbox import PythonEvaluator

__all__ = [
    "cases_evaluator",
    "prompt_evaluator",
    "exact_match",
    "optimize_code",
    "optimize_prompt",
    "CARRY_FACTS",
]

# The task model sees only the instruction, never the feedback, so facts the proposer
# reads there (label names, formats) are lost unless written in. Appending this to
# the objective lifted Banking77 held-out from 0.37 to 0.86 (gpt-4.1-mini, 10 seeds)
# but cost 0.06 on BBH salient-translation-error detection.
CARRY_FACTS = (
    " The task model sees only this prompt, never the feedback: copy every task-specific"
    " fact the feedback reveals (for example the exact expected labels or formats) into"
    " the prompt."
)

Case = Tuple[Any, Any]  # (args, expected)
Cases = Union[Dict[str, Case], Sequence[Case]]


# Diverse failure sample. These run in the parent process (prompt_evaluator) and,
# pasted in as source, inside the cases_evaluator sandbox script.
def _shape(text):
    text = re.sub(r"\d+(?:\.\d+)?", "9", text)
    return re.sub(r"[a-z]+", "a", re.sub(r"[A-Z]+", "A", text))


def _distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def _diverse(inputs, k):
    # first failure, then the one least like those picked (farthest-point)
    if k >= len(inputs):
        return list(range(len(inputs)))
    shapes = [_shape(t) for t in inputs]
    picked = [0]
    # distance from each unpicked input to its nearest picked one
    near = {i: _distance(shapes[i], shapes[0]) for i in range(1, len(inputs))}
    while len(picked) < k:
        best = max(near, key=lambda i: (near[i], -i))
        picked.append(best)
        del near[best]
        for i in near:
            near[i] = min(near[i], _distance(shapes[i], shapes[best]))
    return sorted(picked)


_HELPERS = "\n\n".join(inspect.getsource(f) for f in (_shape, _distance, _diverse))

_CASES_SCRIPT = r"""
import math
import re

FUNCTION = __FUNCTION__
CASES = __CASES__
SHOW = __SHOW__
ORDER = __ORDER__
REL_TOL, ABS_TOL = __TOLS__


def _same(got, want):
    if isinstance(want, float) or isinstance(got, float):
        try:
            return math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        except TypeError:
            return False
    return got == want


def _call_text(kind, args):
    if kind == "kwargs":
        inner = ", ".join(f"{k}={v!r}" for k, v in args.items())
    else:
        inner = ", ".join(repr(a) for a in args)
    return f"{FUNCTION}({inner})"


__HELPERS__


def evaluate(candidate):
    fn = getattr(candidate, FUNCTION, None)
    if not callable(fn):
        raise AttributeError(f"the candidate does not define {FUNCTION}()")
    scores, failures, inputs = {}, [], []
    for name, kind, args, expect, want in CASES:
        call = _call_text(kind, args)
        try:
            got = fn(**args) if kind == "kwargs" else fn(*args)
        except Exception as exc:
            raised = [c.__name__ for c in type(exc).__mro__]
            ok = expect == "raises" and want in raised
            what = f"raised {type(exc).__name__}: {exc}"
            if not ok:
                what += f"; expected {want + ' to be raised' if expect == 'raises' else repr(want)}"
        else:
            if expect == "raises":
                ok, what = False, f"returned {got!r}; expected {want} to be raised"
            else:
                ok, what = _same(got, want), f"returned {got!r}; expected {want!r}"
        scores[name] = float(ok)
        if not ok:
            failures.append(f"{name}: {call} {what}")
            inputs.append(call)
    if ORDER == "diverse":
        shown = [failures[i] for i in _diverse(inputs, SHOW)]
    else:
        shown = failures[:SHOW]
    if len(failures) > len(shown):
        shown.append(f"... {len(failures) - len(shown)} more failing cases")
    return {"instance_scores": scores, "feedback": "\n".join(shown)}
"""


def _literal(value: Any, what: str) -> Any:
    try:
        ok = ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        ok = False
    if not ok:
        raise ValueError(
            f"{what} must be a Python literal (numbers, strings, lists, dicts, "
            f"tuples, None...), got {value!r}"
        )
    return value


def _encode_cases(cases: Cases) -> List[tuple]:
    items = (
        list(cases.items())
        if isinstance(cases, dict)
        else [(f"case-{i:02d}", c) for i, c in enumerate(cases)]
    )
    if not items:
        raise ValueError("cases is empty")
    encoded = []
    for name, case in items:
        if not (isinstance(case, tuple) and len(case) == 2):
            raise ValueError(f"case {name!r} must be an (args, expected) pair")
        args, want = case
        if isinstance(args, dict):
            kind = "kwargs"
        else:
            kind = "args"
            args = args if isinstance(args, tuple) else (args,)
        _literal(args, f"arguments of case {name!r}")
        if isinstance(want, type) and issubclass(want, BaseException):
            encoded.append((str(name), kind, args, "raises", want.__name__))
        else:
            _literal(want, f"expected value of case {name!r}")
            encoded.append((str(name), kind, args, "value", want))
    return encoded


def cases_evaluator(
    function: str,
    cases: Cases,
    *,
    show_failures: int = 8,
    show_order: str = "diverse",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
    **kwargs: Any,
) -> PythonEvaluator:
    """A ``PythonEvaluator`` that calls ``function`` on each case.

    Args:
        function: name of the function the candidate defines.
        cases: ``{name: (args, expected)}`` or a list of ``(args, expected)``.
            ``args`` is a tuple of positional arguments, a dict of keyword
            arguments, or a single value. ``expected`` is the return value
            (floats compare with ``math.isclose``) or an exception class the call
            must raise (``ValueError``). Values must be Python literals.
        show_failures: how many failing calls the feedback spells out.
        show_order: which of them. ``"diverse"`` (default) takes the first, then
            each time the failing call least like those already taken (edit
            distance after mapping digits to ``9`` and letter runs to ``a``/``A``);
            ``"first"`` takes them in case order. Shown failures act as a spec:
            a weaker proposer shown four failures from one input family dropped
            formats it handled with no feedback at all. A diverse sample did better
            in most of our runs but can still leave formats out; with many input
            families, raise ``show_failures``.
        kwargs: ``timeout``, ``memory_mb``... forwarded to ``PythonEvaluator``.
    """
    if not re.fullmatch(r"[A-Za-z_]\w*", function):
        raise ValueError(f"function must be a Python identifier, got {function!r}")
    if show_order not in ("first", "diverse"):
        raise ValueError(f"show_order must be 'first' or 'diverse', got {show_order!r}")
    source = (
        _CASES_SCRIPT.replace("__FUNCTION__", repr(function))
        .replace("__CASES__", repr(_encode_cases(cases)))
        .replace("__SHOW__", repr(int(show_failures)))
        .replace("__ORDER__", repr(show_order))
        .replace("__HELPERS__", _HELPERS)
        .replace("__TOLS__", f"{float(rel_tol)!r}, {float(abs_tol)!r}")
    )
    return PythonEvaluator(source, **kwargs)


def _resolve(model: Any, model_kwargs: Optional[Dict[str, Any]]):
    from adalflow.components.model_client.spec import resolve_model

    return resolve_model(model, model_kwargs)


def optimize_code(
    seed: str,
    objective: str,
    *,
    function: Optional[str] = None,
    cases: Optional[Cases] = None,
    holdout_cases: Optional[Cases] = None,
    evaluator: Optional[Callable[[str], Any]] = None,
    model: Any = "openai/gpt-4o-mini",
    model_kwargs: Optional[Dict[str, Any]] = None,
    budget: int = 20,
    stop_score: Optional[float] = None,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolves a Python program against test cases (or any evaluator).

    Pass ``function`` and ``cases`` (see ``cases_evaluator``), optionally
    ``holdout_cases`` checked only after the search; or pass your own
    ``evaluator``. With cases, the run stops once every case passes.

    Args:
        model: ``"provider/model"`` or a ``ModelClient`` for the proposer.
        budget: evaluator calls, the seed included (``max_evaluations``).
        evolve_kwargs: any other ``evolve`` argument (``selection``, ``holdout``,
            ``log_path``, ``confirm``...).
    """
    if (evaluator is None) == (cases is None):
        raise ValueError("pass exactly one of cases (with function) or evaluator")
    if cases is not None:
        if function is None:
            raise ValueError("cases need the function name to call")
        evaluator = cases_evaluator(function, cases)
        if stop_score is None:
            stop_score = 1.0
    if holdout_cases is not None:
        if function is None:
            raise ValueError("holdout_cases need the function name to call")
        evolve_kwargs.setdefault("holdout", cases_evaluator(function, holdout_cases))
    client, kwargs = _resolve(model, model_kwargs)
    proposer = LLMProposer(
        objective, kind="Python program", model_client=client, model_kwargs=kwargs
    )
    return evolve(
        seed,
        evaluator,
        proposer,
        max_evaluations=budget,
        stop_score=stop_score,
        **evolve_kwargs,
    )


# ------------------------------------------------------------------ prompts
PROMPT_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{ instruction }}
{% if output_format %}
{{ output_format }}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
{{ input }}
<END_OF_USER_PROMPT>"""


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip(".").strip().lower()


def exact_match(prediction: Any, expected: Any) -> float:
    """1.0 when the texts match ignoring case, surrounding space and a final period."""
    return float(_normalize(prediction) == _normalize(expected))


def prompt_evaluator(
    generator: Any,
    examples: Sequence[Tuple[Any, Any]],
    *,
    metric: Callable[[Any, Any], float] = exact_match,
    output_format: str = "",
    workers: int = 8,
    show_failures: int = 12,
    show_order: str = "diverse",
) -> Callable[[str], Dict[str, Any]]:
    """``instruction -> {"instance_scores", "feedback"}`` on labeled ``examples``.

    ``generator`` must accept ``instruction``, ``input`` and ``output_format`` prompt
    kwargs (``PROMPT_TEMPLATE`` does). The feedback lists the examples the
    instruction got wrong, which is what the proposer learns the task from:
    ``show_failures`` of them, picked to differ from each other as in
    ``cases_evaluator`` (``show_order="diverse"``, default) or in example order
    (``"first"``).
    """
    if show_order not in ("first", "diverse"):
        raise ValueError(f"show_order must be 'first' or 'diverse', got {show_order!r}")
    examples = list(examples)
    if not examples:
        raise ValueError("examples is empty")

    def answer(instruction: str, item: Any) -> str:
        out = generator.call(
            prompt_kwargs={
                "instruction": instruction,
                "input": item,
                "output_format": output_format,
            }
        )
        if getattr(out, "error", None):
            return f"<model error: {out.error}>"
        return str(out.data if out.data is not None else "")

    def evaluate(instruction: str) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            answers = list(pool.map(lambda ex: answer(instruction, ex[0]), examples))
        scores, misses, inputs = {}, [], []
        for i, ((item, want), got) in enumerate(zip(examples, answers)):
            value = float(metric(got, want))
            scores[f"example-{i:02d}"] = value
            if value < 1.0:
                misses.append(
                    f"input {str(item)[:300]!r}: answered {got.strip()[:300]!r}, "
                    f"expected {str(want)[:300]!r}"
                )
                inputs.append(str(item)[:300])
        if show_order == "diverse":
            feedback = [misses[i] for i in _diverse(inputs, show_failures)]
        else:
            feedback = misses[:show_failures]
        if len(misses) > len(feedback):
            feedback.append(f"... {len(misses) - len(feedback)} more wrong answers")
        return {"instance_scores": scores, "feedback": "\n".join(feedback)}

    return evaluate


def _split(examples: list, holdout: Union[float, Sequence, None], seed: int):
    if holdout is None or (isinstance(holdout, (int, float)) and holdout == 0):
        return examples, []
    if not isinstance(holdout, (int, float)):
        return examples, list(holdout)
    if not 0 < holdout < 1:
        raise ValueError("holdout must be a fraction in (0, 1), a list or None")
    shuffled = list(examples)
    Random(seed).shuffle(shuffled)
    n = int(round(len(shuffled) * holdout))
    if n == 0 or n == len(shuffled):
        raise ValueError(
            f"holdout={holdout} leaves no search or no held-out examples out of "
            f"{len(shuffled)}; pass more examples or holdout=None"
        )
    return shuffled[n:], shuffled[:n]


def optimize_prompt(
    seed: str,
    examples: Sequence[Tuple[Any, Any]],
    *,
    objective: Optional[str] = None,
    model: Any = "openai/gpt-4o-mini",
    model_kwargs: Optional[Dict[str, Any]] = None,
    task_model: Any = None,
    task_model_kwargs: Optional[Dict[str, Any]] = None,
    metric: Callable[[Any, Any], float] = exact_match,
    output_format: str = "",
    holdout: Union[float, Sequence[Tuple[Any, Any]], None] = 0.25,
    budget: int = 10,
    stop_score: Optional[float] = 1.0,
    workers: int = 8,
    random_seed: int = 0,
    carry_facts: bool = True,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolves an instruction (system prompt) on labeled ``(input, expected)`` pairs.

    Args:
        seed: the starting instruction.
        examples: labeled pairs; a ``holdout`` share of them (default 25%, picked
            with ``random_seed``) is kept out of the search and only scored at the
            end (``result.holdout``). Pass a list to choose them, or ``None``.
        model: proposer model, ``"provider/model"`` or a ``ModelClient``.
        task_model: the model that follows the instruction (default: ``model``).
        metric: ``(prediction, expected) -> float in [0, 1]``; default
            ``exact_match``.
        output_format: fixed text appended to every instruction, such as "Reply
            with one word.", which the search does not change.
        budget: evaluations (each runs every search example once).
        carry_facts: append ``CARRY_FACTS`` to the objective, asking the proposer
            to copy facts from the feedback (such as the label set) into the
            instruction. It helps most when the task's outputs are names the model
            cannot guess; turn it off for reasoning tasks with fixed options.
        evolve_kwargs: any other ``evolve`` argument.
    """
    from adalflow.core.generator import Generator

    search, held_out = _split(list(examples), holdout, random_seed)
    if not search:
        raise ValueError("no examples left to search with")
    client, kwargs = _resolve(model, model_kwargs)
    if task_model is None:
        task_client, task_kwargs = client, dict(kwargs)
        task_kwargs.update(task_model_kwargs or {})
    else:
        task_client, task_kwargs = _resolve(task_model, task_model_kwargs)
    generator = Generator(
        model_client=task_client,
        model_kwargs=task_kwargs,
        template=PROMPT_TEMPLATE,
        use_cache=False,
    )

    def scorer(rows):
        return prompt_evaluator(
            generator, rows, metric=metric, output_format=output_format, workers=workers
        )

    if held_out:
        evolve_kwargs.setdefault("holdout", scorer(held_out))
    objective = (
        objective
        or "A system prompt that makes the model produce the expected output for "
        "every input. Infer the task's rules from the wrong answers and state them."
    )
    proposer = LLMProposer(
        objective + (CARRY_FACTS if carry_facts else ""),
        kind="prompt",
        model_client=client,
        model_kwargs=kwargs,
    )
    return evolve(
        seed,
        scorer(search),
        proposer,
        max_evaluations=budget,
        stop_score=stop_score,
        random_seed=random_seed,
        **evolve_kwargs,
    )
