"""
Evolve a buggy Python function with feedback from real test runs.

The seed ``median`` fails on even-length and empty inputs. Every candidate runs in a
separate Python process (``PythonEvaluator``) against the test cases below; the
proposer reads the failing cases and tracebacks and rewrites the function.

Default ``--provider scripted`` replaces the LLM with a stand-in that fixes whatever
the feedback in its prompt reports, so the run needs no credentials and shows the
mechanics only (the tests and the sandbox are real). ``--provider openai`` uses a real
model (requires OPENAI_API_KEY; not exercised in CI).

    python use_cases/evolution/evolve_function_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "adalflow") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "adalflow"))

from adalflow.core.model_client import ModelClient  # noqa: E402
from adalflow.core.types import GeneratorOutput  # noqa: E402
from adalflow.optim.evolution import LLMProposer, PythonEvaluator, evolve  # noqa: E402

OBJECTIVE = (
    "median(xs) returns the median of a list of numbers (the mean of the two middle "
    "values for even lengths) and raises ValueError on an empty list."
)
SEED = "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"

EVALUATOR = """
CASES = {
    "odd": ([3, 1, 2], 2),
    "even": ([4, 1, 3, 2], 2.5),
    "pair": ([1, 2], 1.5),
    "negative": ([-5, -1, -3], -3),
    "floats": ([0.5, 0.25], 0.375),
}

def evaluate(candidate):
    scores, notes = {}, []
    for name, (xs, want) in CASES.items():
        try:
            got = candidate.median(list(xs))
        except Exception as exc:
            got = repr(exc)
        scores[name] = 1.0 if got == want else 0.0
        if got != want:
            notes.append(f"{name}: median({xs}) expected {want}, got {got}")
    try:
        candidate.median([])
        scores["empty"] = 0.0
        notes.append("empty: median([]) should raise ValueError, returned a value")
    except ValueError:
        scores["empty"] = 1.0
    except Exception as exc:
        scores["empty"] = 0.0
        notes.append(f"empty: median([]) should raise ValueError, raised {exc!r}")
    return {"instance_scores": scores, "feedback": "\\n".join(notes)}
"""

_AVERAGED = (
    "def median(xs):\n"
    "    s = sorted(xs)\n"
    "    m = len(s) // 2\n"
    "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"
)
_GUARDED = (
    "def median(xs):\n"
    "    if not xs:\n"
    "        raise ValueError('median() of an empty list')\n"
    + _AVERAGED.split("\n", 1)[1]
)


class ScriptedClient(ModelClient):
    """Stands in for an LLM: fixes exactly what the feedback in the prompt reports."""

    def convert_inputs_to_api_kwargs(
        self, input=None, model_kwargs={}, model_type=None
    ):
        return {"prompt": input}

    def call(self, api_kwargs={}, model_type=None):
        prompt = api_kwargs["prompt"]
        if "even: median" in prompt:
            return "Even lengths need the mean of the two middle values.\n" + (
                f"```candidate\n{_AVERAGED}```"
            )
        if "empty: median" in prompt:
            return f"Guard the empty list.\n```candidate\n{_GUARDED}```"
        return f"No failures reported.\n```candidate\n{_GUARDED}# reviewed\n```"

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data=None, raw_response=completion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--provider", choices=["scripted", "openai"], default="scripted"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-evaluations", type=int, default=8)
    parser.add_argument("--log", type=Path, help="Write a JSONL event log here.")
    args = parser.parse_args()

    if args.provider == "scripted":
        client, kwargs = ScriptedClient(), {"model": "scripted"}
    else:
        from adalflow.components.model_client import OpenAIClient

        client, kwargs = OpenAIClient(), {"model": args.model}
    proposer = LLMProposer(OBJECTIVE, model_client=client, model_kwargs=kwargs)
    result = evolve(
        SEED,
        PythonEvaluator(EVALUATOR, timeout=10),
        proposer,
        max_evaluations=args.max_evaluations,
        stop_score=1.0,
        log_path=args.log,
    )
    lineage = [
        {
            "id": c.id,
            "parent": c.parent_id,
            "score": c.score,
            "failing": sorted(
                k for k, v in (c.evaluation.instance_scores or {}).items() if v < 1
            ),
            "error": c.evaluation.error,
            "rationale": c.rationale,
        }
        for c in result.archive.candidates
    ]
    print(json.dumps({"stopped": result.stopped, "candidates": lineage}, indent=2))
    print(
        "\nBest candidate:\n" + result.best.text
        if result.best
        else "\nNo candidate ran."
    )


if __name__ == "__main__":
    main()
