"""
Label-free harness optimization demo: totaling ledgers that contain credits.

The base harness says nothing about signs, so the (scripted) agent drops credit rows.
``run_domain`` refines the harness on each task without an answer key, picks a final
blind, and stores an evidence-supported lesson that the next task's editor can reuse.

Default ``--provider scripted`` needs no credentials: the agent, editor, selector and
reflector are deterministic stand-ins, so the run shows the mechanics only. With
``--provider openai`` the same pipeline uses a real model for every role (requires
OPENAI_API_KEY; not exercised in CI).

    python use_cases/harness_optimization/ledger_demo.py --k 2 --mode parallel
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "adalflow") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "adalflow"))

from adalflow.core.model_client import ModelClient  # noqa: E402
from adalflow.core.types import GeneratorOutput  # noqa: E402
from adalflow.optim.harness import (  # noqa: E402
    BlindSelector,
    ExperienceLibrary,
    ExperienceReflector,
    Harness,
    HarnessEditor,
    ProbeResult,
    RunnerExecutor,
    run_domain,
)

LEDGERS: Dict[str, List[float]] = {
    "ledger-1": [120.0, -30.5, 40.0, -24.25],
    "ledger-2": [88.0, 12.5, -40.0, -7.75, 3.0],
}
BASE = Harness(system_prompt="You total ledgers. Answer as 'Total: <amount>'.")


def task_text(ledger_id: str) -> str:
    rows = ", ".join(str(r) for r in LEDGERS[ledger_id])
    return f"Total the amounts in {ledger_id}: {rows}"


# ------------------------------------------------------------------ scripted roles
class ScriptedClient(ModelClient):
    """Deterministic replies computed from the rendered prompt."""

    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def convert_inputs_to_api_kwargs(
        self, input=None, model_kwargs={}, model_type=None
    ):
        return {"prompt": input}

    def call(self, api_kwargs={}, model_type=None):
        return self.reply(api_kwargs["prompt"])

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data=None, raw_response=completion)


def _rows_in(prompt: str) -> List[float]:
    match = re.search(r"Total the amounts in [\w-]+: ([-\d., ]+)", prompt)
    return [float(x) for x in match.group(1).split(",")] if match else []


def _editor_reply(prompt: str) -> str:
    if "signed" in prompt and "Experience from earlier tasks" in prompt:
        notes = "reused the signed-totals lesson from an earlier task"
    else:
        notes = "keep signed amounts -> base runs dropped every negative row"
    return json.dumps(
        {
            "harness": {
                "system_prompt": BASE.system_prompt,
                "guardrails": "Keep signed amounts: credits are negative rows.",
                "skills": {
                    "signed-totals": {
                        "description": "Totals over ledgers that contain credits",
                        "body": "Add every row with its sign, then re-add to check.",
                    }
                },
            },
            "notes": notes,
        }
    )


def _selector_reply(prompt: str) -> str:
    # Re-adds the task's rows (source data), as a careful reviewer would.
    expected = f"{sum(_rows_in(prompt)):.2f}"
    for ident, body in re.findall(
        r"## Deliverable (\w)\n(.*?)(?=\n## |\Z)", prompt, re.S
    ):
        if expected in body:
            return json.dumps(
                {"why": "consistent with the source rows", "choice": ident}
            )
    return json.dumps({"why": "no deliverable matches the rows", "choice": "A"})


def _reflector_reply(prompt: str) -> str:
    return json.dumps(
        {
            "edits": [
                {
                    "component": "guardrails",
                    "candidates": "all candidates",
                    "change": "keep signed amounts",
                    "observed": "credit rows kept in every candidate run",
                    "verdict": "helped",
                }
            ],
            "experience": {
                "cue": "ledger totals with negative (credit) rows",
                "lesson": "State the sign convention in guardrails; add a "
                "signed-totals skill.",
                "evidence": "guardrails row: helped; base runs dropped credits",
                "confidence": "medium",
            },
        }
    )


class ScriptedLedgerAgent:
    """Stands in for an LLM agent: keeps credits only if its harness says so."""

    def execute(self, harness: Harness, task: str, label: str) -> ProbeResult:
        rows = _rows_in(task)
        if "signed" not in harness.render_role_desc().lower():
            rows = [r for r in rows if r > 0]
        total = sum(rows)
        trajectory = "\n".join(f"[ACTION] add({r})" for r in rows)
        return ProbeResult(
            label=label,
            harness=harness,
            answer=f"Total: {total:.2f}",
            trajectory=f"{trajectory}\n[ANSWER] Total: {total:.2f}",
            steps=len(rows),
        )


def build(provider: str, model: str):
    if provider == "scripted":
        kw = {"model": "scripted"}
        return (
            ScriptedLedgerAgent(),
            HarnessEditor(model_client=ScriptedClient(_editor_reply), model_kwargs=kw),
            BlindSelector(
                model_client=ScriptedClient(_selector_reply), model_kwargs=kw
            ),
            ExperienceReflector(
                model_client=ScriptedClient(_reflector_reply), model_kwargs=kw
            ),
        )
    from adalflow.components.model_client import OpenAIClient

    client, kw = OpenAIClient(), {"model": model}
    executor = RunnerExecutor(
        agent_factory=lambda h: h.to_agent(
            model_client=client, model_kwargs=kw, max_steps=5
        )
    )
    return (
        executor,
        HarnessEditor(model_client=client, model_kwargs=kw),
        BlindSelector(model_client=client, model_kwargs=kw),
        ExperienceReflector(model_client=client, model_kwargs=kw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--provider", choices=["scripted", "openai"], default="scripted"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--mode", choices=["parallel", "sequential"], default="parallel"
    )
    parser.add_argument("--library", type=Path, help="Library dir (default: temp).")
    args = parser.parse_args()

    executor, editor, selector, reflector = build(args.provider, args.model)
    with tempfile.TemporaryDirectory() as tmp:
        library = ExperienceLibrary(args.library or Path(tmp), mode=args.mode)
        runs = run_domain(
            [(tid, task_text(tid)) for tid in LEDGERS],
            "accounting",
            BASE,
            executor,
            editor,
            reflector,
            library,
            selector,
            k=args.k,
        )
        report = []
        for run in runs:
            res = run.result
            report.append(
                {
                    "task": run.task_id,
                    "status": res.status,
                    "base_answers": [r.answer for r in res.base_runs],
                    "selected": res.selected.label if res.selected else None,
                    "selected_answer": res.selected.answer if res.selected else None,
                    "selection_fallback": res.selection.fallback,
                    "budget": {"R": res.ledger.rollouts, "P": res.ledger.sessions},
                    "lesson_stored": run.stored,
                    "adopted_skills": sorted(res.best_harness.skills),
                }
            )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
