"""Probing a harness: run the agent once on a task and keep what it did.

A probe never raises. A failed run is recorded with ``ok=False`` so the recursion
can treat it as evidence (or skip it), the way a dead rollout is still a data point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from adalflow.optim.harness.harness import Harness

__all__ = [
    "ProbeResult",
    "HarnessExecutor",
    "RunnerExecutor",
    "render_trajectory",
]


@dataclass
class ProbeResult:
    """One execution of one harness on one task.

    ``delivered`` means the run produced a non-empty answer; it is not a score.
    """

    label: str
    harness: Harness
    answer: Optional[str] = None
    trajectory: str = ""
    steps: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def delivered(self) -> bool:
        return self.ok and bool((self.answer or "").strip())


class HarnessExecutor(Protocol):
    """Runs a harness on a task. Implementations must not raise for run failures."""

    def execute(self, harness: Harness, task: str, label: str) -> ProbeResult: ...


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [{len(text) - limit} chars truncated]"


def _format_call(function: Any, args_trunc: int) -> str:
    name = getattr(function, "name", None) or "?"
    kwargs = getattr(function, "kwargs", None) or {}
    args = getattr(function, "args", None) or []
    try:
        payload = json.dumps({"args": list(args), **dict(kwargs)}, default=str)
    except (TypeError, ValueError):
        payload = str(kwargs)
    return f"{name}({_clip(payload, args_trunc)})"


def render_trajectory(
    step_history: List[Any],
    answer: Optional[str] = None,
    error: Optional[str] = None,
    *,
    obs_trunc: int = 1500,
    args_trunc: int = 400,
    max_chars: int = 20_000,
) -> str:
    """Renders a ``Runner`` step history into one plain-text trajectory."""
    lines: List[str] = []
    for i, step in enumerate(step_history, start=1):
        function = getattr(step, "function", None)
        thought = getattr(function, "thought", None) if function is not None else None
        if thought:
            lines.append(f"[THOUGHT {i}] {_clip(str(thought), obs_trunc)}")
        if function is not None:
            lines.append(f"[ACTION {i}] {_format_call(function, args_trunc)}")
        observation = getattr(step, "observation", None)
        if observation is not None:
            lines.append(f"[OBSERVATION {i}] {_clip(str(observation), obs_trunc)}")
    if answer is not None:
        lines.append(f"[ANSWER] {_clip(str(answer), obs_trunc)}")
    if error:
        lines.append(f"[ERROR] {_clip(str(error), obs_trunc)}")
    rendered = "\n".join(lines) or "(no steps recorded)"
    return _clip(rendered, max_chars)


class RunnerExecutor:
    """Executes a harness with an AdalFlow ``Runner``.

    Args:
        agent_factory: builds an agent from a harness, typically
            ``lambda h: h.to_agent(model_client=..., model_kwargs=..., tools=...)``.
        prompt_kwargs: maps the task text to ``Runner.call`` prompt kwargs;
            defaults to ``{"input_str": task}``.
        max_steps: forwarded to ``Runner``.
        render_kwargs: forwarded to ``render_trajectory``.
    """

    def __init__(
        self,
        agent_factory: Callable[[Harness], Any],
        prompt_kwargs: Optional[Callable[[str], Dict[str, Any]]] = None,
        max_steps: Optional[int] = None,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.prompt_kwargs = prompt_kwargs or (lambda task: {"input_str": task})
        self.max_steps = max_steps
        self.render_kwargs = dict(render_kwargs or {})

    def execute(self, harness: Harness, task: str, label: str) -> ProbeResult:
        from adalflow.components.agent.runner import Runner

        try:
            agent = self.agent_factory(harness)
            runner = (
                Runner(agent=agent, max_steps=self.max_steps)
                if self.max_steps is not None
                else Runner(agent=agent)
            )
            result = runner.call(prompt_kwargs=self.prompt_kwargs(task))
        except Exception as exc:  # a probe records failures instead of raising
            error = f"{type(exc).__name__}: {exc}"
            return ProbeResult(
                label=label,
                harness=harness,
                error=error,
                trajectory=render_trajectory([], error=error, **self.render_kwargs),
            )

        steps = list(getattr(result, "step_history", None) or [])
        answer = getattr(result, "answer", None)
        error = getattr(result, "error", None)
        return ProbeResult(
            label=label,
            harness=harness,
            answer=None if answer is None else str(answer),
            trajectory=render_trajectory(steps, answer, error, **self.render_kwargs),
            steps=len(steps),
            error=error,
        )
