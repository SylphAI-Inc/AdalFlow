"""One entry point for harness optimization: model, tools and roles configured once.

    opt = HarnessOptimizer(model_client=OpenAIClient(), model_kwargs={"model": "gpt-4o-mini"},
                           tools=[search], k=2)
    res = opt.refine("Summarize the refund policy for order 1182")
    agent = opt.build_agent(res.best_harness)

or start from an agent you already have with ``HarnessOptimizer.from_agent(agent)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from adalflow.optim.harness.harness import (
    DEFAULT_ALWAYS_ON_CAP,
    SKILL_TOOL_NAME,
    Harness,
)
from adalflow.optim.harness.ledger import expected_rollouts, expected_sessions
from adalflow.optim.harness.library import DomainRun, ExperienceLibrary, run_domain
from adalflow.optim.harness.probe import HarnessExecutor, RunnerExecutor
from adalflow.optim.harness.refine import PARALLEL, RefinementResult, refine_harness
from adalflow.optim.harness.roles import (
    BlindSelector,
    ExperienceReflector,
    HarnessEditor,
)

__all__ = ["CostEstimate", "HarnessOptimizer"]

# Agent settings carried over by ``from_agent`` (besides model, tools and role_desc).
_AGENT_SETTINGS = ("max_steps", "answer_data_type", "is_thinking_model", "model_type")


@dataclass
class CostEstimate:
    """Upper bounds on model calls before running; actual usage is usually lower."""

    tasks: int
    rollouts: int
    agent_calls_max: int  # rollouts x max_steps planner calls
    editor_calls_max: int  # sessions x (1 + max_retries)
    selector_calls_max: int
    reflector_calls_max: int

    @property
    def total_calls_max(self) -> int:
        return (
            self.agent_calls_max
            + self.editor_calls_max
            + self.selector_calls_max
            + self.reflector_calls_max
        )


class HarnessOptimizer:
    """Configures the agent, editor, selector and reflector from one model setting.

    Args:
        model_client, model_kwargs: the model the optimized agent runs on. Also used
            for the optimization roles unless ``editor_client`` is given.
        base: the starting harness, or a plain system prompt string.
        tools: tools the agent gets in every rollout (a ``load_skill`` tool is added
            automatically when the harness has skills).
        agent_kwargs: extra ``Agent`` arguments (``max_steps``, ``answer_data_type``...).
            Rollouts are built with ``use_cache=False`` unless set here, so repeated
            runs of one harness are independent samples rather than cache hits.
        editor_client, editor_model_kwargs: a (usually stronger) model for the editor,
            selector and reflector.
        executor: replaces the default ``RunnerExecutor`` (e.g. a non-Runner agent).
        prompt_kwargs: maps task text to ``Runner`` prompt kwargs.
        k, mode, seed, max_retries, always_on_cap: forwarded to ``refine_harness``.
    """

    def __init__(
        self,
        model_client: Any = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        *,
        base: Union[Harness, str, None] = None,
        tools: Optional[list] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        editor_client: Any = None,
        editor_model_kwargs: Optional[Dict[str, Any]] = None,
        executor: Optional[HarnessExecutor] = None,
        prompt_kwargs: Optional[Callable[[str], Dict[str, Any]]] = None,
        k: int = 1,
        mode: str = PARALLEL,
        seed: int = 0,
        max_retries: int = 1,
        always_on_cap: int = DEFAULT_ALWAYS_ON_CAP,
    ) -> None:
        if editor_client is None and model_client is None:
            raise ValueError("pass model_client (or editor_client with an executor)")
        if executor is None and model_client is None:
            raise ValueError("model_client is required unless an executor is given")
        if isinstance(base, str):
            base = Harness(system_prompt=base)
        self.base = base if base is not None else Harness()
        self.base.validate(always_on_cap=always_on_cap)

        self.model_client = model_client
        self.model_kwargs = dict(model_kwargs or {})
        self.tools = list(tools or [])
        self.agent_kwargs = {"use_cache": False, **(agent_kwargs or {})}
        self.k, self.mode, self.seed = k, mode, seed
        self.max_retries, self.always_on_cap = max_retries, always_on_cap

        role_client = editor_client if editor_client is not None else model_client
        role_kwargs = dict(
            editor_model_kwargs
            if editor_model_kwargs is not None
            else (self.model_kwargs if editor_client is None else {})
        )
        role = {"model_client": role_client, "model_kwargs": role_kwargs}
        self.editor = HarnessEditor(**role)
        self.selector = BlindSelector(**role)
        self._role = role
        self._reflector: Optional[ExperienceReflector] = None
        self.executor = executor or RunnerExecutor(
            agent_factory=self.build_agent, prompt_kwargs=prompt_kwargs
        )

    @classmethod
    def from_agent(cls, agent: Any, **kwargs: Any) -> "HarnessOptimizer":
        """Starts from an existing ``Agent``: its role_desc, model, tools and settings.

        ``kwargs`` override anything taken from the agent. A ``load_skill`` tool the
        agent already has is dropped; the harness adds its own.
        """
        tools = [
            t
            for t in getattr(agent.tool_manager, "tools", [])
            if getattr(getattr(t, "definition", None), "func_name", None)
            != SKILL_TOOL_NAME
        ]
        agent_kwargs = {
            name: getattr(agent, name)
            for name in _AGENT_SETTINGS
            if getattr(agent, name, None) is not None
        }
        agent_kwargs.update(kwargs.pop("agent_kwargs", None) or {})
        options: Dict[str, Any] = dict(
            model_client=agent.model_client,
            model_kwargs=agent.model_kwargs,
            base=Harness.from_agent(agent),
            tools=tools,
            agent_kwargs=agent_kwargs,
        )
        options.update(kwargs)
        return cls(**options)

    # ------------------------------------------------------------------- agents
    def build_agent(self, harness: Optional[Harness] = None, name: str = "agent"):
        """An ``Agent`` with this optimizer's model and tools, configured by ``harness``.

        For deployment, pass ``use_cache`` in ``agent_kwargs`` if you want caching.
        """
        harness = harness if harness is not None else self.base
        return harness.to_agent(
            name=name,
            tools=self.tools,
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            **self.agent_kwargs,
        )

    # ----------------------------------------------------------------- running
    def refine(
        self,
        task: str,
        *,
        task_id: str = "task",
        base: Optional[Harness] = None,
        library: str = "",
    ) -> RefinementResult:
        """Refines the harness on one task (R = 4k rollouts, P = 3k editor sessions)."""
        return refine_harness(
            task,
            base if base is not None else self.base,
            self.executor,
            self.editor,
            self.selector,
            k=self.k,
            mode=self.mode,
            task_id=task_id,
            library=library,
            seed=self.seed,
            max_retries=self.max_retries,
            always_on_cap=self.always_on_cap,
        )

    @property
    def reflector(self) -> ExperienceReflector:
        if self._reflector is None:
            self._reflector = ExperienceReflector(**self._role)
        return self._reflector

    def run_domain(
        self,
        tasks: Sequence[Tuple[str, str]],
        domain: str,
        library: Union[ExperienceLibrary, str, Path],
        *,
        base: Optional[Harness] = None,
        include_existing: bool = False,
    ) -> List[DomainRun]:
        """Refines each ``(task_id, task_text)`` in order, storing lessons in ``library``.

        A path creates an ``ExperienceLibrary`` for this optimizer's mode.
        """
        if not isinstance(library, ExperienceLibrary):
            library = ExperienceLibrary(library, mode=self.mode)
        elif library.mode != self.mode:
            raise ValueError(
                f"library mode {library.mode!r} does not match optimizer mode "
                f"{self.mode!r}; keep one library per mode"
            )
        return run_domain(
            tasks,
            domain,
            base if base is not None else self.base,
            self.executor,
            self.editor,
            self.reflector,
            library,
            self.selector,
            k=self.k,
            seed=self.seed,
            max_retries=self.max_retries,
            include_existing=include_existing,
            always_on_cap=self.always_on_cap,
        )

    def estimate(self, n_tasks: int = 1, *, reflect: bool = False) -> CostEstimate:
        """Upper bounds on model calls for ``n_tasks`` (``reflect`` for run_domain)."""
        attempts = 1 + self.max_retries
        rollouts = expected_rollouts(self.k) * n_tasks
        max_steps = self.agent_kwargs.get("max_steps") or 10
        return CostEstimate(
            tasks=n_tasks,
            rollouts=rollouts,
            agent_calls_max=rollouts * max_steps,
            editor_calls_max=expected_sessions(self.k) * attempts * n_tasks,
            selector_calls_max=(attempts if self.k > 1 else 0) * n_tasks,
            reflector_calls_max=(attempts if reflect else 0) * n_tasks,
        )
