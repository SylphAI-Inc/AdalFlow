"""An editable agent harness: the prompt-side system around a fixed model.

Inspired by Harness² (Xu et al., 2026, arXiv:2610.14224), which optimizes an agent's
harness -- system prompt, guardrails, memory, tool guidance, skills and more -- while
keeping model weights fixed. This is an independent implementation for AdalFlow agents.

A ``Harness`` has five text components. Four are always on (rendered into the agent's
``role_desc``); skills are loaded on demand through a ``load_skill`` tool, and only
their one-line descriptions are always on. Executable components (scripts, plugins)
are deliberately not supported: an optimizer would be writing code that then runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "ALWAYS_ON_COMPONENTS",
    "HarnessSkill",
    "Harness",
    "HarnessDiff",
    "HarnessValidationError",
]

ALWAYS_ON_COMPONENTS = ("system_prompt", "guardrails", "memory", "tool_guidance")
DEFAULT_ALWAYS_ON_CAP = 24_000
SKILL_TOOL_NAME = "load_skill"
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SECTION_TITLES = {
    "guardrails": "Guardrails",
    "memory": "Memory",
    "tool_guidance": "Tool guidance",
}


class HarnessValidationError(ValueError):
    """Raised with every problem found, so an editor can fix them in one retry."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class HarnessSkill:
    """An on-demand skill: ``description`` is always on, ``body`` loads on request."""

    description: str
    body: str


@dataclass
class HarnessDiff:
    """Component-level changes from one harness to another."""

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def summary(self) -> str:
        if self.is_empty:
            return "no changes"
        parts = []
        for verb, names in (
            ("added", self.added),
            ("removed", self.removed),
            ("modified", self.modified),
        ):
            if names:
                parts.append(f"{verb}: {', '.join(names)}")
        return "; ".join(parts)


@dataclass
class Harness:
    """The editable components around a model that make it act as an agent."""

    system_prompt: str = ""
    guardrails: str = ""
    memory: str = ""
    tool_guidance: str = ""
    skills: Dict[str, HarnessSkill] = field(default_factory=dict)

    # ---------------------------------------------------------------- rendering
    def render_role_desc(self) -> str:
        """Renders the always-on text an ``Agent`` receives as ``role_desc``."""
        sections = [self.system_prompt.strip()] if self.system_prompt.strip() else []
        for name, title in _SECTION_TITLES.items():
            text = getattr(self, name).strip()
            if text:
                sections.append(f"## {title}\n{text}")
        if self.skills:
            index = "\n".join(
                f"- {name}: {self.skills[name].description.strip()}"
                for name in sorted(self.skills)
            )
            sections.append(
                "## Skills\n"
                f"Call `{SKILL_TOOL_NAME}` with a skill name to read its full "
                f"instructions when its description applies:\n{index}"
            )
        return "\n\n".join(sections)

    def load_skill(self, name: str) -> str:
        """Returns a skill's instructions, or the list of known skills."""
        skill = self.skills.get(name.strip())
        if skill is None:
            return (
                f"Unknown skill {name!r}. Available: {', '.join(sorted(self.skills))}"
            )
        return skill.body

    def skill_tool(self):
        """A plain ``load_skill(name)`` function for the agent's tool list.

        A bound method would be registered under a class-prefixed tool name, which
        would not match the name the rendered prompt tells the agent to call.
        """
        harness = self

        def load_skill(name: str) -> str:
            """Load the full instructions of a skill listed under Skills."""
            return harness.load_skill(name)

        return load_skill

    def to_agent(
        self, name: str = "harnessed_agent", tools: Optional[list] = None, **kwargs
    ):
        """Builds an ``adalflow`` ``Agent`` configured by this harness.

        ``kwargs`` are passed to ``Agent`` (``model_client``, ``model_kwargs``,
        ``max_steps``, ...). A ``load_skill`` tool is added when there are skills.
        """
        from adalflow.components.agent import Agent
        from adalflow.core.func_tool import FunctionTool

        all_tools = list(tools or [])
        if self.skills:
            all_tools.append(FunctionTool(self.skill_tool()))
        return Agent(
            name=name, tools=all_tools, role_desc=self.render_role_desc(), **kwargs
        )

    # --------------------------------------------------------------- validation
    def validate(self, always_on_cap: int = DEFAULT_ALWAYS_ON_CAP) -> None:
        """Raises ``HarnessValidationError`` listing every problem found."""
        errors: List[str] = []
        for name in ALWAYS_ON_COMPONENTS:
            if not isinstance(getattr(self, name), str):
                errors.append(f"{name} must be a string")
        if errors:
            raise HarnessValidationError(errors)

        rendered = len(self.render_role_desc())
        if rendered > always_on_cap:
            errors.append(
                f"always-on text is {rendered} chars, over the {always_on_cap} cap; "
                "move detail into skills"
            )
        for name, skill in sorted(self.skills.items()):
            if not _SKILL_NAME.match(name):
                errors.append(
                    f"skill name {name!r} must match {_SKILL_NAME.pattern} "
                    "(lowercase letters, digits, '-' or '_')"
                )
            if not isinstance(skill, HarnessSkill):
                errors.append(f"skill {name!r} must be a HarnessSkill")
                continue
            if not skill.description.strip() or "\n" in skill.description.strip():
                errors.append(f"skill {name!r} needs a one-line description")
            if not skill.body.strip():
                errors.append(f"skill {name!r} has an empty body")
        if errors:
            raise HarnessValidationError(errors)

    # ------------------------------------------------------------------- diffing
    def components(self) -> Dict[str, str]:
        """Flat ``{component: text}`` view; skills appear as ``skills/<name>``."""
        flat = {name: getattr(self, name) for name in ALWAYS_ON_COMPONENTS}
        for name, skill in self.skills.items():
            flat[f"skills/{name}"] = f"{skill.description}\n\n{skill.body}"
        return flat

    def diff(self, other: "Harness") -> HarnessDiff:
        """Changes needed to turn ``self`` into ``other``.

        An always-on component that goes from empty to non-empty counts as added.
        """
        before, after = self.components(), other.components()
        out = HarnessDiff()
        for key in sorted(set(before) | set(after)):
            old, new = before.get(key, "").strip(), after.get(key, "").strip()
            if old == new:
                continue
            if not old:
                out.added.append(key)
            elif not new:
                out.removed.append(key)
            else:
                out.modified.append(key)
        return out

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            name: getattr(self, name) for name in ALWAYS_ON_COMPONENTS
        }
        out["skills"] = {
            name: {"description": s.description, "body": s.body}
            for name, s in sorted(self.skills.items())
        }
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Harness":
        """Builds a harness, ignoring unknown keys; raises on malformed skills."""
        if not isinstance(data, dict):
            raise HarnessValidationError(["harness must be a JSON object"])
        skills_raw = data.get("skills") or {}
        if not isinstance(skills_raw, dict):
            raise HarnessValidationError(["skills must be an object keyed by name"])
        skills: Dict[str, HarnessSkill] = {}
        errors: List[str] = []
        for name, raw in skills_raw.items():
            if not isinstance(raw, dict):
                errors.append(f"skill {name!r} must be an object")
                continue
            skills[str(name)] = HarnessSkill(
                description=str(raw.get("description") or ""),
                body=str(raw.get("body") or ""),
            )
        if errors:
            raise HarnessValidationError(errors)
        return cls(
            **{name: str(data.get(name) or "") for name in ALWAYS_ON_COMPONENTS},
            skills=skills,
        )

    def copy(self) -> "Harness":
        return Harness.from_dict(self.to_dict())

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Harness":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
