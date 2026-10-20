"""Harnessing-level recursion: a persistent library of adopted harnesses and lessons.

After each task, the adopted harness and an evidence-supported lesson are stored per
refinement mode and domain. Later tasks in the same domain see what earlier tasks
stored; each task still starts from the base harness. After Harness²
(arXiv:2610.14224), implemented independently.

Layout::

    <root>/<mode>/<domain>/index.json             # task ids in insertion order
    <root>/<mode>/<domain>/<task_id>/harness.json
    <root>/<mode>/<domain>/<task_id>/experience.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from adalflow.optim.harness.harness import Harness
from adalflow.optim.harness.probe import HarnessExecutor
from adalflow.optim.harness.refine import (
    PARALLEL,
    RefinementResult,
    _evidence,
    _PoolEntry,
    _section,
    refine_harness,
)
from adalflow.optim.harness.roles import (
    BlindSelector,
    EditVerdict,
    Experience,
    ExperienceReflector,
    HarnessEditor,
    Reflection,
    RoleOutputError,
)

__all__ = ["LibraryEntry", "ExperienceLibrary", "DomainRun", "run_domain"]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(name: str) -> str:
    slug = _SAFE.sub("-", name).strip("-.")
    if not slug:
        raise ValueError(f"cannot make a directory name from {name!r}")
    return slug


@dataclass
class LibraryEntry:
    task_id: str
    domain: str
    harness: Harness
    experience: Experience
    chosen_label: str = ""
    why_chosen: str = ""
    edits: List[EditVerdict] = field(default_factory=list)

    def render(self) -> str:
        rows = "\n".join(
            f"- {e.component} [{e.verdict}] ({e.candidates}): {e.change} -> {e.observed}"
            for e in self.edits
        )
        exp = self.experience
        return (
            f"### From task {self.task_id}\n"
            f"Cue: {exp.cue}\nLesson: {exp.lesson}\nEvidence: {exp.evidence}\n"
            f"Confidence: {exp.confidence}\n"
            f"Edit verdicts:\n{rows or '(none)'}\n"
            f"Adopted harness:\n"
            f"{json.dumps(self.harness.to_dict(), indent=2, ensure_ascii=False)}"
        )


class ExperienceLibrary:
    """Stores adopted harnesses with lessons, per refinement mode and domain.

    Keep one library per mode: parallel and sequential runs must not share lessons.
    """

    def __init__(self, root: Union[str, Path], mode: str = PARALLEL) -> None:
        self.root = Path(root)
        self.mode = mode

    def _dir(self, domain: str) -> Path:
        return self.root / _slug(self.mode) / _slug(domain)

    def _index(self, domain: str) -> List[str]:
        path = self._dir(domain) / "index.json"
        if not path.is_file():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")))

    def add(self, entry: LibraryEntry) -> bool:
        """Stores ``entry`` if its lesson is supported; returns whether it was stored."""
        if not entry.experience.is_supported:
            return False
        ddir = self._dir(entry.domain)
        tdir = ddir / _slug(entry.task_id)
        tdir.mkdir(parents=True, exist_ok=True)
        entry.harness.save(tdir / "harness.json")
        record = {
            "task_id": entry.task_id,
            "domain": entry.domain,
            "chosen_label": entry.chosen_label,
            "why_chosen": entry.why_chosen,
            "experience": asdict(entry.experience),
            "edits": [asdict(e) for e in entry.edits],
        }
        (tdir / "experience.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        index = [t for t in self._index(entry.domain) if t != entry.task_id]
        index.append(entry.task_id)
        (ddir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        return True

    def entries(
        self, domain: str, only: Optional[Sequence[str]] = None
    ) -> List[LibraryEntry]:
        """Entries for ``domain`` in insertion order, optionally limited to ``only``."""
        allowed = None if only is None else set(only)
        out = []
        for task_id in self._index(domain):
            if allowed is not None and task_id not in allowed:
                continue
            tdir = self._dir(domain) / _slug(task_id)
            raw = json.loads((tdir / "experience.json").read_text(encoding="utf-8"))
            out.append(
                LibraryEntry(
                    task_id=raw["task_id"],
                    domain=raw["domain"],
                    harness=Harness.load(tdir / "harness.json"),
                    experience=Experience(**raw["experience"]),
                    chosen_label=raw.get("chosen_label", ""),
                    why_chosen=raw.get("why_chosen", ""),
                    edits=[EditVerdict(**e) for e in raw.get("edits", [])],
                )
            )
        return out

    def render(self, domain: str, only: Optional[Sequence[str]] = None) -> str:
        """Editor-facing text for ``domain``; empty when there is nothing stored."""
        entries = self.entries(domain, only)
        if not entries:
            return ""
        return (
            "Lessons and adopted harnesses from earlier tasks in this domain. Reuse a "
            "component only if this task's runs support it, and re-verify any "
            "workspace facts it states.\n\n" + "\n\n".join(e.render() for e in entries)
        )


@dataclass
class DomainRun:
    """Per-task outcome of ``run_domain``."""

    task_id: str
    result: RefinementResult
    reflection: Optional[Reflection] = None
    stored: bool = False
    reflect_error: str = ""


def _reflection_evidence(task: str, res: RefinementResult) -> str:
    pool = [
        _PoolEntry(stage, run)
        for run in res.candidates + res.finals
        for stage in res.stages
        if stage.label == run.label
    ]
    adopted = (
        f"Adopted final: {res.selected.label}\n"
        f"Selector's reason: {res.selection.why or '(trivial or fallback selection)'}"
    )
    return "\n\n".join(
        [
            _evidence(task, res.base, res.base_runs, "", pools=(("Candidates", pool),)),
            _section("Adoption", adopted),
        ]
    )


def run_domain(
    tasks: Sequence[Tuple[str, str]],
    domain: str,
    base: Harness,
    executor: HarnessExecutor,
    editor: HarnessEditor,
    reflector: ExperienceReflector,
    library: ExperienceLibrary,
    selector: Optional[BlindSelector] = None,
    *,
    k: int = 1,
    seed: int = 0,
    max_retries: int = 1,
    include_existing: bool = False,
    **refine_kwargs: Any,
) -> List[DomainRun]:
    """Refines each ``(task_id, task_text)`` in order, growing ``library`` as it goes.

    Each task starts from ``base``. The editor sees lessons from the tasks of this
    call that came before it; with ``include_existing=True`` it also sees entries the
    domain's library already held when the call started (default off, so a run is
    reproducible from its task order alone).
    """
    runs: List[DomainRun] = []
    seen: List[str] = (
        [e.task_id for e in library.entries(domain)] if include_existing else []
    )
    for task_id, task in tasks:
        res = refine_harness(
            task,
            base,
            executor,
            editor,
            selector,
            k=k,
            mode=library.mode,
            task_id=task_id,
            library=library.render(domain, only=seen),
            seed=seed,
            max_retries=max_retries,
            **refine_kwargs,
        )
        run = DomainRun(task_id=task_id, result=res)
        if res.status == "ok":
            evidence, feedback = _reflection_evidence(task, res), ""
            for _ in range(max_retries + 1):
                try:
                    run.reflection = reflector.reflect(evidence, feedback=feedback)
                    break
                except RoleOutputError as exc:
                    feedback = run.reflect_error = str(exc)
                except Exception as exc:
                    run.reflect_error = f"{type(exc).__name__}: {exc}"
                    feedback = "the previous call failed; reply again"
            if run.reflection is not None and run.reflection.experience is not None:
                run.stored = library.add(
                    LibraryEntry(
                        task_id=task_id,
                        domain=domain,
                        harness=res.best_harness,
                        experience=run.reflection.experience,
                        chosen_label=res.selected.label,
                        why_chosen=res.selection.why,
                        edits=run.reflection.edits,
                    )
                )
                if run.stored:
                    run.reflect_error = ""
        runs.append(run)
        seen.append(task_id)
    return runs
