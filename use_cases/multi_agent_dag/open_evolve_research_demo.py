"""
OpenEvolve-style autonomous research loop demo built with AdalFlow Components.

This is a deterministic, local smoke-testable prototype by default. It models the
structure of an evolutionary research agent:

    seed -> mutate -> execute -> evaluate -> reflect -> archive -> repeat

Phase 1 also supports optional LLM-backed seed/mutation/reflection/report writing.
The executor and evaluator stay deterministic/safe so benchmark runs remain
reproducible unless explicitly switched to LLM mode.

Note: the executor is a mock. ``eval_accuracy`` is ``0.65 + 0.04 * generation`` plus a
hash-based term, so scores rise with generation by construction, even in LLM mode.
For real execution of candidates, see ``adalflow.optim.evolution``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from adalflow.components.model_client import AnthropicAPIClient, GroqAPIClient, OpenAIClient
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter, ParameterType


@dataclass
class ResearchCandidate:
    """A candidate research direction/program in the evolution loop."""

    candidate_id: str
    parent_id: Optional[str]
    generation: int
    topic: str
    hypothesis: str
    experiment_code: str
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    repaired: bool = False


@dataclass
class ResearchState:
    """Mutable state for one autonomous research run."""

    topic: str
    candidates: List[ResearchCandidate] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    iteration: int = 0

    @property
    def best_candidate(self) -> Optional[ResearchCandidate]:
        if self.best_candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.best_candidate_id:
                return candidate
        return None


def _stable_unit_interval(*parts: str) -> float:
    """Return a deterministic float in [0, 1) for reproducible smoke tests."""

    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _generator_text(response: Any) -> str:
    """Extract a text payload from an AdalFlow Generator response."""

    data = getattr(response, "data", response)
    if data is None:
        raw_response = getattr(response, "raw_response", None)
        return "" if raw_response is None else str(raw_response)
    return str(data).strip()


def _build_model_client(provider: str):
    """Create a model client for optional LLM-backed demo components."""

    if provider == "openai":
        return OpenAIClient()
    if provider == "groq":
        return GroqAPIClient()
    if provider == "anthropic":
        return AnthropicAPIClient()
    raise ValueError(f"Unsupported provider: {provider}")


class LLMBackedComponent(Component):
    """Small helper for optional Generator-backed research agents."""

    def __init__(self, *, provider: str, model_kwargs: Dict[str, Any], template: str):
        super().__init__()
        self.generator = Generator(
            model_client=_build_model_client(provider),
            model_kwargs=model_kwargs,
            template=template,
        )

    def generate_text(self, prompt_kwargs: Dict[str, Any]) -> str:
        return _generator_text(self.generator(prompt_kwargs=prompt_kwargs))


class SeedAgent(Component):
    """Create the initial hypothesis and experiment sketch."""

    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data=(
                "Create an initial research candidate with a concrete hypothesis, "
                "simple experiment plan, and measurable success criterion."
            ),
            requires_opt=True,
            role_desc="Seed instructions for the first research candidate.",
            param_type=ParameterType.PROMPT,
        )

    def call(self, topic: str) -> ResearchCandidate:
        return ResearchCandidate(
            candidate_id="cand-000",
            parent_id=None,
            generation=0,
            topic=topic,
            hypothesis=f"Initial hypothesis for: {topic}",
            experiment_code="# initial experiment\nmetric = baseline_score",
        )


class LLMSeedAgent(LLMBackedComponent):
    """LLM-backed seed agent that drafts the first research hypothesis."""

    def __init__(self, *, provider: str, model_kwargs: Dict[str, Any]):
        super().__init__(
            provider=provider,
            model_kwargs=model_kwargs,
            template=(
                "You are a research agent creating the first candidate for an "
                "evolutionary autonomous research loop.\n\n"
                "Topic: {{topic}}\n\n"
                "Return a concise hypothesis and experiment sketch. Keep it specific, "
                "measurable, and safe to evaluate in a sandbox."
            ),
        )

    def call(self, topic: str) -> ResearchCandidate:
        hypothesis = self.generate_text({"topic": topic})
        if not hypothesis:
            hypothesis = f"Initial hypothesis for: {topic}"

        return ResearchCandidate(
            candidate_id="cand-000",
            parent_id=None,
            generation=0,
            topic=topic,
            hypothesis=hypothesis,
            experiment_code="# llm-seeded experiment\nmetric = baseline_score",
        )


class MutationAgent(Component):
    """Generate child candidates from the current best candidate and lessons."""

    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data=(
                "Mutate the current research candidate using evaluator feedback. "
                "Prefer small, testable changes with clear measurement hooks."
            ),
            requires_opt=True,
            role_desc="Mutation policy for improving research candidates.",
            param_type=ParameterType.PROMPT,
        )

    def call(
        self,
        parent: ResearchCandidate,
        *,
        topic: str,
        lessons: List[str],
        generation: int,
    ) -> ResearchCandidate:
        lesson_hint = lessons[-1] if lessons else "start with a measurable baseline"
        candidate_id = f"cand-{generation:03d}"
        hypothesis = (
            f"{parent.hypothesis} | refinement {generation}: "
            f"apply lesson '{lesson_hint}'"
        )

        code = (
            f"# candidate {candidate_id}\n"
            f"# parent: {parent.candidate_id}\n"
            "metric = baseline_score\n"
            f"improvement_step = {generation}\n"
        )

        # Odd generations intentionally omit eval_accuracy so the repair path is exercised.
        if generation % 2 == 0:
            code += "eval_accuracy = 0.70 + 0.03 * improvement_step\n"

        return ResearchCandidate(
            candidate_id=candidate_id,
            parent_id=parent.candidate_id,
            generation=generation,
            topic=topic,
            hypothesis=hypothesis,
            experiment_code=code,
        )


class LLMMutationAgent(LLMBackedComponent):
    """LLM-backed mutation agent.

    The LLM mutates the research direction, while the experiment code remains a
    deterministic skeleton for safe smoke benchmarking.
    """

    def __init__(self, *, provider: str, model_kwargs: Dict[str, Any]):
        super().__init__(
            provider=provider,
            model_kwargs=model_kwargs,
            template=(
                "You are mutating a research candidate in an OpenEvolve-style loop.\n\n"
                "Topic: {{topic}}\n"
                "Generation: {{generation}}\n"
                "Parent candidate:\n{{parent}}\n\n"
                "Recent lessons:\n{{lessons}}\n\n"
                "Return one improved hypothesis and experiment direction. Prefer a "
                "small testable change, not a broad rewrite."
            ),
        )

    def call(
        self,
        parent: ResearchCandidate,
        *,
        topic: str,
        lessons: List[str],
        generation: int,
    ) -> ResearchCandidate:
        candidate_id = f"cand-{generation:03d}"
        hypothesis = self.generate_text(
            {
                "topic": topic,
                "generation": generation,
                "parent": asdict(parent),
                "lessons": "\n".join(lessons[-3:]) if lessons else "No prior lessons.",
            }
        )
        if not hypothesis:
            hypothesis = f"{parent.hypothesis} | LLM refinement {generation}"

        code = (
            f"# candidate {candidate_id}\n"
            f"# parent: {parent.candidate_id}\n"
            "metric = baseline_score\n"
            f"improvement_step = {generation}\n"
        )
        if generation % 2 == 0:
            code += "eval_accuracy = 0.70 + 0.03 * improvement_step\n"

        return ResearchCandidate(
            candidate_id=candidate_id,
            parent_id=parent.candidate_id,
            generation=generation,
            topic=topic,
            hypothesis=hypothesis,
            experiment_code=code,
        )


class EvolutionSandboxExecutor(Component):
    """Mock sandbox executor with deterministic repair behavior."""

    def call(self, candidate: ResearchCandidate) -> ResearchCandidate:
        if "eval_accuracy" not in candidate.experiment_code:
            candidate.metrics["first_attempt_error"] = "Missing eval_accuracy metric"
            candidate.experiment_code += "eval_accuracy = 0.68 + 0.025 * improvement_step\n"
            candidate.repaired = True

        candidate.metrics["status"] = "success"
        candidate.metrics["eval_accuracy"] = round(
            0.65
            + 0.04 * candidate.generation
            + 0.03 * _stable_unit_interval(candidate.topic, candidate.candidate_id),
            4,
        )
        return candidate


class ResearchEvaluator(Component):
    """Score candidates from execution metrics and lightweight quality signals."""

    def call(self, candidate: ResearchCandidate) -> float:
        accuracy = float(candidate.metrics.get("eval_accuracy", 0.0))
        novelty_bonus = 0.02 * min(candidate.generation, 3)
        repair_penalty = 0.01 if candidate.repaired else 0.0
        score = accuracy + novelty_bonus - repair_penalty
        candidate.score = round(score, 4)
        return candidate.score


class ReflectorAgent(Component):
    """Convert candidate outcomes into reusable lessons for the next mutation."""

    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data=(
                "Diagnose candidate results and produce concise, reusable lessons "
                "for the next mutation."
            ),
            requires_opt=True,
            role_desc="Reflection instructions for extracting lessons from runs.",
            param_type=ParameterType.PROMPT,
        )

    def call(self, candidate: ResearchCandidate) -> str:
        if candidate.repaired:
            feedback = (
                f"{candidate.candidate_id}: keep explicit metric logging; repaired missing "
                "eval_accuracy before scoring."
            )
        else:
            feedback = (
                f"{candidate.candidate_id}: metric logging was complete; continue refining "
                "experiment specificity."
            )
        candidate.feedback = feedback
        return feedback


class LLMReflectorAgent(LLMBackedComponent):
    """LLM-backed reflection agent for lesson extraction."""

    def __init__(self, *, provider: str, model_kwargs: Dict[str, Any]):
        super().__init__(
            provider=provider,
            model_kwargs=model_kwargs,
            template=(
                "You are reflecting on an evolutionary research candidate.\n\n"
                "Candidate:\n{{candidate}}\n\n"
                "Write one concise reusable lesson for the next mutation. Mention "
                "metric logging, repair, novelty, or experiment design only if relevant."
            ),
        )

    def call(self, candidate: ResearchCandidate) -> str:
        feedback = self.generate_text({"candidate": asdict(candidate)})
        if not feedback:
            feedback = ReflectorAgent().call(candidate)
        candidate.feedback = feedback
        return feedback


class CandidateArchive(Component):
    """Track lineage, lessons, and best candidate selection."""

    def call(
        self,
        state: ResearchState,
        candidate: ResearchCandidate,
        feedback: str,
    ) -> ResearchState:
        state.candidates.append(candidate)
        state.lessons.append(feedback)

        best = state.best_candidate
        if best is None or candidate.score >= best.score:
            state.best_candidate_id = candidate.candidate_id

        return state


class FinalReportWriter(Component):
    """Produce a final report artifact from the best candidate."""

    def call(self, state: ResearchState) -> Dict[str, Any]:
        best = state.best_candidate
        if best is None:
            return {
                "output_status": "Failed to produce final report.",
                "paper_generated": False,
            }

        return {
            "output_status": "Successfully drafted evolution-loop research report.",
            "paper_generated": True,
            "hypothesis": best.hypothesis,
            "best_candidate": asdict(best),
            "lessons": state.lessons,
        }


class LLMFinalReportWriter(LLMBackedComponent):
    """LLM-backed report writer for final research summaries."""

    def __init__(self, *, provider: str, model_kwargs: Dict[str, Any]):
        super().__init__(
            provider=provider,
            model_kwargs=model_kwargs,
            template=(
                "You are writing a concise research progress report from an "
                "OpenEvolve-style autonomous research loop.\n\n"
                "Topic: {{topic}}\n"
                "Best candidate:\n{{best_candidate}}\n\n"
                "Lessons:\n{{lessons}}\n\n"
                "Return Markdown with sections: Hypothesis, Experiment, Results, "
                "Lessons, Next Steps."
            ),
        )

    def call(self, state: ResearchState) -> Dict[str, Any]:
        best = state.best_candidate
        if best is None:
            return {
                "output_status": "Failed to produce final report.",
                "paper_generated": False,
            }

        report_markdown = self.generate_text(
            {
                "topic": state.topic,
                "best_candidate": asdict(best),
                "lessons": "\n".join(state.lessons[-5:]),
            }
        )

        return {
            "output_status": "Successfully drafted LLM-backed evolution-loop research report.",
            "paper_generated": True,
            "hypothesis": best.hypothesis,
            "best_candidate": asdict(best),
            "lessons": state.lessons,
            "report_markdown": report_markdown,
        }


class OpenEvolveResearchLoop(Component):
    """OpenEvolve-style research loop orchestrator."""

    def __init__(
        self,
        max_iterations: int = 5,
        target_score: float = 0.86,
        *,
        use_llm: bool = False,
        provider: str = "openai",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.use_llm = use_llm

        llm_model_kwargs = model_kwargs or {"model": "gpt-4o-mini", "temperature": 0.3}

        if use_llm:
            self.seed_agent = LLMSeedAgent(provider=provider, model_kwargs=llm_model_kwargs)
            self.mutator = LLMMutationAgent(provider=provider, model_kwargs=llm_model_kwargs)
            self.reflector = LLMReflectorAgent(provider=provider, model_kwargs=llm_model_kwargs)
            self.final_writer = LLMFinalReportWriter(provider=provider, model_kwargs=llm_model_kwargs)
        else:
            self.seed_agent = SeedAgent()
            self.mutator = MutationAgent()
            self.reflector = ReflectorAgent()
            self.final_writer = FinalReportWriter()

        self.executor = EvolutionSandboxExecutor()
        self.evaluator = ResearchEvaluator()
        self.archive = CandidateArchive()

    def call(self, topic: str) -> Dict[str, Any]:
        print(f"\n===== OpenEvolveResearchLoop Topic: '{topic}' =====")
        state = ResearchState(topic=topic)

        seed = self.seed_agent(topic)
        seed = self.executor(seed)
        self.evaluator(seed)
        feedback = self.reflector(seed)
        state = self.archive(state, seed, feedback)

        initial_score = seed.score
        retry_count = 1 if seed.repaired else 0
        convergence_iteration = seed.generation

        for generation in range(1, self.max_iterations + 1):
            state.iteration = generation
            parent = state.best_candidate
            if parent is None:
                break

            candidate = self.mutator(
                parent,
                topic=topic,
                lessons=state.lessons,
                generation=generation,
            )
            candidate = self.executor(candidate)
            if candidate.repaired:
                retry_count += 1

            self.evaluator(candidate)
            feedback = self.reflector(candidate)
            previous_best = state.best_candidate
            state = self.archive(state, candidate, feedback)

            if state.best_candidate is not None and (
                previous_best is None or state.best_candidate.candidate_id != previous_best.candidate_id
            ):
                convergence_iteration = generation

            print(
                f"[Evolution] generation={generation} "
                f"candidate={candidate.candidate_id} score={candidate.score:.4f} "
                f"best={state.best_candidate.score:.4f}"
            )

            if state.best_candidate and state.best_candidate.score >= self.target_score:
                break

        report = self.final_writer(state)
        best = state.best_candidate
        best_score = best.score if best else 0.0

        result = {
            **report,
            "llm_enabled": self.use_llm,
            "experiment_metrics": {
                "best_score": best_score,
                "initial_score": initial_score,
                "score_improvement": round(best_score - initial_score, 4),
                "num_candidates": len(state.candidates),
                "retry_count": retry_count,
                "convergence_iteration": convergence_iteration,
                "archive_size": len(state.candidates),
            },
            "evolution_history": [asdict(candidate) for candidate in state.candidates],
        }
        return result


def demo_run() -> None:
    loop = OpenEvolveResearchLoop(max_iterations=5)
    result = loop(
        "Investigate whether iterative prompt mutation improves autonomous research reliability."
    )
    print("\n" + "=" * 60)
    print("OPENEVOLVE-STYLE DEMO SUMMARY")
    print(f"Best score: {result['experiment_metrics']['best_score']}")
    print(f"Improvement: {result['experiment_metrics']['score_improvement']}")
    print(f"Candidates: {result['experiment_metrics']['num_candidates']}")
    print(f"Retries: {result['experiment_metrics']['retry_count']}")
    print(f"LLM enabled: {result['llm_enabled']}")
    print(f"Status: {result['output_status']}")
    print("=" * 60)


if __name__ == "__main__":
    demo_run()
