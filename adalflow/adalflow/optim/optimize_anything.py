"""optimize_anything: evolutionary search over arbitrary text artifacts.

The API shape follows GEPA's ``optimize_anything``, but the search is heuristic: no
LLM is called. Mutations are fixed string operations (append or prepend a hint line,
swap two lines, collapse whitespace), crossover splices lines from two parents, and
the Pareto front is over (score, token_cost, latency_ms), not per-instance scores.
For LLM-guided, feedback-driven evolution with a per-instance Pareto front, see
``adalflow.optim.evolution``.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from random import Random
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from adalflow.core.base_data_class import DataClass


_EVAL_LOG_BUFFER: ContextVar[Optional[List[str]]] = ContextVar(
    "optimize_anything_eval_log_buffer", default=None
)


def log(message: str) -> None:
    """Log actionable side information during evaluator execution."""
    buffer = _EVAL_LOG_BUFFER.get()
    if buffer is None:
        return
    buffer.append(str(message))


@dataclass
class EngineConfig(DataClass):
    """Execution budget and runtime controls."""

    max_metric_calls: int = field(default=100)
    max_parallel: int = field(default=1)
    random_seed: int = field(default=0)


@dataclass
class GEPAConfig(DataClass):
    """Optimization controls for evolutionary + Pareto search."""

    engine: EngineConfig = field(default_factory=EngineConfig)
    population_size: int = field(default=8)
    elite_size: int = field(default=3)
    mutation_rate: float = field(default=0.7)
    crossover_rate: float = field(default=0.2)
    stop_score: Optional[float] = field(default=None)


@dataclass
class CandidateEvaluation(DataClass):
    candidate: str = field(default="")
    score: float = field(default=0.0)
    token_cost: int = field(default=0)
    latency_ms: float = field(default=0.0)
    side_info: List[str] = field(default_factory=list)


@dataclass
class OptimizeAnythingResult(DataClass):
    best_candidate: str = field(default="")
    best_score: float = field(default=0.0)
    metric_calls: int = field(default=0)
    history: List[CandidateEvaluation] = field(default_factory=list)
    pareto_frontier: List[CandidateEvaluation] = field(default_factory=list)
    objective: str = field(default="")


def _extract_eval_output(result: Union[float, int, Dict[str, Any]]) -> Dict[str, float]:
    if isinstance(result, (float, int)):
        return {"score": float(result)}
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError("evaluator dict output must include 'score'")
        output = {"score": float(result["score"])}
        if "token_cost" in result and result["token_cost"] is not None:
            output["token_cost"] = float(result["token_cost"])
        if "latency_ms" in result and result["latency_ms"] is not None:
            output["latency_ms"] = float(result["latency_ms"])
        return output
    raise TypeError("evaluator must return float/int or dict containing 'score'")


def _estimate_token_cost(candidate: str) -> int:
    stripped = candidate.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _dominates(a: CandidateEvaluation, b: CandidateEvaluation) -> bool:
    not_worse = (
        a.score >= b.score
        and a.token_cost <= b.token_cost
        and a.latency_ms <= b.latency_ms
    )
    strictly_better = (
        a.score > b.score
        or a.token_cost < b.token_cost
        or a.latency_ms < b.latency_ms
    )
    return not_worse and strictly_better


def _compute_pareto_frontier(
    records: Sequence[CandidateEvaluation],
) -> List[CandidateEvaluation]:
    frontier: List[CandidateEvaluation] = []
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: (-item.score, item.token_cost, item.latency_ms))
    return frontier


def _mutate(candidate: str, objective: str, side_info: Sequence[str], rng: Random) -> str:
    lines = candidate.splitlines()
    hints = [item for item in side_info if item]
    hint = hints[-1][:180] if hints else f"Objective: {objective[:120]}"

    operation = rng.choice(["append_hint", "prepend_hint", "line_swap", "dedupe_spaces"])

    if operation == "append_hint":
        return f"{candidate}\n# Hint: {hint}".strip()
    if operation == "prepend_hint":
        return f"# Objective: {objective}\n{candidate}".strip()
    if operation == "line_swap" and len(lines) >= 2:
        idx_a = rng.randrange(len(lines))
        idx_b = rng.randrange(len(lines))
        lines[idx_a], lines[idx_b] = lines[idx_b], lines[idx_a]
        return "\n".join(lines)
    return " ".join(candidate.split())


def _crossover(left: str, right: str, rng: Random) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    if not left_lines or not right_lines:
        return left if left else right
    left_cut = rng.randrange(1, len(left_lines) + 1)
    right_cut = rng.randrange(0, len(right_lines))
    merged = left_lines[:left_cut] + right_lines[right_cut:]
    return "\n".join(merged).strip()


def optimize_anything(
    seed_candidate: str,
    evaluator: Callable[[str], Union[float, int, Dict[str, Any]]],
    objective: str,
    config: GEPAConfig,
) -> OptimizeAnythingResult:
    """Optimize any text artifact with heuristic (non-LLM) mutations and Pareto search."""
    if not isinstance(seed_candidate, str):
        raise TypeError("seed_candidate must be a string")
    if config.engine.max_metric_calls <= 0:
        raise ValueError("config.engine.max_metric_calls must be > 0")
    if config.population_size <= 0:
        raise ValueError("config.population_size must be > 0")
    if config.elite_size <= 0:
        raise ValueError("config.elite_size must be > 0")

    rng = Random(config.engine.random_seed)
    seen_candidates = set()
    history: List[CandidateEvaluation] = []
    metric_calls = 0

    def evaluate_candidate(candidate: str) -> CandidateEvaluation:
        nonlocal metric_calls
        log_buffer: List[str] = []
        token = _EVAL_LOG_BUFFER.set(log_buffer)
        started_at = perf_counter()
        try:
            eval_output = _extract_eval_output(evaluator(candidate))
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            _EVAL_LOG_BUFFER.reset(token)

        metric_calls += 1
        token_cost = int(eval_output.get("token_cost", _estimate_token_cost(candidate)))
        latency_ms = float(eval_output.get("latency_ms", elapsed_ms))
        return CandidateEvaluation(
            candidate=candidate,
            score=float(eval_output["score"]),
            token_cost=token_cost,
            latency_ms=latency_ms,
            side_info=log_buffer,
        )

    seed_eval = evaluate_candidate(seed_candidate)
    history.append(seed_eval)
    seen_candidates.add(seed_candidate)
    best_eval = seed_eval

    population: List[CandidateEvaluation] = [seed_eval]

    while metric_calls < config.engine.max_metric_calls:
        frontier = _compute_pareto_frontier(population)
        elites = frontier[: min(len(frontier), config.elite_size)]
        if not elites:
            elites = sorted(
                population, key=lambda item: (-item.score, item.token_cost, item.latency_ms)
            )[: config.elite_size]

        next_generation: List[CandidateEvaluation] = list(elites)
        attempts = 0
        max_attempts = max(10, config.population_size * 8)
        generation_start_calls = metric_calls

        while (
            len(next_generation) < config.population_size
            and metric_calls < config.engine.max_metric_calls
            and attempts < max_attempts
        ):
            attempts += 1
            parent = rng.choice(elites)
            child_text = parent.candidate

            if rng.random() < config.crossover_rate and len(population) > 1:
                other = rng.choice(population)
                child_text = _crossover(parent.candidate, other.candidate, rng)

            if rng.random() < config.mutation_rate:
                child_text = _mutate(child_text, objective, parent.side_info, rng)

            if not child_text:
                continue
            if child_text in seen_candidates:
                child_text = f"{child_text}\n# variant:{metric_calls}:{attempts}"

            child_eval = evaluate_candidate(child_text)
            seen_candidates.add(child_text)
            history.append(child_eval)
            next_generation.append(child_eval)

            if child_eval.score > best_eval.score:
                best_eval = child_eval

            if config.stop_score is not None and child_eval.score >= config.stop_score:
                frontier_now = _compute_pareto_frontier(next_generation + population)
                return OptimizeAnythingResult(
                    best_candidate=best_eval.candidate,
                    best_score=best_eval.score,
                    metric_calls=metric_calls,
                    history=history,
                    pareto_frontier=frontier_now,
                    objective=objective,
                )

        if metric_calls == generation_start_calls:
            break

        population = sorted(
            next_generation, key=lambda item: (-item.score, item.token_cost, item.latency_ms)
        )[: config.population_size]

    final_frontier = _compute_pareto_frontier(history)
    return OptimizeAnythingResult(
        best_candidate=best_eval.candidate,
        best_score=best_eval.score,
        metric_calls=metric_calls,
        history=history,
        pareto_frontier=final_frontier,
        objective=objective,
    )