"""LLM-guided evolution of text artifacts (programs, prompts, configs).

An LLM proposer reads a candidate's evaluation feedback and rewrites it; parents are
picked from a per-instance Pareto front (after GEPA, Agrawal et al., 2025); programs
are scored in a separate process (``PythonEvaluator``), in the spirit of OpenEvolve.
Independent implementation.
"""

from adalflow.optim.evolution.archive import (
    BEST,
    MAP_ELITES,
    PARETO,
    RESAMPLE,
    Archive,
)
from adalflow.optim.evolution.loop import EvolutionResult, evolve
from adalflow.optim.evolution.proposer import (
    PROPOSE_TEMPLATE,
    LLMProposer,
    Proposal,
    ProposalError,
    Proposer,
    extract_candidate,
    render_feedback,
)
from adalflow.optim.evolution.presets import (
    cases_evaluator,
    exact_match,
    optimize_code,
    optimize_prompt,
    prompt_evaluator,
)
from adalflow.optim.evolution.sandbox import PythonEvaluator, SandboxResult, run_python
from adalflow.optim.evolution.types import (
    Candidate,
    Evaluation,
    combine,
    to_evaluation,
)

__all__ = [
    "optimize_code",
    "optimize_prompt",
    "cases_evaluator",
    "prompt_evaluator",
    "exact_match",
    "Archive",
    "BEST",
    "PARETO",
    "RESAMPLE",
    "MAP_ELITES",
    "EvolutionResult",
    "evolve",
    "PROPOSE_TEMPLATE",
    "LLMProposer",
    "Proposal",
    "ProposalError",
    "Proposer",
    "extract_candidate",
    "render_feedback",
    "PythonEvaluator",
    "SandboxResult",
    "run_python",
    "Candidate",
    "Evaluation",
    "combine",
    "to_evaluation",
]
