"""LLM-guided evolution of text artifacts (programs, prompts, configs).

An LLM proposer reads a candidate's evaluation feedback and rewrites it; parents are
picked from a per-instance Pareto front (after GEPA, Agrawal et al., 2025); programs
are scored in a separate process (``PythonEvaluator``), in the spirit of OpenEvolve.
Independent implementation.
"""

from adalflow.optim.evolution.archive import BEST, PARETO, Archive
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
from adalflow.optim.evolution.sandbox import PythonEvaluator, SandboxResult, run_python
from adalflow.optim.evolution.types import Candidate, Evaluation, to_evaluation

__all__ = [
    "Archive",
    "BEST",
    "PARETO",
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
    "to_evaluation",
]
