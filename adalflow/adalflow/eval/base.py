"""Abstract base class for evaluation metrics."""

from typing import Optional, List, Any

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Evaluation result."""

    avg_score: float
    per_item_scores: Optional[List[float]] = None
    additional_info: Optional[dict] = None


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def compute_single_item(self, *args, **kwargs) -> float:
        """Compute the score for a single item."""
        raise NotImplementedError("Subclasses must implement this method.")

    # TODO: support multi-threading or async to speed up evaluation
    def compute(self, *args, **kwargs) -> Any:
        """Evaluate a list of predictions and ground truth values. and return overall score and per-item scores."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
