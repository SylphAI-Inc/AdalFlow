"""Abstract base class for evaluation metrics."""

from typing import Optional


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def compute(
        self, x: object, y_pred: object, y_gt: Optional[object] = None
    ) -> object:
        """Evaluate one x, y, y_pred pair. y_gt is optional if you use llm or other models to predict the metrics."""
        raise NotImplementedError("Subclasses must implement this method.")
