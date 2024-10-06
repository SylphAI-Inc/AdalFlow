"""This is the metric for QA generation. It compares the predicted answer with the ground truth answer."""

from typing import List, Literal
from adalflow.eval.base import BaseEvaluator, EvaluationResult
from adalflow.optim.parameter import Parameter


class AnswerMatchAcc(BaseEvaluator):
    r"""
    Metric for answer matching. It compares the predicted answer with the ground truth answer.

    Args:
        type (str): Type of matching evaluation. Can be "exact_match" or "fuzzy_match". "exact_match" requires the predicted answer to be exactly the same as the ground truth answer. "fuzzy_match" requires the predicted answer to contain the ground truth answer.

    Examples:
        >>> pred_answers = ["positive", "negative", "this is neutral"]
        >>> gt_answers = ["positive", "negative", "neutral"]
        >>> answer_match_acc = AnswerMatchAcc(type="exact_match")
        >>> avg_acc, acc_list = answer_match_acc.compute(all_pred_answer, all_gt_answer)
        >>> avg_acc
        2 / 3
        >>> acc_list
        [1.0, 1.0, 0.0]
        >>> answer_match_acc = AnswerMatchAcc(type="fuzzy_match")
        >>> avg_acc, acc_list = answer_match_acc.compute(all_pred_answer, all_gt_answer)
        >>> avg_acc
        1.0
        >>> acc_list
        [1.0, 1.0, 1.0]
    """

    def __init__(self, type: Literal["exact_match", "fuzzy_match"] = "exact_match"):
        self.type = type

    def compute_single_item(
        self,
        y: object,
        y_gt: object,
    ) -> float:
        r"""
        Compute the match accuracy of the predicted answer for a single query.

        Allow any type of input for pred_answer and gt_answer.
        When evaluating, the input will be converted to string.

        Args:
            pred_answer (object): Predicted answer.
            gt_answer (object): Ground truth answer.

        Returns:
            float: Match accuracy.
        """
        if isinstance(y, Parameter):
            y = y.data
        if isinstance(y_gt, Parameter):
            y_gt = y_gt.data
        try:
            y = str(y).strip()
            y_gt = str(y_gt).strip()
        except Exception as e:
            raise ValueError(
                f"Error converting pred_answer and gt_answer to string: {e}"
            )
        if self.type == "exact_match":
            return 1.0 if y == y_gt else 0.0
        elif self.type == "fuzzy_match":
            y = y.lower()
            y_gt = y_gt.lower()
            return 1.0 if y_gt in y else 0.0
        else:
            raise NotImplementedError

    def compute(
        self, pred_answers: List[str], gt_answers: List[str]
    ) -> EvaluationResult:
        r"""
        Compute the match accuracy of the predicted answer for a list of queries.

        Args:
            pred_answers (List[str]): List of predicted answer strings.
            gt_answers (List[str]): List of ground truth answer strings.

        Returns:
            tuple:
                - float: Average match accuracy.
                - List[float]: Match accuracy values for each query.
        """
        match_acc_list = []
        for pred_answer, gt_answer in zip(pred_answers, gt_answers):
            match = self.compute_single_item(pred_answer, gt_answer)
            match_acc_list.append(match)

        return EvaluationResult(
            avg_score=sum(match_acc_list) / len(match_acc_list),
            per_item_scores=match_acc_list,
        )
