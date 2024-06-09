"""This is the metric for answer matching. It compares the predicted answer with the ground truth answer."""

from typing import List, Tuple


class AnswerMatchAcc:
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

    def __init__(self, type: str = "exact_match"):
        self.type = type

    def _compute_single_item(self, pred_answer: str, gt_answer: str) -> float:
        r"""
        Compute the match accuracy of the predicted answer for a single query.

        Args:
            pred_answer (str): Predicted answer string.
            gt_answer (str): Ground truth answer string.

        Returns:
            float: Match accuracy.
        """
        if self.type == "exact_match":
            return 1.0 if pred_answer == gt_answer else 0.0
        elif self.type == "fuzzy_match":
            return 1.0 if gt_answer in pred_answer else 0.0
        else:
            raise NotImplementedError

    def compute(
        self, pred_answers: List[str], gt_answers: List[str]
    ) -> Tuple[float, List[float]]:
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
            match = self._compute_single_item(pred_answer, gt_answer)
            match_acc_list.append(match)

        return sum(match_acc_list) / len(match_acc_list), match_acc_list
