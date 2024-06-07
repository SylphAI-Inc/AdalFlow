"""This is the evaluator for answer matching. It compares the predicted answer with the ground truth answer."""

from typing import List, Tuple


class AnswerMatchEvaluator:
    r"""
    Evaluator for evaluating the match between predicted and ground truth answer.
    """

    def __init__(self, type: str = "exact_match"):
        r"""
        Initialize a new instance of AnswerMacthEvaluator.

        Args:
            type (str): Type of matching evaluation. Can be "exact_match" or "fuzzy_match". "exact_match" requires the predicted answer to be exactly the same as the ground truth answer. "fuzzy_match" requires the predicted answer to contain the ground truth answer.
        """
        self.type = type

    def compute_match_acc_single_query(self, pred_answer: str, gt_answer: str) -> float:
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

    def compute_match_acc(
        self, all_pred_answer: List[str], all_gt_answer: List[str]
    ) -> Tuple[float, List[float]]:
        r"""
        Compute the match accuracy of the predicted answer for a list of queries.

        Args:
            all_pred_answer (List[str]): List of predicted answer strings.
            all_gt_answer (List[str]): List of ground truth answer strings.

        Returns:
            tuple:
                - float: Average match accuracy.
                - List[float]: Match accuracy values for each query.
        """
        match_acc_list = []
        for pred_answer, gt_answer in zip(all_pred_answer, all_gt_answer):
            match = self.compute_match_acc_single_query(pred_answer, gt_answer)
            match_acc_list.append(match)

        return sum(match_acc_list) / len(match_acc_list), match_acc_list
