"""Retriever Recall @k metric."""

from typing import List, Union

from adalflow.eval.base import BaseEvaluator, EvaluationResult


class RetrieverRecall(BaseEvaluator):
    __doc__ = r"""Recall@k measures the ratio of the number of relevant context strings in the top-k retrieved context to the total number of ground truth relevant context strings.

    In our implementation, we use exact string matching between each gt context and the joined retrieved context string.
    You can use the longest common subsequence (LCS) or other similarity metrics(or embedding based) to decide if it is a match or not.

    If you do not even have the ground truth context, but only grounth truth answers, you can consider using
    RAGAS framework for now. It computes the recall as:

    Recall = [GT statements that can be attributed to the retrieved context] / [GT statements]

    Examples:
        >>> all_retrieved_context = [
        ["Apple is founded before Google.",
        "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year.",
        ]
        >>> all_gt_context = [
            [
                "Apple is founded in 1976.",
                "Google is founded in 1998.",
                "Apple is founded before Google.",
            ],
            ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"],
        ]
        >>> retriever_recall = RetrieverRecall()
        >>> avg_recall, recall_list = retriever_recall.compute(all_retrieved_context, all_gt_context)
        >>> avg_recall
        2 / 3
        >>> recall_list
        [1 / 3, 1.0]

    References:
        - RAGAS: https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html
    """

    def __init__(self):
        super().__init__()

    def _compute_single_item(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        r"""
        Compute the recall of the retrieved context for a single query.

        Args:
            retrieved_context (str): Retrieved context string.
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against.

        Returns:
            float: Recall value.
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        recalled = 0
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                recalled += 1
        return recalled / len(gt_context)

    def compute(
        self,
        retrieved_contexts: Union[List[str], List[List[str]]],
        gt_contexts: List[List[str]],
    ) -> EvaluationResult:
        r"""
        Compute the recall of the retrieved context for a list of queries.
        Args:
            retrieved_contexts (Union[List[str], List[List[str]]): List of retrieved context strings. Using List[str] we assume you have joined all the context sentences into one string.
            gt_contexts ( List[List[str]]): List of ground truth context strings.

        Returns:
            tuple:
                - float: Average recall value.
                - List[float]: Recall values for each query.
        """
        if len(retrieved_contexts) != len(gt_contexts):
            raise ValueError(
                "The number of retrieved context lists and ground truth context lists should be the same."
            )
        k = len(retrieved_contexts)
        recall_list = []
        for retrieved_context, gt_context in zip(retrieved_contexts, gt_contexts):
            if isinstance(retrieved_context, list):
                retrieved_context = " ".join(retrieved_context)
            recall = self._compute_single_item(retrieved_context, gt_context)
            recall_list.append(recall)

        avg_score = sum(recall_list) / len(recall_list)
        return EvaluationResult(
            avg_score, recall_list, additional_info={"type": f"RetrieverRecall@{k}"}
        )
