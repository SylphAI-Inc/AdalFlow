"""This is the metric to evaluate the recall of the retriever."""

from typing import List, Union, Tuple


class RetrieverRecall:
    r"""
    Metric to evaluate the recall of the retriever. The recall is the ratio of the number of relevant context strings in the retrieved context to the total number of ground truth relevant context strings.

    Examples:
        >>> all_retrieved_context = [
        "Apple is founded before Google.",
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
    """

    def __init__(self):
        pass

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
        retrieved_contexts: List[str],
        gt_contexts: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        r"""
        Compute the recall of the retrieved context for a list of queries.
        Args:
            retrieved_contexts (List[str]): List of retrieved context strings.
            gt_contexts (Union[List[str], List[List[str]]]: List of ground truth context strings and each of them can be a string or a list of strings.

        Returns:
            tuple:
                - float: Average recall value.
                - List[float]: Recall values for each query.
        """
        recall_list = []
        for retrieved_context, gt_context in zip(retrieved_contexts, gt_contexts):
            recall = self._compute_single_item(retrieved_context, gt_context)
            recall_list.append(recall)

        return sum(recall_list) / len(recall_list), recall_list
