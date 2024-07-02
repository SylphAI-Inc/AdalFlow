"""This is the metric for evaluating the relevance of the retrieved context."""

from typing import List, Union, Tuple
from lightrag.core.tokenizer import Tokenizer


class RetrieverRelevance:
    r"""
    Metric for evaluating the relevance of the retrieved context. The context relevance is the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.

    Examples:
        >>> retrieved_contexts = [
        "Apple is founded before Google.",
        "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year.",
        ]
        >>> gt_contexts = [
            [
                "Apple is founded in 1976.",
                "Google is founded in 1998.",
                "Apple is founded before Google.",
            ],
            ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"],
        ]
        >>> retriever_relevance = RetrieverRelevance()
        >>> avg_relevance, relevance_list = retriever_relevance.compute(all_retrieved_context, all_gt_context)
        >>> avg_relevance
        0.803030303030303
        >>> relevance_list
        [1.0, 0.6060606060606061]

    """

    def __init__(self):
        pass

    def _compute_single_item(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        r"""
        Compute the context relevance of the retrieved context for a single query. The context relevance is the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.

        Args:
            retrieved_context (str): Retrieved context string.
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against.

        Returns:
            float: Context relevance value.
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        relevant_tokens = 0
        tokenizer = Tokenizer()
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                relevant_tokens += tokenizer.count_tokens(gt_context_sentence)
        return relevant_tokens / tokenizer.count_tokens(retrieved_context)

    def compute(
        self,
        retrieved_contexts: List[str],
        gt_contexts: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        r"""
        Compute the context relevance of the retrieved context for a list of queries.

        Args:
            retrieved_contexts (List[str]): List of retrieved context strings.
            gt_contexts (Union[List[str], List[List[str]]]): List of ground truth context strings and each of them can be a string or a list of strings.

        Returns:
            tuple:
                - float: Average context relevance value.
                - List[float]: Context relevance values for each query.
        """
        context_relevance_list = []
        for retrieved_context, gt_context in zip(retrieved_contexts, gt_contexts):
            context_relevance = self._compute_single_item(retrieved_context, gt_context)
            context_relevance_list.append(context_relevance)

        return (
            sum(context_relevance_list) / len(context_relevance_list),
            context_relevance_list,
        )
