from typing import List, Union, Tuple
from core.tokenizer import Tokenizer


class RetrieverEvaluator:
    def __init__(self):
        pass

    def compute_recall_single_query(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        """
        Compute the recall of the retrieved context for a single query.
        Args:
            retrieved_context (str): Retrieved context string
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against
        Returns:
            float: Recall value
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        recalled = 0
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                recalled += 1
        return recalled / len(gt_context)

    def compute_recall(
        self,
        all_retrieved_context: List[str],
        all_gt_context: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        """
        Compute the recall of the retrieved context for a list of queries.
        Args:
            all_retrieved_context: List of retrieved context strings
            all_gt_context: List of ground truth context strings and each of them can be a string or a list of strings
        Returns:
            float: Average recall value
            List[float]: Recall values for each query
        """
        recall_list = []
        for retrieved_context, gt_context in zip(all_retrieved_context, all_gt_context):
            recall = self.compute_recall_single_query(retrieved_context, gt_context)
            recall_list.append(recall)

        return sum(recall_list) / len(recall_list), recall_list

    def compute_context_relevance_single_query(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        """
        Compute the context relevance of the retrieved context for a single query. The context relevance is the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
        Args:
            retrieved_context (str): Retrieved context string
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against
        Returns:
            float: Context relevance value
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        relevant_tokens = 0
        tokenizer = Tokenizer()
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                relevant_tokens += tokenizer.count_tokens(gt_context_sentence)
        return relevant_tokens / tokenizer.count_tokens(retrieved_context)

    def compute_context_relevance(
        self,
        all_retrieved_context: List[str],
        all_gt_context: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        """
        Compute the context relevance of the retrieved context for a list of queries.
        Args:
            all_retrieved_context: List of retrieved context strings
            all_gt_context: List of ground truth context strings and each of them can be a string or a list of strings
        Returns:
            float: Average context relevance value
            List[float]: Context relevance values for each query
        """
        context_relevance_list = []
        for retrieved_context, gt_context in zip(all_retrieved_context, all_gt_context):
            context_relevance = self.compute_context_relevance_single_query(
                retrieved_context, gt_context
            )
            context_relevance_list.append(context_relevance)

        return (
            sum(context_relevance_list) / len(context_relevance_list),
            context_relevance_list,
        )
