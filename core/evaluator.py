from typing import List, Union


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
    ) -> float:
        """
        Compute the recall of the retrieved context for a list of queries.
        Args:
            all_retrieved_context: List of retrieved context strings
            all_gt_context: List of ground truth context strings
        """
        recall_list = []
        for retrieved_context, gt_context in zip(all_retrieved_context, all_gt_context):
            recall = self.compute_recall_single_query(retrieved_context, gt_context)
            recall_list.append(recall)

        return sum(recall_list) / len(recall_list)
