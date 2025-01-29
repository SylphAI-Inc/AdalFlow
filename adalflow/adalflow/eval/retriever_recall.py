"""Retriever Recall @k metric."""

from typing import List, Dict

from adalflow.eval.base import BaseEvaluator, EvaluationResult
from adalflow.eval.utils import normalize_answer


class RetrieverEvaluator(BaseEvaluator):
    __doc__ = r"""Return Recall@k and Precision@k.

    Recall@k = Number of relevant retrieved documents/ Total number of relevant documents (len(gt_contexts))
    Precision@k = Number of relevant retrieved documents/ Total number of retrieved documents (len(retrieved_contexts))


    In our implementation, we use exact string matching between each gt context and the joined retrieved context string.
    You can use the longest common subsequence (LCS) or other similarity metrics(or embedding based) to decide if it is a match or not.

    You can also pass ids of retrieved and the reference.

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

    def compute_single_item(
        self, retrieved_context: List[str], gt_context: List[str]
    ) -> Dict[str, float]:
        r"""
        Compute the recall of the retrieved context for a single query.

        Args:
            retrieved_context (List[str]): List of retrieved context strings.
            gt_context (List[str]): List of ground truth context strings.

        Returns:
            float: Recall value.
        """
        # 1 normalize the text
        normalized_retrieved_context = [
            normalize_answer(doc) for doc in retrieved_context
        ]

        normalized_gt_context = [normalize_answer(doc) for doc in gt_context]

        set_retrieved = set(normalized_retrieved_context)
        set_gt = set(normalized_gt_context)

        # 2 calculate the recall with intersection

        recall = len(set_gt.intersection(set_retrieved)) / len(set_gt)
        precision = len(set_gt.intersection(set_retrieved)) / len(set_retrieved)

        return {"recall": recall, "precision": precision}

    def compute(
        self,
        retrieved_contexts: List[List[str]],
        gt_contexts: List[List[str]],
    ) -> EvaluationResult:
        r"""
        Compute the recall of the retrieved context for a list of queries.
        Args:
            retrieved_context: List of retrieved context strings.
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
        k = len(retrieved_contexts[0])
        metric_list = []
        for retrieved_context, gt_context in zip(retrieved_contexts, gt_contexts):

            metric = self.compute_single_item(retrieved_context, gt_context)
            metric_list.append(metric)

        # average through each key value

        avg_recall = sum([metric["recall"] for metric in metric_list]) / len(
            metric_list
        )
        avg_precision = sum([metric["precision"] for metric in metric_list]) / len(
            metric_list
        )

        return {
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "recall_list": [metric["recall"] for metric in metric_list],
            "precision_list": [metric["precision"] for metric in metric_list],
            "top_k": k,
        }


if __name__ == "__main__":
    from adalflow.datasets import HotPotQA, HotPotQAData

    train_dataset = HotPotQA(split="train", size=10)
    data: HotPotQAData = train_dataset[0]
    gold_titles = data.gold_titles
    context_titles = data.context["title"]
    print(f"gold_titles: {gold_titles}, context_titles: {context_titles}")
    print(f"train: {len(train_dataset)}, example: {train_dataset[0]}")

    # compute the recall and precision for 10 items
    retriever_eval = RetrieverEvaluator()

    gt_contexts = [list(data.gold_titles) for data in train_dataset[:10]]

    retrieved_contexts = [list(data.context["title"]) for data in train_dataset[:10]]

    result = retriever_eval.compute(retrieved_contexts, gt_contexts)

    print(f"result: {result}")
