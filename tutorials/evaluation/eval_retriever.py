from adalflow.eval import RetrieverRecall, RetrieverRelevance

retrieved_contexts = [
    "Apple is founded before Google.",
    "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year.",
]
gt_contexts = [
    [
        "Apple is founded in 1976.",
        "Google is founded in 1998.",
        "Apple is founded before Google.",
    ],
    ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"],
]


def evaluate_retriever(retrieved_contexts, gt_contexts):
    retriever_recall = RetrieverRecall()
    avg_recall, recall_list = retriever_recall.compute(
        retrieved_contexts, gt_contexts
    )  # Compute the recall of the retriever
    retriever_relevance = RetrieverRelevance()
    avg_relevance, relevance_list = retriever_relevance.compute(
        retrieved_contexts, gt_contexts
    )  # Compute the relevance of the retriever
    return avg_recall, recall_list, avg_relevance, relevance_list


if __name__ == "__main__":
    avg_recall, recall_list, avg_relevance, relevance_list = evaluate_retriever(
        retrieved_contexts, gt_contexts
    )
    print(f"avg_recall: {avg_recall}, recall_list: {recall_list}")
    print(f"avg_relevance: {avg_relevance}, relevance_list: {relevance_list}")
