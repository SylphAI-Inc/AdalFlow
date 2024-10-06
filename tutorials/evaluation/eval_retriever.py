from adalflow.eval import RetrieverRecall

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
    avg_recall, recall_list = retriever_recall.compute(retrieved_contexts, gt_contexts)
    return avg_recall, recall_list


if __name__ == "__main__":
    avg_recall, recall_list = evaluate_retriever(retrieved_contexts, gt_contexts)
    print(f"avg_recall: {avg_recall}, recall_list: {recall_list}")
