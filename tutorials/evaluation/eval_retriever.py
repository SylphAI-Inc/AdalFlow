import json

from adalflow.eval import RetrieverEvaluator

retrieved_contexts = [
    ["Apple is founded before Google."],
    ["Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year."],
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
    retriever_recall = RetrieverEvaluator()
    eval_result = retriever_recall.compute(retrieved_contexts, gt_contexts)
    return eval_result


if __name__ == "__main__":
    eval_result = evaluate_retriever(retrieved_contexts, gt_contexts)
    print(f"eval_result: {json.dumps(eval_result, indent=4)}")
