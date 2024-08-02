import os
import pytest

from lightrag.eval import (
    AnswerMatchAcc,
    RetrieverRecall,
    RetrieverRelevance,
    LLMasJudge,
)


def test_answer_match_acc():
    pred_answers = ["positive", "negative", "this is neutral"]
    gt_answers = ["positive", "negative", "neutral"]
    answer_match_acc = AnswerMatchAcc(type="exact_match")
    acc = answer_match_acc.compute(pred_answers, gt_answers)
    avg_acc, acc_list = acc.avg_score, acc.per_item_scores
    assert avg_acc == 2 / 3
    assert acc_list == [1.0, 1.0, 0.0]
    answer_match_acc = AnswerMatchAcc(type="fuzzy_match")
    acc = answer_match_acc.compute(pred_answers, gt_answers)
    avg_acc, acc_list = acc.avg_score, acc.per_item_scores
    assert avg_acc == 1.0
    assert acc_list == [1.0, 1.0, 1.0]


def test_retriever_recall():
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
    retriever_recall = RetrieverRecall()
    avg_recall, recall_list = retriever_recall.compute(retrieved_contexts, gt_contexts)
    assert avg_recall == 2 / 3
    assert recall_list == [1 / 3, 1.0]


def test_retriever_relevance():
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
    retriever_relevance = RetrieverRelevance()
    avg_relevance, relevance_list = retriever_relevance.compute(
        retrieved_contexts, gt_contexts
    )
    assert 0.8 < avg_relevance < 0.81
    assert relevance_list[0] == 1.0
    assert 0.6 < relevance_list[1] < 0.61


# This test is skipped by default. To run this test locally, set the environment variable RUN_LOCAL_TESTS to True (export RUN_LOCAL_TESTS=true).
@pytest.mark.skipif(not os.getenv("RUN_LOCAL_TESTS"), reason="Skip unless on local")
def test_llm_as_judge():

    questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    gt_answers = ["Yes", "Yes", "No"]
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    llm_judge = LLMasJudge()
    avg_judgement, judgement_list = llm_judge.compute(
        questions, gt_answers, pred_answers, judgement_query
    )
    assert avg_judgement == 2 / 3
    assert judgement_list == [True, True, False]
