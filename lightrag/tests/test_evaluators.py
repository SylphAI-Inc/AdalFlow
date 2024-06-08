import os
import pytest

from lightrag.eval.answer_match_evaluator import AnswerMatchEvaluator
from lightrag.eval.retriever_evaluator import RetrieverEvaluator


def test_answer_match_evaluator():
    all_pred_answer = ["positive", "negative", "this is neutral"]
    all_gt_answer = ["positive", "negative", "neutral"]
    generator_evaluator = AnswerMatchEvaluator(type="exact_match")
    answer_match_acc, match_acc_list = generator_evaluator.compute_match_acc(
        all_pred_answer, all_gt_answer
    )
    assert answer_match_acc == 2 / 3
    assert match_acc_list == [1.0, 1.0, 0.0]
    generator_evaluator = AnswerMatchEvaluator(type="fuzzy_match")
    answer_match_acc, match_acc_list = generator_evaluator.compute_match_acc(
        all_pred_answer, all_gt_answer
    )
    assert answer_match_acc == 1.0
    assert match_acc_list == [1.0, 1.0, 1.0]


def test_retriever_evaluator():
    all_retrieved_context = [
        "Apple is founded before Google.",
        "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year.",
    ]
    all_gt_context = [
        [
            "Apple is founded in 1976.",
            "Google is founded in 1998.",
            "Apple is founded before Google.",
        ],
        ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"],
    ]
    retriever_evaluator = RetrieverEvaluator()
    avg_recall, recall_list = retriever_evaluator.compute_recall(
        all_retrieved_context, all_gt_context
    )
    assert avg_recall == 2 / 3
    assert recall_list == [1 / 3, 1.0]
    avg_relevance, relevance_list = retriever_evaluator.compute_context_relevance(
        all_retrieved_context, all_gt_context
    )
    assert 0.8 < avg_relevance < 0.81
    assert relevance_list[0] == 1.0
    assert 0.6 < relevance_list[1] < 0.61


# This test is skipped by default. To run this test, set the environment variable RUN_LOCAL_TESTS to True (export RUN_LOCAL_TESTS=true).
@pytest.mark.skipif(not os.getenv("RUN_LOCAL_TESTS"), reason="Skip unless on local")
def test_llm_as_judge():
    from lightrag.eval.llm_as_judge_evaluator import LLMasJudge

    all_questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    all_pred_answer = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    all_gt_answer = ["Yes", "Yes", "No"]
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    llm_judge = LLMasJudge()
    avg_judgement, judgement_list = llm_judge.compute_judgement(
        all_questions, all_pred_answer, all_gt_answer, judgement_query
    )
    assert avg_judgement == 2 / 3
    assert judgement_list == [True, True, False]
