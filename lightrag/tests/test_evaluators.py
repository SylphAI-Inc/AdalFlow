from lightrag.eval.answer_match_evaluator import AnswerMatchEvaluator
from lightrag.eval.llm_as_judge_evaluator import LLMasJudge


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


def test_llm_as_judge():
    all_questions = ["question1", "question2", "question3"]
    all_pred_answer = ["positive", "negative", "this is neutral"]
    all_gt_answer = ["positive", "positive", "neutral"]
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    llm_judge = LLMasJudge()
    avg_judgement, judgement_list = llm_judge.compute_judgement(
        all_questions, all_pred_answer, all_gt_answer, judgement_query
    )
    assert avg_judgement == 2 / 3
    assert judgement_list == [1.0, 0.0, 1.0]
