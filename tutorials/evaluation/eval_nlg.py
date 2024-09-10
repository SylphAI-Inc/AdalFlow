gt = "Brazil has won 5 FIFA World Cup titles"
pred = "Brazil is the five-time champion of the FIFA WorldCup."


def compute_rouge(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/rouge_score.html
    """
    from torchmetrics.text.rouge import ROUGEScore

    rouge = ROUGEScore()
    return rouge(pred, gt)


def compute_bleu(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/bleu_score.html
    """
    from torchmetrics.text.bleu import BLEUScore

    bleu = BLEUScore()
    # preds = ["the cat is on the mat"]
    # target = [["there is a cat on the mat", "a cat is on the mat"]]
    # score = bleu(preds, target)
    # print(f"score: {score}")
    # print(f"pred: {[pred]}, gt: {[[gt]]}")
    return bleu([pred], [[gt]])


def compute_bertscore(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/bert_score.html
    """
    from torchmetrics.text.bert import BERTScore

    bertscore = BERTScore()
    return bertscore([pred], [gt])


def compute_llm_as_judge():
    import adalflow as adal
    from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge
    from adalflow.components.model_client import OpenAIClient

    adal.setup_env()

    questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    gt_answers = ["Yes", "Yes", "No"]

    llm_judge = DefaultLLMJudge(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 10,
        },
    )
    llm_evaluator = LLMasJudge(llm_judge=llm_judge)
    print(llm_judge)
    eval_rslt = llm_evaluator.compute(
        questions=questions, gt_answers=gt_answers, pred_answers=pred_answers
    )
    print(eval_rslt)


def compute_llm_as_judge_wo_questions():
    import adalflow as adal
    from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge
    from adalflow.components.model_client import OpenAIClient

    adal.setup_env()

    llm_judge = DefaultLLMJudge(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 10,
        },
        jugement_query="Does the predicted answer means the same as the ground truth answer? Say True if yes, False if no.",
    )
    llm_evaluator = LLMasJudge(llm_judge=llm_judge)
    print(llm_judge)
    eval_rslt = llm_evaluator.compute(gt_answers=[gt], pred_answers=[pred])
    print(eval_rslt)


def compute_g_eval_summarization(source, summary):
    from adalflow.eval.g_eval import GEvalLLMJudge, GEvalJudgeEvaluator, NLGTask

    model_kwargs = {
        "model": "gpt-4o",
        "n": 20,
        "top_p": 1,
        "max_tokens": 5,
        "temperature": 1,
    }

    g_eval = GEvalLLMJudge(
        default_task=NLGTask.SUMMARIZATION, model_kwargs=model_kwargs
    )
    print(g_eval)
    input_template = """Source Document: {source}
    Summary: {summary}
    """

    input_str = input_template.format(
        source=source,
        summary=summary,
    )

    g_evaluator = GEvalJudgeEvaluator(llm_judge=g_eval)

    response = g_evaluator(input_strs=[input_str])
    print(f"response: {response}")


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab")
    from adalflow.utils import setup_env

    setup_env()
    # print(f"ROUGE score: {compute_rouge(gt, pred)}")
    # # fmeasure: 0.22, precision: 0.25
    # print(f"BLEU score: {compute_bleu(gt, pred)}")
    # # score: 0.0

    # print(f"BERT score: {compute_bertscore(gt, pred)}")
    # # score 0.9752, recall: 0.9827, and precision: 0.9789

    # compute_llm_as_judge()

    # compute_llm_as_judge_wo_questions()
    source = (
        "Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with Burnley on Sunday . 'Just been watching the game , did you miss the coach ? # RubberDub # 7minutes , ' Merson put on Twitter . Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in ( the England team ) then it opens it up to anybody . ' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England 's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake . 'It 's not as though I was watching hoping he would n't score for England , I 'm genuinely pleased for him and fair play to him \u00e2\u20ac\u201c it was a great goal , ' Merson said . 'It 's just a matter of opinion , and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson , so he should n't have been in the squad . 'When I 'm wrong , I hold my hands up . I do n't have a problem with doing that - I 'll always be the first to admit when I 'm wrong . ' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit Merson ( centre ) criticised Townsend 's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday , saying 'Not bad for a player that should be 'nowhere near the squad ' ay @ PaulMerse ? ' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor .",
    )
    summary = (
        "Paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . Andros townsend scored the tottenham midfielder in the 89th minute . Paul merson had another dig at andros townsend after his appearance . The midfielder had been brought on to the england squad last week . Click here for all the latest arsenal news news .",
    )

    compute_g_eval_summarization(source=source, summary=summary)
    compute_g_eval_summarization(source=gt, summary=pred)
