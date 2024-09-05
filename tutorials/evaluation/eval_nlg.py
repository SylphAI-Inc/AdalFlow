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
    from adalflow.eval.llm_as_judge import LLMasJudge

    adal.setup_env()

    questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    gt_answers = ["Yes", "Yes", "No"]
    # judgement_query = (
    #     "For the question, does the predicted answer contain the ground truth answer?"
    # )
    llm_judge = LLMasJudge()
    print(llm_judge)
    avg_judgement, judgement_list = llm_judge.compute(
        questions, gt_answers, pred_answers
    )
    print(avg_judgement)
    print(judgement_list)


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab")
    print(f"ROUGE score: {compute_rouge(gt, pred)}")
    # fmeasure: 0.22, precision: 0.25
    print(f"BLEU score: {compute_bleu(gt, pred)}")
    # score: 0.0

    print(f"BERT score: {compute_bertscore(gt, pred)}")
    # score 0.9752, recall: 0.9827, and precision: 0.9789

    compute_llm_as_judge()
