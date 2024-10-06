import dspy
from dspy.datasets import HotPotQA
import dspy.evaluate


turbo = dspy.OpenAI(model="gpt-3.5-turbo")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
from adalflow.eval.answer_match_acc import AnswerMatchAcc


def load_datasets():
    # Load the dataset.
    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(f"trainset, devset: {len(trainset)}, {len(devset)}, example: {trainset[0]}")

    len(trainset), len(devset)
    return trainset, devset


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


from dsp.utils import deduplicate


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)


# pred: Prediction


def validate_answer(example, pred, trace=None):
    evaluator = AnswerMatchAcc(type="fuzzy_match")
    return evaluator.compute_single_item(pred.answer, example["answer"])


def validate_context_and_answer_and_hops(example, pred, trace=None):
    # print(f"example: {example}, pred: {pred}, trace: {trace}")
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    # print("answer_exact_match")
    return True
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    # print("answer_passage_match")
    return True

    hops = [example.question] + [
        outputs.query for *_, outputs in trace if "query" in outputs
    ]

    if max([len(h) for h in hops]) > 100:
        return False
    if any(
        dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
        for idx in range(2, len(hops))
    ):
        return False

    return True


def train(trainset, save_path, filename):
    from dspy.teleprompt import BootstrapFewShot
    import os

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
    teleprompter = BootstrapFewShot(metric=validate_answer)
    compiled_baleen = teleprompter.compile(
        SimplifiedBaleen(),
        teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=trainset,
    )
    turbo.inspect_history(n=3)
    compiled_baleen.save(os.path.join(save_path, filename))
    return compiled_baleen


def validate(devset, compiled_baleen, uncompiled_baleen):
    from dspy.evaluate.evaluate import Evaluate
    import dspy

    # Define metric to check if we retrieved the correct documents
    def gold_passages_retrieved(example, pred, trace=None):
        gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
        found_titles = set(
            map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
        )
        return gold_titles.issubset(found_titles)

    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    evaluate_on_hotpotqa = Evaluate(
        devset=devset,
        num_threads=1,
        display_progress=True,
        display_table=5,
        # metric=validate_answer,
    )
    uncompiled_baleen_answer_score = evaluate_on_hotpotqa(
        uncompiled_baleen, metric=validate_answer, display_progress=True
    )
    print(f"## Answer Score for uncompiled Baleen: {uncompiled_baleen_answer_score}")

    if compiled_baleen is None:
        return

    compiled_baleen_answer_score = evaluate_on_hotpotqa(
        compiled_baleen, metric=validate_answer, display_progress=True
    )
    print(f"## Answer Score for compiled Baleen: {compiled_baleen_answer_score}")

    # uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(
    #     uncompiled_baleen, metric=gold_passages_retrieved, display=False
    # )

    # compiled_baleen_retrieval_score = evaluate_on_hotpotqa(
    #     compiled_baleen, metric=gold_passages_retrieved
    # )

    # print(
    #     f"## Retrieval Score for uncompiled Baleen: {uncompiled_baleen_retrieval_score}"
    # )
    # print(f"## Retrieval Score for compiled Baleen: {compiled_baleen_retrieval_score}")


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    # Ask any question you like to this simple RAG program.
    my_question = "How many storeys are in the castle that David Gregory inherited?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
    # pred = uncompiled_baleen(my_question)

    # # Print the contexts and the answer.
    # print(f"Question: {my_question}")
    # print(f"Predicted Answer: {pred.answer}")
    # print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
    # turbo.inspect_history(n=3)

    # Load the datasets.
    trainset, devset = load_datasets()
    from benchmarks.config import dspy_save_path

    validate(
        devset, uncompiled_baleen, uncompiled_baleen
    )  # dspy has 58.0% accuracy untrained. it is very slow at the inference, 3.58s per example

    # train the model
    compiled_baleen = train(trainset, dspy_save_path, "hotpotqa.json")
    validate(devset, compiled_baleen, uncompiled_baleen)

    # dspy 16 raw shots, 4 demos
    # dspy supports multiple generators,  in this case 3. Two query generator and one answer generator, they all choose the same examples.
    # accuracy 62.0
