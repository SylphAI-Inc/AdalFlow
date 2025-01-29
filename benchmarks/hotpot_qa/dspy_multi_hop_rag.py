"""
Dspy's CoT is handled by itself with ChainOfThought. That is why our prompt (signature) does not need to include "think step by step".
"""

import dspy
import dspy.evaluate

from dspy import Example


turbo = dspy.OpenAI(model="gpt-3.5-turbo-0125")
gpt_4 = dspy.OpenAI(model="gpt-4o")

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
from adalflow.eval.answer_match_acc import AnswerMatchAcc


def load_datasets():
    from benchmarks.hotpot_qa.config import load_datasets

    trainset, valset, testset = load_datasets()
    # dspy requires us to package the dataset to Example objects and specify the inputs

    dspy_trainset, dspy_valset, dspy_testset = [], [], []
    for dataset in zip(
        [trainset, valset, testset], [dspy_trainset, dspy_valset, dspy_testset]
    ):
        for item in dataset[0]:
            example = Example(question=item.question, answer=item.answer)
            example = example.with_inputs("question")
            dataset[1].append(example)

    return dspy_trainset, dspy_valset, dspy_testset


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer to the question")


task_desc_str = """
You will receive an original question, last search query, and the retrieved context from the last search query.
Write the next search query to help retrieve all relevant context to answer the original question.
Think step by step."""


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


# class GenerateSearchQuery(dspy.Signature):
#     """You will receive an original question, last search query, and the retrieved context from the last search query.
#     Write the next search query to help retrieve all relevant context to answer the original question.
#     """

#     context = dspy.InputField(desc="may contain relevant facts")
#     question = dspy.InputField()
#     last_search_query = dspy.InputField(desc="The last search query")
#     query = dspy.OutputField()


from dsp.utils import deduplicate


class DsPyMultiHopRAG(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        # last_query = None

        for hop in range(self.max_hops):
            query = self.generate_query[hop](
                context=context,
                question=question,  # last_search_query=last_query
            ).query
            # last_query = query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)


# pred: Prediction


def validate_answer(example, pred, trace=None):
    evaluator = AnswerMatchAcc(type="exact_match")
    print(f"pred: {pred.answer}, example: {example['answer']}")
    return evaluator.compute_single_item(str(pred.answer), example["answer"])


def train_MIPROv2(trainset, valset, save_path, filename):

    import os
    from dspy.teleprompt import MIPROv2

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tp = MIPROv2(
        metric=validate_answer,
        prompt_model=gpt_4,
        task_model=turbo,
        num_candidates=12,
        init_temperature=1.0,
        log_dir=save_path,
    )
    compiled_task = tp.compile(
        DsPyMultiHopRAG(),
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=0,  # 2,
        max_labeled_demos=0,  # 2,
        num_batches=12,  # MINIBATCH_SIZE = 25, (eval on trainset)
        seed=2025,
        requires_permission_to_run=False,
    )
    compiled_task.save(os.path.join(save_path, filename))
    return compiled_task


# def train(trainset, save_path, filename):
#     from dspy.teleprompt import BootstrapFewShot
#     import os

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     # teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
#     teleprompter = BootstrapFewShot(metric=validate_answer)
#     compiled_baleen = teleprompter.compile(
#         SimplifiedBaleen(),
#         teacher=SimplifiedBaleen(passages_per_hop=2),
#         trainset=trainset,
#     )
#     turbo.inspect_history(n=3)
#     compiled_baleen.save(os.path.join(save_path, filename))
#     return compiled_baleen


def validate(devset, compiled_baleen=None, uncompiled_baleen=None):
    from dspy.evaluate.evaluate import Evaluate

    # Define metric to check if we retrieved the correct documents
    # def gold_passages_retrieved(example, pred, trace=None):
    #     gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    #     found_titles = set(
    #         map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    #     )
    #     return gold_titles.issubset(found_titles)

    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    evaluate_on_hotpotqa = Evaluate(
        devset=devset,
        num_threads=4,
        display_progress=True,
        display_table=5,
        # metric=validate_answer,
    )
    if uncompiled_baleen is not None:
        uncompiled_baleen_answer_score = evaluate_on_hotpotqa(
            uncompiled_baleen, metric=validate_answer, display_progress=True
        )
        print(
            f"## Answer Score for uncompiled Baleen: {uncompiled_baleen_answer_score}"
        )

    if compiled_baleen is None:
        return

    compiled_baleen_answer_score = evaluate_on_hotpotqa(
        compiled_baleen, metric=validate_answer, display_progress=True
    )
    print(f"## Answer Score for compiled Baleen: {compiled_baleen_answer_score}")
    return compiled_baleen_answer_score


def train_and_validate():
    import os

    save_path = "benchmarks/hotpot_qa/dspy_multi_hop_rag_zero_shot"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import time
    import tqdm

    dspy_trainset, dspy_valset, dspy_testset = load_datasets()

    val_accs = []
    test_accs = []
    training_times = []
    max_val_score = 0
    max_test_score = 0

    num_runs = 4

    for i in tqdm.tqdm(range(num_runs)):
        start = time.time()
        output_file = f"compiled_count_{i}.json"

        compiled_count = train_MIPROv2(
            dspy_trainset, dspy_valset, save_path, output_file
        )
        val_acc = validate(dspy_valset, compiled_count)  # 46
        test_acc = validate(dspy_testset, compiled_count)  # 52

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        training_times.append(time.time() - start)

    # compute the mean and standard deviation
    import numpy as np

    val_accs = np.array(val_accs)
    test_accs = np.array(test_accs)
    training_times = np.array(training_times)
    max_val_score = val_accs.max()
    max_test_score = test_accs.max()
    print("Validation accuracy:", val_accs.mean(), val_accs.std())
    print("Test accuracy:", test_accs.mean(), test_accs.std())

    print("Training time:", training_times.mean())
    print("Max val score: ", max_val_score)
    print("Max test score: ", max_test_score)


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    # Ask any question you like to this simple RAG program.
    my_question = "How many storeys are in the castle that David Gregory inherited?"

    task = DsPyMultiHopRAG()
    trainset, valset, testset = load_datasets()

    # pred = uncompiled_baleen(my_question)

    # # Print the contexts and the answer.
    # print(f"Question: {my_question}")
    # print(f"Predicted Answer: {pred.answer}")
    # print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
    # turbo.inspect_history(n=3)

    # Load the datasets.
    trainset, valset, testset = load_datasets()

    # 32.0% EM val, 36.5
    validate(valset, None, task)
    validate(testset, None, task)

    train_and_validate()

    # train the model
    # compiled_baleen = train(trainset, dspy_save_path, "hotpotqa.json")
    # validate(devset, compiled_baleen, uncompiled_baleen)

    # with demos (2, 2)
    # Validation accuracy: 47.25 3.031088913245535
    # Test accuracy: 50.625 3.0898017735770686
    # Training time: 2465.3250265717506

    # zero shot
    # Validation accuracy: 35.5 4.330127018922194
    # Test accuracy: 37.875 5.140221298738022
    # Training time: 182.31551551818848
    # Max val score:  42.0
    # Max test score:  46.5
