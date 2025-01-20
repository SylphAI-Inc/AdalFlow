import dspy
import dspy.evaluate
from dspy import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc


# DSPY cache:~/cachedir_joblib/joblib/dsp/modules
turbo = dspy.OpenAI(model="gpt-3.5-turbo-0125")
gpt_4 = dspy.OpenAI(model="gpt-4o")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

# dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
dspy.configure(lm=turbo)


class GenerateAnswer(dspy.Signature):
    """You are a classifier. Given a question, you need to classify it into one of the following classes:
    Format: class_index. class_name, class_description
    0. ABBR, Abbreviation
    1. ENTY, Entity
    2. DESC, Description and abstract concept
    3. HUM, Human being
    4. LOC, Location
    5. NUM, Numeric value
    - Do not try to answer the question:"""

    question: str = dspy.InputField(desc="Question to be classified")
    answer: str = dspy.OutputField(
        desc="Select one from ABBR, ENTY, DESC, HUM, LOC, NUM"
    )


class TrecClassifier(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):

        pred = self.generate_answer(question=question)
        return dspy.Prediction(answer=pred.answer)


def exact_match(example, pred, trace=None):
    # if str(pred.answer.strip()) == str(example.answer.strip()):
    #     return True

    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    if eval_fn(pred.answer, example.answer):
        return True

    return False


def load_dspy_datasets():
    trainset, valset, testset = load_datasets()
    dspy_trainset, dspy_valset, dspy_testset = [], [], []
    for dataset in zip(
        [trainset, valset, testset], [dspy_trainset, dspy_valset, dspy_testset]
    ):
        for item in dataset[0]:
            example = Example(question=item.question, answer=str(item.class_name))
            example = example.with_inputs("question")
            dataset[1].append(example)

    return dspy_trainset, dspy_valset, dspy_testset


def train_signature(trainset, valset, save_path, filename):
    from dspy.teleprompt import COPRO
    import os

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    teleprompter = COPRO(
        metric=dspy.evaluate.answer_exact_match,
        verbose=True,
    )
    kwargs = dict(
        num_threads=64, display_progress=True, display_table=0
    )  # Used in Evaluate class in the optimization process

    compiled_baleen = teleprompter.compile(
        TrecClassifier(), trainset=trainset, eval_kwargs=kwargs
    )
    turbo.inspect_history(n=3)
    compiled_baleen.save(os.path.join(save_path, filename))


def train(trainset, valset, save_path, filename):
    r"""
    Use the MIPROv2 teleprompter to train the model.
    """
    import os
    from dspy.teleprompt import MIPROv2

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # I dont know how to config teacher_config, cant find their documentation on this.
    # teleprompter = BootstrapFewShotWithRandomSearch(
    #     metric=dspy.evaluate.answer_exact_match,
    #     teacher_settings=dict(lm=gpt_4),
    #     max_rounds=1,
    #     max_bootstrapped_demos=4,
    #     max_labeled_demos=40,
    # )
    # compiled_baleen = teleprompter.compile(
    #     TrecClassifier(),
    #     # teacher=TrecClassifier(),
    #     trainset=trainset,
    #     valset=valset,
    # )
    # turbo.inspect_history(n=3)
    # compiled_baleen.save(os.path.join(save_path, filename))
    # return compiled_baleen

    tp = MIPROv2(
        metric=exact_match,
        prompt_model=gpt_4,
        task_model=turbo,
        num_candidates=30,
        init_temperature=1.0,
    )
    compiled_task = tp.compile(
        TrecClassifier(),
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=5,
        max_labeled_demos=2,
        num_batches=12,  # MINIBATCH_SIZE = 25,
        seed=2025,
        requires_permission_to_run=False,
    )
    compiled_task.save(os.path.join(save_path, filename))
    return compiled_task


def evaluate(devset, compiled_task):
    from dspy.evaluate.evaluate import Evaluate

    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    eval = Evaluate(
        devset=devset, num_threads=4, display_progress=True, display_table=5
    )

    # Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
    metric = dspy.evaluate.answer_exact_match
    output = eval(compiled_task, metric=metric)
    return output


if __name__ == "__main__":
    from adalflow.utils import setup_env
    from use_cases.classification.data import load_datasets

    setup_env()

    task = TrecClassifier()

    trainset, valset, testset = load_dspy_datasets()

    start_val_acc = evaluate(valset, task)
    print("Val start: ", start_val_acc)  # 72.89%
    start_test_acc = evaluate(testset, task)
    print("Test start: ", start_test_acc)  # 76.35%

    dspy_save_path = "benchmarks/trec_classification/dspy_models"
    import os

    # preevaluate the model before training

    os.makedirs(dspy_save_path, exist_ok=True)
    # even the same prompt, dspy underperforms
    # output = evaluate(testset, task)  # val start: 61.11, train: 57.5%, # test: 60.42%
    # print(output)

    import time

    start = time.time()

    # train the model
    compiled_baleen = train(
        trainset, valset, dspy_save_path, "trec_classifier_MIPROv2_50_name_2.json"
    )

    # # select class: optimizeed: test: 83.3%, val: 83.3%
    test_score = evaluate(testset, compiled_baleen)
    val_score = evaluate(valset, compiled_baleen)
    print("Time taken: ", time.time() - start)
    print("Test score: ", test_score)
    print("Val score: ", val_score)

    # MIROv2: 85.6 on test, 81.9 on val, 86.2, 83.1, total trial 12
    # 87.4, 80.1 val
    test_scores = [85.6, 86.2, 87.4]
    val_scores = [81.9, 83.1, 80.1]

    # compute mean and std
    import numpy as np

    mean_test_score = np.mean(test_scores)
    std_test_score = np.std(test_scores)
    mean_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    print(
        mean_test_score, std_test_score, mean_val_score, std_val_score
    )  # 86.4%, 0.74, 81.7%, 1.23
