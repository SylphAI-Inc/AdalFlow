import logging


log = logging.getLogger(__name__)

import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random

# get_logger(level="DEBUG", filename="lib_text_grad.log")

load_dotenv()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(
        x, requires_grad=False, role_description="query to the language model"
    )
    y = tg.Variable(
        y, requires_grad=False, role_description="correct answer for the query"
    )
    response = model(x)
    try:
        eval_output_variable = eval_fn(
            inputs=dict(prediction=response, ground_truth_answer=y)
        )
        return int(eval_output_variable.value)
    except Exception as e:
        log.info(f"Error: {e}")
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int = None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), position=0
        )
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


def test_text_grad():
    from textgrad.engine import get_engine
    from textgrad import Variable, TextualGradientDescent
    from textgrad.loss import TextLoss
    from dotenv import load_dotenv
    from lightrag.utils import get_logger

    get_logger(level="DEBUG", filename="lib_text_grad.log")

    load_dotenv()

    x = Variable(
        "A sntence with a typo",
        role_description="The input sentence",
        requires_grad=True,
    )  # weights
    print(x.gradients)
    engine = get_engine("gpt-3.5-turbo")
    output = engine.generate("Hello how are you?")

    print(engine)
    print(output)

    # Call it Eval Feedback, no gradient, a judge? takes y and y_hat (no y_hat) so no normal loss, but text feedback on the response.
    system_prompt = Variable(
        "Evaluate the correctness of this sentence",
        role_description="The system prompt",
    )  # this is llm
    # EvalFeedback
    loss = TextLoss(
        system_prompt, engine=engine
    )  # generate messages [{'role': 'system', 'content': 'Evaluate the correctness of this sentence'}, {'role': 'user', 'content': 'A sntence with a typo'}]
    print(loss)
    optimizer = TextualGradientDescent(
        parameters=[x], engine=engine
    )  # TODO: pass system prompt instead of x?
    print(optimizer)

    # putting together
    # loss takes x, isnt thi
    l = loss(x)  # noqa: E741
    print(f"loss: {l}")
    # computes the gradients for the variable x
    """
    v: The sentence you provided does indeed contain a typo.
    The word "sntence" should be corrected to "sentence."
    v.gradients: set()
    v: A sntence with a typo (x)
    v.gradients: {Variable(value=Since the language model correctly identified a typo in the sentence provided, the feedback for the variable "<VARIABLE> A sntence with a typo </VARIABLE>" would be to ensure that the text is free of any spelling errors before presenting it. One way to improve the variable is to run a spell check or proofread the text to catch any typos or spelling mistakes before using it in a context where accuracy is crucial. By ensuring that the text is error-free, the overall quality and credibility of the content will be enhanced, leading to better performance according to the <OBJECTIVE_FUNCTION>., role=feedback to The input sentence, grads=)}
    v: Evaluate the correctness of this sentence (prompt variable)
    v.gradients: {Variable(value=The system prompt could be improved by providing a more specific and detailed instruction to the language model. Instead of a general directive like "Evaluate the correctness of this sentence," you could consider providing more context or guidance to the model. For example, you could ask the model to specifically identify and correct any spelling errors, grammatical mistakes, or punctuation issues in the given sentence. This way, the model would have a clearer understanding of the task at hand and could provide more targeted feedback. Additionally, you could include examples of common errors that the model should look out for, which would help guide its evaluation process and improve the quality of the feedback provided., role=feedback to The system prompt, grads=)}
    """
    l.backward(engine)
    log.info(f"l: {l}")
    # print(f"loss: {l}")
    # optimizer.step()
    # print(x)
    # print(x.gradients)

    """
    {feedback_str}
    loss: loss: The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence."

    gradient: (feedback to The input sentence)
    {Variable(value=Since the language model correctly identified a typo in the sentence provided, the feedback for the variable "<VARIABLE> A sntence with a typo </VARIABLE>" would be to ensure that the text is free of any spelling errors before presenting it. One way to improve the variable is to run a spell check or proofread the text to catch any typos or spelling mistakes before using it in a context where accuracy is crucial. By ensuring that the text is error-free, the overall quality and credibility of the content will be enhanced, leading to better performance according to the <OBJECTIVE_FUNCTION>., role=feedback to The input sentence, grads=)}

    """


# ln -s /Users/liyin/Library/Caches/textgrad/ textgrad


if __name__ == "__main__":

    set_seed(12)
    llm_api_eval = tg.get_engine(engine_name="gpt-4o")
    llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo")
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load the data and the evaluation function
    train_set, val_set, test_set, eval_fn = load_task(
        "BBH_object_counting", evaluation_api=llm_api_eval
    )
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(STARTING_SYSTEM_PROMPT)

    train_loader = tg.tasks.DataLoader(
        train_set, batch_size=3, shuffle=True
    )  # why not torch loader?

    # Testing the 0-shot performance of the evaluation engine
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt to the language model",
    )
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task",
    )
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    optimizer = tg.TextualGradientDescent(
        engine=llm_api_eval, parameters=[system_prompt]
    )

    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model))  # 0.79
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))  # 0.72
    results["prompt"].append(system_prompt.get_value())
    print(results)
