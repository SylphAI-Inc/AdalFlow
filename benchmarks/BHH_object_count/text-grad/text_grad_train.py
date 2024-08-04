import logging


log = logging.getLogger(__name__)

import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random


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
            tqdm_loader.set_description(f"Batch Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


# use ours propose if accept, set, if not , revert
def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model, max_samples=None))
    previous_performance = np.mean(results["validation_acc"][-1])
    # print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


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

    train_loader = tg.tasks.DataLoader(train_set, batch_size=4, shuffle=True)

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
    results["test_acc"].append(np.mean(eval_dataset(test_set, eval_fn, model)))  # 0.79
    results["validation_acc"].append(
        np.mean(eval_dataset(val_set, eval_fn, model))
    )  # 0.72
    results["prompt"].append(system_prompt.get_value())
    from lightrag.utils import save_json

    max_steps = 5

    # train the model
    for epoch in range(1):
        for steps, (batch_x, batch_y) in enumerate(
            (pbar := tqdm(train_loader, position=0))
        ):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for x, y in zip(batch_x, batch_y):
                x = tg.Variable(
                    x,
                    requires_grad=False,
                    role_description="query to the language model",
                )
                y = tg.Variable(
                    y,
                    requires_grad=False,
                    role_description="correct answer for the query",
                )
                response = model(x)

                try:
                    eval_output_variable = eval_fn(
                        inputs=dict(prediction=response, ground_truth_answer=y)
                    )
                    # print("eval_output_variable: ", eval_output_variable)
                except Exception as e:
                    log.info(f"Error: {e}")
                    eval_output_variable = eval_fn([x, y, response])
                print(f" y_gt: {y.value}")

                losses.append(eval_output_variable)
            total_loss = tg.sum(losses)  # operator aggregrate the feedbacks,
            total_loss.backward()  # it is still like separete other than the gradients now have a list from the batch.
            # loss_to_dict = total_loss.to_dict()

            # print("loss_to_dict: ", loss_to_dict)
            optimizer.step()
            # save_json(loss_to_dict, "loss_to_dict.json")

            run_validation_revert(system_prompt, results, model, eval_fn, val_set)

            # print("sys prompt: ", system_prompt)
            test_acc = eval_dataset(test_set, eval_fn, model)
            test_acc_mean = np.mean(test_acc)
            results["test_acc"].append(test_acc_mean)
            results["prompt"].append(system_prompt.get_value())
            save_json(results, "results_text_grad.json")

            if steps >= max_steps:
                break
