"""
Text grad's object count implementation:

self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

We use the same task description, the only difference is dspy send over a messages: [system_prompt, user_message] and we do ["system": <> system_prompt <> <> user_message<>]
"""

import logging


log = logging.getLogger(__name__)

from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
from benchmarks.BHH_object_count.text_grad.src import (
    eval_dataset,
    set_seed,
    run_validation_revert,
)

load_dotenv()


def train(max_steps=12, optimizer_model=None, task_model=None, runs=0):

    first_start_time = time.time()

    # Load the data and the evaluation function
    train_set, val_set, test_set, eval_fn = load_task(
        "BBH_object_counting", evaluation_api=task_model
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
    task_llm = tg.BlackboxLLM(task_model, system_prompt)

    optimizer = tg.TextualGradientDescent(
        engine=optimizer_model, parameters=[system_prompt], gradient_memory=3
    )

    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    results["test_acc"].append(
        np.mean(eval_dataset(test_set, eval_fn, task_llm))
    )  # 0.78
    results["validation_acc"].append(
        np.mean(eval_dataset(val_set, eval_fn, task_llm))
    )  # 0.74
    results["prompt"].append(system_prompt.get_value())
    from adalflow.utils import save_json

    file_name = f"results_text_grad_{runs}.json"

    file_path = os.path.join(text_grad_save_path, file_name)

    max_steps = 12
    current_step = 0

    # train the model
    num_epochs = int(max_steps / ((len(train_set) // 4))) + 1
    for epoch in range(num_epochs):
        for steps, (batch_x, batch_y) in enumerate(
            (pbar := tqdm(train_loader, position=0))
        ):
            current_step += 1
            pbar.set_description(f"Training step {current_step}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for x, y in zip(batch_x, batch_y):
                x = tg.Variable(
                    x,
                    requires_grad=False,
                    role_description="query to the language model",
                )
                y = str(y)
                y = tg.Variable(
                    y,
                    requires_grad=False,
                    role_description="correct answer for the query",
                )
                response = task_llm(x)

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
            start_time = time.time()
            total_loss = tg.sum(losses)  # operator aggregrate the feedbacks,
            total_loss.backward()  # it is still like separete other than the gradients now have a list from the batch.
            # print(f"total_loss: {total_loss}")
            end_time = time.time()
            print(
                "Time taken for backward: ", end_time - start_time
            )  # 62s for bacjward
            # loss_to_dict = total_loss.to_dict()

            # print("loss_to_dict: ", loss_to_dict)
            start_time = time.time()
            optimizer.step()
            end_time = time.time()
            print("Time taken for step: ", end_time - start_time)
            # save_json(loss_to_dict, "loss_to_dict.json")

            run_validation_revert(system_prompt, results, task_llm, eval_fn, val_set)

            # print("sys prompt: ", system_prompt)
            results["prompt"].append(system_prompt.get_value())
            save_json(results, file_path)

            if current_step >= max_steps:
                break

    test_acc = eval_dataset(test_set, eval_fn, task_llm)
    print(f"Test Accuracy: {test_acc}")
    test_acc_mean = np.mean(test_acc)
    results["test_acc"].append(test_acc_mean)

    save_json(results, file_path)
    end_time = time.time()
    print("Time taken: ", end_time - first_start_time)  # 6 steps.
    training_time = end_time - first_start_time
    return training_time, results


if __name__ == "__main__":

    from benchmarks.config import text_grad_save_path
    import os
    import time

    set_seed(12)
    gpt4o = tg.get_engine(engine_name="gpt-4o")
    gpt_3_5 = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
    tg.set_backward_engine(gpt4o, override=True)
    num_runs = 3
    training_times = []
    test_scores = []
    val_scores = []
    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}")
        training_time, results = train(
            max_steps=12, optimizer_model=gpt4o, task_model=gpt_3_5, runs=i + 1
        )
        print("Training Time: ", training_time)
        print("Results: ", results)
        training_times.append(training_time)
        test_scores.append(results["test_acc"][-1])
        val_scores.append(results["validation_acc"][-1])
    # average pass rate, average pass prompts
    avg_training_time = np.mean(training_times)
    avg_test_score = np.mean(test_scores)
    avg_val_score = np.mean(val_scores)
    print("Average Training Time: ", avg_training_time)
    print("Average Test Score: ", avg_test_score)
    print("Average Val Score: ", avg_val_score)
    # std
    std_test_score = np.std(test_scores)
    std_val_score = np.std(val_scores)
    print("Std Test Score: ", std_test_score)
    print("Std Val Score: ", std_val_score)
    # 3056.7141630649567 s, 1641s
