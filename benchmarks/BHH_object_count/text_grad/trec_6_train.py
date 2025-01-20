"""
We use the dataloader and metric from adalflow to ensure the exact data is used and metric is used.
We have one difference on the prompt: text-grad use a message ["system", "user"] while adalflow uses a single prompt. and put it as the system field.
The training is using text-grad to test the performance
"""

from use_cases.classification.data import load_datasets, TRECExtendedData
from use_cases.classification.trec_task_structured_output import (
    task_desc_template,
)
import time
import os
import adalflow as adal
from adalflow.utils.data import DataLoader
import numpy as np
import concurrent.futures
from tqdm import tqdm

from adalflow.datasets.trec import _COARSE_LABELS_DESC, _COARSE_LABELS
from adalflow.eval.answer_match_acc import AnswerMatchAcc


import textgrad as tg
from benchmarks.BHH_object_count.text_grad.config import gpt4o, gpt_3_5

tg.set_backward_engine(gpt4o, override=True)

import logging

log = logging.getLogger(__name__)

template = r"""{{system_prompt}}
{% if output_format_str is not none %}
{{output_format_str}}
{% endif %}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
"""


def get_task_desc():
    label_desc = [
        {"label": label, "desc": desc}
        for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
    ]

    task_desc_str = adal.Prompt(
        template=task_desc_template, prompt_kwargs={"classes": label_desc}
    )()
    parser = adal.DataClassParser(
        data_class=TRECExtendedData, return_data_class=True, format_type="yaml"
    )

    return adal.Prompt(
        template=template,
        prompt_kwargs={
            "system_prompt": task_desc_str,
            "output_format_str": parser.get_output_format_str(),
        },
    )()


def response_to_classname(response: str):
    parser = adal.DataClassParser(
        data_class=TRECExtendedData, return_data_class=True, format_type="yaml"
    )
    parsed_data = parser.call(response)
    if isinstance(parsed_data, TRECExtendedData):
        return parsed_data.class_name
    return "error"


def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """

    x, y, response = run_one_sample(model, item)
    try:
        # print(f"y: {y.value}, response: {response}")
        eval_output_variable = eval_fn(y=response, y_gt=y.value)
        score = eval_output_variable
        # print("score: ", score)
        return score
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
            # print("sample: ", sample)

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


def run_one_sample(model, data: TRECExtendedData):
    x, y = data.question, data.class_name
    x = tg.Variable(
        x, requires_grad=False, role_description="query to the language model"
    )
    y = str(y)
    y = tg.Variable(
        y, requires_grad=False, role_description="correct answer for the query"
    )

    response = model(x)
    response_value = response_to_classname(response.value)
    response.value = response_value
    # print("response: ", response)
    return x, y, response


from textgrad.autograd.string_based_ops import StringBasedFunction

# NOTE: need to customize two lines in StringBasedFunction
# class StringBasedFunction(Function):
#     def __init__(self, fn: Callable, function_purpose: str):
#         """
#         Autograd function for string-based functions.

#         :param fn: The function to execute for the forward pass.
#         :type fn: Callable
#         :param function_purpose: The description of the purpose of the function. Analogous to role description for variables.
#         :type function_purpose: str
#         """
#         super().__init__()
#         self.fn = fn
#         self.function_purpose = function_purpose

#     def forward(self,
#                 inputs: Dict[str, Variable],
#                 response_role_description: str = None) -> Variable:
#         """
#         The forward mode for string-based functions

#         :param inputs: The arguments that will be passed to the string based function. The keys are the names of the arguments.
#         :type fn: Dict[str, Variable]
#         :param response_role_description: The role description of the output variable.
#         :type response_role_description: str
#         """
#         if response_role_description is None:
#             response_role_description = f"Output of the string-based function with purpose: {self.function_purpose}"
#         converted_inputs = {k: v.get_value() for k, v in inputs.items()}
#         response_string = str(self.fn(**converted_inputs))

#         # Create the response variable
#         response = Variable(
#             value=response_string,
#             predecessors=list(inputs.values()),
#             role_description=response_role_description
#         )

#         logger.info(f"StringBasedFunction", extra={"text": f"In: {inputs}, Out: {response_string}"})

#         # Populate the gradient function, using a container to store the backward function and the context
#         response.set_grad_fn(BackwardContext(backward_fn=self.backward,
#                                              response=response,
#                                              function_purpose=self.function_purpose,
#                                              inputs=inputs))


#         return response
def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model, max_samples=None))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


def train(
    max_steps=12,
    optimizer_model=None,
    task_llm=None,
    runs=0,
    train_set=None,
    test_set=None,
    val_set=None,
    text_grad_save_path=None,
    eval_fn=None,
    system_prompt=None,
):

    first_start_time = time.time()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

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

    current_step = 0
    fn_purpose = (
        "The runtime of string-based function that checks if the prediction is correct."
    )
    eval_loss_fn = StringBasedFunction(eval_fn, function_purpose=fn_purpose)

    # train the model
    num_epochs = int(max_steps / ((len(train_set) // 4))) + 1
    for epoch in range(num_epochs):
        for steps, (batch) in enumerate((pbar := tqdm(train_loader, position=0))):
            current_step += 1
            pbar.set_description(f"Training step {current_step}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []

            for data in batch:

                x, y, response = run_one_sample(task_llm, data)

                try:

                    eval_score = eval_loss_fn(inputs={"y": response, "y_gt": y})
                except Exception as e:
                    log.info(f"Error: {e}")
                    eval_score = f"Error: {e} at evaluating y: {y.value} and response: {response}"

                losses.append(eval_score)
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


def multi_run_train(num_runs=4):
    training_times = []
    test_scores = []
    val_scores = []
    for runs in range(0, num_runs):
        print(f"Run: {runs}")
        system_prompt = tg.Variable(
            task_desc,
            requires_grad=True,
            role_description="system prompt to the language model",
        )
        task_llm = tg.BlackboxLLM(gpt_3_5, system_prompt)
        training_time, results = train(
            max_steps=12,
            optimizer_model=gpt4o,
            task_llm=task_llm,
            runs=runs,
            train_set=train_dataset,
            test_set=test_dataset,
            val_set=val_dataset,
            text_grad_save_path=os.path.join(text_grad_save_path, "trec_6"),
            eval_fn=eval_fn,
            system_prompt=system_prompt,
        )
        print("Training Time: ", training_time)
        print("Results: ", results)
        training_times.append(training_time)
        test_scores.append(results["test_acc"][-1])
        val_scores.append(results["validation_acc"][-1])
    # average pass rate, average pass prompts
    print(f"test_scores: {test_scores}")
    print(f"val_scores: {val_scores}")
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


if __name__ == "__main__":
    from benchmarks.config import text_grad_save_path

    train_dataset, val_dataset, test_dataset = load_datasets()
    task_desc = get_task_desc()
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    # train_loader = tg.tasks.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # test_one_sample(task_llm, test_dataset[0])
    # output = eval_dataset(val_dataset, eval_fn, task_llm)  # 80.12%
    # print(output)
    # test_score = eval_dataset(test_dataset, eval_fn, task_llm)  # 83.5%
    # print(test_score)
    # training_time, results = train(
    #     max_steps=12,
    #     optimizer_model=gpt4o,
    #     task_llm=task_llm,
    #     train_set=train_dataset,
    #     test_set=test_dataset,
    #     val_set=val_dataset,
    #     text_grad_save_path=os.path.join(text_grad_save_path, "trec_6"),
    #     eval_fn=eval_fn,
    # )
    multi_run_train(num_runs=4)
