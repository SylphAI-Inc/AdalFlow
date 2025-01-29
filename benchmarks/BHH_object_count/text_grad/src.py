import random
import concurrent.futures
from tqdm import tqdm
import numpy as np

import textgrad as tg

log = tg.logging.getLogger(__name__)


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
    y = str(y)
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
            print("sample: ", sample)

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
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)
