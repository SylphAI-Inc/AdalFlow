import subprocess
import tempfile
import json
import numpy as np
import argparse

num_runs = 4
# List of experiments to run
object_count = "use_cases/question_answering/bbh/object_count/train_new.py"
trec_6_classification = "use_cases/classification/train.py"
hotpot_qa_multi_hop_rag = "benchmarks/hotpot_qa/adal_exp/train_multi_hop_rag.py"
hotpot_qa_multi_hop_rag_cycle = (
    "benchmarks/hotpot_qa/adal_exp/train_multi_hop_rag_cycle.py"
)
hotpot_qa_vanilla = "benchmarks/hotpot_qa/adal_exp/train_vanilla.py"
hotpot_qa_vanilla_rag = "benchmarks/hotpot_qa/adal_exp/train_vanilla_rag.py"
hotpot_qa_agent_rag = "benchmarks/hotpot_qa/adal_exp/train_agent_rag.py"


ckpt_values = []
experiments = [
    # object_count,
    trec_6_classification,
    # hotpot_qa_vanilla_rag,
    # hotpot_qa_multi_hop_rag,
    # hotpot_qa_multi_hop_rag_cycle,
    # hotpot_qa_agent_rag,
]

# set up the strategy for each experiment

argparser = argparse.ArgumentParser()
argparser.add_argument("--strategy", type=str, default="constrained")
argparser.add_argument("--use_tg", action="store_true")
argparser.add_argument("--max_proposals_per_step", type=int, default=5)
argparser.add_argument(
    "--disable_backward", action="store_true"
)  # no training data context
argparser.add_argument(
    "--disable_backward_gradients", action="store_true"
)  # no backward gradients

args = argparser.parse_args()

strategy = args.strategy
use_tg = args.use_tg
max_proposals_per_step = args.max_proposals_per_step
disable_backward = args.disable_backward
disable_backward_gradients = args.disable_backward_gradients

# Optional: Arguments for each experiment (if needed)

setup_str = f"--strategy {strategy}"

if use_tg:
    setup_str += " --use_tg"

setup_str += f" --max_proposals_per_step {max_proposals_per_step}"
if disable_backward:
    setup_str += " --disable_backward"

if disable_backward_gradients:
    setup_str += " --disable_backward_gradients"


experiment_args = {
    object_count: setup_str,
    trec_6_classification: setup_str,
    hotpot_qa_vanilla_rag: setup_str,
    hotpot_qa_multi_hop_rag: setup_str,
}
ckpt_values = {}


def run_experiment(script, args):
    try:
        # Use a temporary file to store the ckpt
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        print(f"Running {script} with args: {args}")
        subprocess.run(
            f"python {script} {temp_path} {args}",
            check=True,
            shell=True,
            text=True,
        )

        # Read the ckpt value from the temporary file
        with open(temp_path, "r") as f:
            data = json.load(f)
            ckpt = data.get("ckpt")
            print(f"Checkpoint from {script}: {ckpt}")
            return ckpt

    except subprocess.CalledProcessError as e:
        print(f"Experiment {script} failed with error: {e}")
        return None


if __name__ == "__main__":

    result_file = "text_grad_2_results"
    # add important run information in the naming of the file
    import uuid

    result_file = f"{result_file}_{num_runs}_runs_{uuid.uuid4()}.json"

    for experiment in experiments:
        args = experiment_args.get(experiment, "")
        for i in range(num_runs):
            print(f"\nRun {i + 1}/{num_runs}")
            ckpt = run_experiment(experiment, args)
            ckpt_index = f"{experiment}_{i + 1}"
            if ckpt:
                ckpt_values[ckpt_index] = ckpt
        # load all json files using the ckpt paths
        highest_test_score, last_test_score, mean_test_score, standard_deviation = (
            0,
            0,
            0,
            0,
        )
        last_test_scores = []
        highest_val_scores = []
        total_passes = (
            []
        )  # each is the number of unique val scores in the highest val scores
        total_prompts = []  # how many prompts tried in total

        past_highest_val_scores = []
        # # average pass rate, average pass prompts
        # average_pass_rate_list = []
        # average_pass_prompts_list = []
        # average_total_prompts = []
        # highest_test_score_json_file = None
        total_steps = []
        training_times = []
        subset_pass_rate = []
        valset_pass_rate = []
        for experiment_index, ckpt in ckpt_values.items():
            with open(ckpt, "r") as f:
                data = json.load(f)
                print(f"Experiment: {experiment_index}")
                print(f"Data: {data}")
                _high_val_score = max(data["val_scores"])
                _unique_val_scores = len(set(data["val_scores"])) - 1
                _last_test_score = data["test_score"]
                # read the effective measures
                effective_measures = data.get("effective_measure", {})

                _total_prompts = effective_measures.get("subset", {}).get(
                    "pass", 0
                ) + effective_measures.get("subset", {}).get("fail", 0)
                if _total_prompts == 0:
                    _total_prompts = effective_measures.get("valset", {}).get(
                        "pass", 0
                    ) + effective_measures.get("valset", {}).get("fail", 0)
                _total_steps = len(data["steps"]) - 1
                _training_time = data.get("total_time", 0)
                _subset_pass = effective_measures.get("subset", {}).get("pass", 0)
                _subset_fail = effective_measures.get("subset", {}).get("fail", 0)
                _valset_pass = effective_measures.get("valset", {}).get("pass", 0)
                _valset_fail = effective_measures.get("valset", {}).get("fail", 0)
                subset_pass_rate.append(_subset_pass / (_subset_pass + _subset_fail))
                valset_pass_rate.append(_valset_pass / (_valset_pass + _valset_fail))
                # save the results in the lists
                past_highest_val_scores.append(_high_val_score)
                total_passes.append(_unique_val_scores)
                total_prompts.append(_total_prompts)
                last_test_scores.append(_last_test_score)
                total_steps.append(_total_steps)
                training_times.append(_training_time)

        # ensure all steps are the same
        assert all(
            [step == total_steps[0] for step in total_steps]
        ), "All steps should be the same"

        # compute the metrics
        mean_test_score = np.mean(last_test_scores)
        std_test_score = np.std(last_test_scores)

        # val scores
        mean_val_score = np.mean(past_highest_val_scores)
        std_val_score = np.std(past_highest_val_scores)

        # pass rate total_passes / steps
        average_pass_rate = np.mean(total_passes) / total_steps[0]

        # average total prompts
        average_total_prompts = np.mean(total_prompts)

        # average training time
        average_training_time = np.mean(training_times)

        # subset pass rate
        average_subset_pass_rate = np.mean(subset_pass_rate)

        # valset pass rate
        average_valset_pass_rate = np.mean(valset_pass_rate)

        # add these numbers in the ckpt_values
        index = f"{experiment}_summary"
        ckpt_values[index] = {
            "config": {
                "num_runs": num_runs,
                "args": args,
            },
            "metrics": {
                "mean_test_score": mean_test_score,
                "std_test_score": std_test_score,
                "mean_val_score": mean_val_score,
                "std_val_score": std_val_score,
                "average_pass_rate": average_pass_rate,
                "average_total_prompts": average_total_prompts,
                "average_training_time": average_training_time,
                "average_subset_pass_rate": average_subset_pass_rate,
                "average_valset_pass_rate": average_valset_pass_rate,
            },
        }

    print("\nAll Checkpoints:")
    for experiment, ckpt in ckpt_values.items():
        print(f"{experiment}: {ckpt}")

    # Save the results to a file
    with open(result_file, "w") as f:
        json.dump(ckpt_values, f, indent=4)

    print(f"\nResults saved to {result_file}")
