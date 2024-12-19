import subprocess
import tempfile
import json

num_runs = 4
# List of experiments to run
object_count = "use_cases/question_answering/bbh/object_count/train_new.py"
hotpot_qa_multi_hop_rag = "benchmarks/hotpot_qa/adal_exp/train_multi_hop_rag.py"

ckpt_values = []
experiments = [
    object_count,
    # hotpot_qa_multi_hop_rag,
]

# Optional: Arguments for each experiment (if needed)
experiment_args = {
    object_count: "--strategy constrained",
    # hotpot_qa_multi_hop_rag: "",
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
        highest_test_score, mean_test_score, standard_deviation = 0, 0, 0
        past_highest_scores = []
        # average pass rate, average pass prompts
        average_pass_rate_list = []
        average_pass_prompts_list = []
        average_total_prompts = []
        total_prompts = 0
        highest_test_score_json_file = None
        for experiment_index, ckpt in ckpt_values.items():
            with open(ckpt, "r") as f:
                data = json.load(f)
                print(f"Experiment: {experiment_index}")
                print(f"Data: {data}")
                _high_test_score = max(data["val_scores"])
                print(f" val score: {data["val_scores"]}")
                past_highest_scores.append(_high_test_score)
                if _high_test_score > highest_test_score:
                    highest_test_score = _high_test_score
                    highest_test_score_json_file = ckpt
                # read the effective measures
                effective_measures = data.get("effective_measure", {})
                if not effective_measures:
                    total_prompts = len(data["val_scores"]) - 1
                    # count the total number of different test scores
                    pass_num = len(set(data["val_scores"])) - 1
                    average_pass_rate = pass_num / total_prompts
                    average_pass_rate_list.append(average_pass_rate)
                    average_pass_prompts_list.append(pass_num)
                    average_total_prompts.append(total_prompts)
                else:
                    total_prompts = (
                        effective_measures["subset"]["pass"]
                        + effective_measures["subset"]["fail"]
                    )

                    pass_num = effective_measures["valset"]["pass"]
                    total_val_prompts = (
                        effective_measures["valset"]["pass"]
                        + effective_measures["valset"]["fail"]
                    )
                    average_pass_rate = pass_num / total_val_prompts
                    average_pass_rate_list.append(average_pass_rate)
                    average_pass_prompts_list.append(pass_num)
                    average_total_prompts.append(total_prompts)
        # calculate the mean test score
        mean_test_score = sum(past_highest_scores) / len(past_highest_scores)
        # calculate the standard deviation
        standard_deviation = sum(
            [(x - mean_test_score) ** 2 for x in past_highest_scores]
        ) / len(past_highest_scores)
        standard_deviation = standard_deviation**0.5
        # calculate the average pass rate
        average_pass_rate = sum(average_pass_rate_list) / len(average_pass_rate_list)
        # calculate the average pass prompts
        average_pass_prompts = sum(average_pass_prompts_list) / len(
            average_pass_prompts_list
        )
        # calculate the average total prompts
        average_total_prompts = sum(average_total_prompts) / num_runs

        # add these numbers in the ckpt_values
        index = f"{experiment}_summary"
        ckpt_values[index] = {
            "config": {
                "num_runs": num_runs,
                "args": args,
            },
            "highest_test_score": highest_test_score,
            "mean_test_score": mean_test_score,
            "standard_deviation": standard_deviation,
            "highest_test_score_json_file": highest_test_score_json_file,
            "average_pass_rate": average_pass_rate,
            "average_pass_prompts": average_pass_prompts,
            "average_total_prompts": average_total_prompts,
            "past_highest_scores": past_highest_scores,
        }

    print("\nAll Checkpoints:")
    for experiment, ckpt in ckpt_values.items():
        print(f"{experiment}: {ckpt}")

    # Save the results to a file
    with open(result_file, "w") as f:
        json.dump(ckpt_values, f)

    print(f"\nResults saved to {result_file}")
