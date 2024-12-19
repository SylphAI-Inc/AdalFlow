import json
import os
import math


def recompute_metrics_and_update_summary(result_file):
    try:
        # Load the results file
        with open(result_file, "r") as f:
            ckpt_values = json.load(f)

        # Initialize variables for metrics computation
        highest_test_score = 0
        mean_test_score = 0
        standard_deviation = 0
        past_highest_scores = []
        past_highest_test_scores = []

        average_pass_rate_list = []
        average_pass_prompts_list = []
        average_total_prompts_list = []

        highest_val_score = 0

        # Process each experiment
        for experiment, data in ckpt_values.items():
            if "summary" in experiment:
                continue  # Skip summary entries

            ckpt_path = data

            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as ckpt_file:
                    experiment_data = json.load(ckpt_file)

                val_scores = experiment_data.get("val_scores", [])
                test_scores = experiment_data.get("test_scores", [])
                _high_test_score = max(val_scores, default=0)
                _high_val_score = max(test_scores, default=0)

                past_highest_scores.append(_high_test_score)
                past_highest_test_scores.append(_high_val_score)

                if _high_test_score > highest_test_score:
                    highest_test_score = _high_test_score

                if _high_val_score > highest_val_score:
                    highest_val_score = _high_val_score

                effective_measures = experiment_data.get("effective_measure", {})

                if effective_measures:
                    pass_num = effective_measures["valset"].get("pass", 0)
                    total_val_prompts = effective_measures["valset"].get(
                        "pass", 0
                    ) + effective_measures["valset"].get("fail", 0)
                else:
                    total_val_prompts = len(val_scores) - 1
                    pass_num = len(set(val_scores))

                average_pass_rate = (
                    pass_num / total_val_prompts if total_val_prompts > 0 else 0
                )
                average_pass_rate_list.append(average_pass_rate)
                average_pass_prompts_list.append(pass_num)
                average_total_prompts_list.append(total_val_prompts)

        # Compute final metrics
        if past_highest_scores:
            mean_test_score = sum(past_highest_scores) / len(past_highest_scores)
            standard_deviation = math.sqrt(
                sum((x - mean_test_score) ** 2 for x in past_highest_scores)
                / len(past_highest_scores)
            )

        average_pass_rate = (
            sum(average_pass_rate_list) / len(average_pass_rate_list)
            if average_pass_rate_list
            else 0
        )
        average_pass_prompts = (
            sum(average_pass_prompts_list) / len(average_pass_prompts_list)
            if average_pass_prompts_list
            else 0
        )
        average_total_prompts = (
            sum(average_total_prompts_list) / len(average_total_prompts_list)
            if average_total_prompts_list
            else 0
        )

        # Update the summary in ckpt_values
        summary_key = "summary"
        ckpt_values[summary_key] = {
            "highest_test_score": highest_test_score,
            "mean_test_score": mean_test_score,
            "standard_deviation": standard_deviation,
            "average_pass_rate": average_pass_rate,
            "average_pass_prompts": average_pass_prompts,
            "average_total_prompts": average_total_prompts,
            "past_highest_scores": past_highest_scores,
            "past_highest_test_scores": past_highest_test_scores,
            "highest_val_score": highest_val_score,
        }

        # Save updated ckpt_values back to the file
        with open(result_file, "w") as f:
            json.dump(ckpt_values, f, indent=4)

        return ckpt_values[summary_key]

    except Exception as e:
        print(f"Error while recomputing metrics: {e}")
        return None


# Usage
if __name__ == "__main__":
    result_file = "results.json"  # Replace with your actual result file
    result_file = "text_grad_2_results_4_runs_1872c441-0db2-4640-9cf6-8ef910744a93.json"
    result_file = "text_grad_2_results_4_runs_02b9f463-aa21-4485-9899-07ac2542ddac.json"  # only use fullset
    summary = recompute_metrics_and_update_summary(result_file)

    if summary:
        print("Updated Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
