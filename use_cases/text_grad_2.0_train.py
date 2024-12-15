import subprocess
import tempfile
import json

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
    object_count: "",
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
    for experiment in experiments:
        args = experiment_args.get(experiment, "")
        ckpt = run_experiment(experiment, args)
        if ckpt:
            ckpt_values[experiment] = ckpt

    print("\nAll Checkpoints:")
    for experiment, ckpt in ckpt_values.items():
        print(f"{experiment}: {ckpt}")
