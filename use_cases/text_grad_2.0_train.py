import subprocess

# List of experiments to run
experiments = [
    "experiment1.py",
    "experiment2.py",
    "experiment3.py",
]

# Optional: Arguments for each experiment (if needed)
experiment_args = {
    "experiment1.py": "",
    "experiment2.py": "",
    "experiment3.py": "",
}

# Loop through experiments and run them
for experiment in experiments:
    args = experiment_args.get(experiment, "")
    try:
        print(f"Running {experiment} with args: {args}")
        subprocess.run(f"python {experiment} {args}", check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment} failed with error: {e}")
