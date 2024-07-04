import os
import subprocess

# Define the directories to exclude
exclude_dirs = ["source/apis/components/_autosummary"]


# Function to check if a path should be excluded
def should_exclude(path):
    for excl in exclude_dirs:
        if path.startswith(excl):
            return True
    return False


# Find all .rst files, excluding those in the exclude_dirs
rst_files = []
for root, _, files in os.walk("source"):
    for file in files:
        if file.endswith(".rst"):
            full_path = os.path.join(root, file)
            if not should_exclude(full_path):
                rst_files.append(full_path)

# Run sphinx-autogen on the filtered list of .rst files
if rst_files:
    subprocess.run(["sphinx-autogen"] + rst_files)
