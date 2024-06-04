import os


def add_reference_labels(directory: str):
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".rst") and "index" not in filename:
                filepath = os.path.join(directory, filename)
                with open(filepath, "r+") as file:
                    content = file.read()
                    file.seek(0, 0)
                    module_label = filename.replace(".rst", "").replace(".", "-")
                    if module_label not in content:
                        label_line = f".. _{module_label}:\n\n"
                        file.write(label_line + content)
    except:
        print(f"directory {directory} not exists")

if __name__ == "__main__":
    # Specify the directories you want to process
    add_reference_labels("./source/apis/core")
    add_reference_labels("./source/apis/core/_autosummary")
    add_reference_labels("./source/apis/components")
    add_reference_labels("./source/apis/components/_autosummary")
    add_reference_labels("./source/apis/eval")
    add_reference_labels("./source/apis/eval/_autosummary")
    add_reference_labels("./source/apis/prompts")
    add_reference_labels("./source/apis/prompts/_autosummary")
    add_reference_labels("./source/apis/utils")
    add_reference_labels("./source/apis/utils/_autosummary")