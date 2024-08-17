import os


def remove_bom(s):
    # Remove BOM if present
    return s.encode().decode("utf-8-sig")


def update_file_content(directory: str):
    module_name = (
        directory.split("/")[-1] if "_autosummary" not in directory else "components"
    )
    # print(f"directory: {directory}; module_name {module_name}")
    for filename in os.listdir(directory):
        # print(filename)
        if filename.endswith(".rst") and "index" not in filename:
            filepath = os.path.join(directory, filename)
            print(filepath, module_name)
            with open(filepath, "r+", encoding="utf-8") as file:
                lines = file.readlines()
                modified = False  # To track if modifications have been made
                if modified:
                    break
                for i in range(len(lines) - 1):
                    line = remove_bom(lines[i].strip())
                    # Replace escaped underscores with actual underscores
                    line = line.replace("\\_", "_")
                    next_line = lines[i + 1].strip()

                    # Check if the next line is a title underline
                    if next_line == "=" * len(next_line) and not modified:
                        print(f"filepath: {filepath}")
                        print(
                            f"line: {repr(line)}, module_name: {repr(module_name)}, {line.startswith(module_name)}"
                        )

                        # Check if the current line starts with the module_name
                        if line.startswith(module_name):
                            print(f"here: {line}, ")
                            # Replace the full module path with only the last segment
                            new_title = line.split(".")[-1]
                            # print(f"new_title: {new_title}")
                            lines[i] = new_title + "\n"  # Update the title line
                            print(
                                f"Updated title in {filepath}, lnew_title: {new_title}"
                            )
                            modified = True  # Mark that modification has been made
                            # No need to break since we are preserving the rest of the content

                # Rewind and update the file only if modifications were made
                if modified:
                    file.seek(0)
                    file.writelines(lines)
                    file.truncate()  # Ensure the file is cut off at the new end if it's shorter
                    print(f"Updated {filepath} ")


if __name__ == "__main__":
    # Specify the directory or directories you want to process
    directories = [
        "./source/apis/core",
        "./source/apis/components",
        # "./source/apis/components/_autosummary",
        "./source/apis/datasets",
        "./source/apis/utils",
        "./source/apis/eval",
        "./source/apis/tracing",
        "./source/apis/optim",
        # "./source/apis/components/_autosummary",
    ]
    for diretory in directories:
        update_file_content(diretory)
