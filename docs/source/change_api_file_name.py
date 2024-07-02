import os
import re

def update_file_content(directory):
    module_name = directory.split("/")[-1]
    
    for filename in os.listdir(directory):
        if filename.endswith(".rst") and "index" not in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, "r+", encoding='utf-8') as file:
                lines = file.readlines()
                modified = False  # To track if modifications have been made
                for i in range(len(lines) - 1):
                    line = lines[i].strip()
                    next_line = lines[i + 1].strip()
                    
                    # Check if the next line is a title underline
                    if next_line == "=" * len(next_line) and not modified:
                        # Check if the current line starts with the module_name
                        if line.startswith(module_name):
                            # Replace the full module path with only the last segment
                            new_title = line.split('.')[-1]
                            lines[i] = new_title + '\n'  # Update the title line
                            modified = True  # Mark that modification has been made
                            # No need to break since we are preserving the rest of the content

                # Rewind and update the file only if modifications were made
                if modified:
                    file.seek(0)
                    file.writelines(lines)
                    file.truncate()  # Ensure the file is cut off at the new end if it's shorter
                    print(f"Updated {filepath}")
                
# def rename_files(directory):
#     remove_prefix = directory.split("/")[-1] + "."
#     for filename in os.listdir(directory):
#         if filename.endswith(".rst") and remove_prefix in filename:
#             new_filename = filename.replace(remove_prefix, '')
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
#             print(f"Renamed {filename} to {new_filename}")
            

if __name__ == "__main__":
    # Specify the directory or directories you want to process
    directories = [
        "./source/apis/core",
        "./source/apis/components",
        "./source/apis/utils",
        "./source/apis/eval",
        "./source/apis/tracing",
        "./source/apis/optim",
    ]
    for diretory in directories:
        update_file_content(diretory)
        # rename_files(diretory)
    #     rename_and_update_files(dir)