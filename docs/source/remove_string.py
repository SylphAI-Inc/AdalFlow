import os

def remove_unwanted_string(directory: str, target_string: str):
    """
    Removes the specified string from the end of titles in .rst files within the specified directory.

    Parameters:
    - directory (str): The path to the directory containing .rst files.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".rst"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r+") as file:
                lines = file.readlines()
                file.seek(0)
                file.truncate()
                for line in lines:
                    # Check if the line ends with 'module' and remove it
                    if line.strip().endswith(target_string):
                        line = line.replace(target_string, "")
                    file.write(line)

if __name__ == "__main__":
    # Specify the directory or directories you want to process
    directories = [
        "./source/apis/core",
        "./source/apis/components",
        "./source/apis/utils"
    ]
    for directory in directories:
        remove_unwanted_string(directory, 'module')
        remove_unwanted_string(directory, 'package')
        remove_unwanted_string(directory, 'Module contents')
        
