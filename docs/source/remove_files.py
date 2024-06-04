import os


def remove_file(directory: str):
    """Remove duplicated files.
    
    During the automatic generation, some files are not used or duplicated, including:
    * modules.rst
    * components.rst, core.rst, prompt.rst, ... , corresponding to directory name
    * duplicated files in autosummary and api reference directory

    Args:
        directory (str): directory that contains duplicated files
    """
    
    # remove modules.rst
    try:
        for filename in os.listdir(directory):
            module_file = os.path.join(directory, "modules.rst")
            os.remove(module_file)
    except:
        print(f"No files to remove in {directory}")
    
    # remove components.rst, core.rst, prompt.rst, ...
    try:
        for filename in os.listdir(directory):
            name = directory.split('/')[-1] + ".rst"
            module_file = os.path.join(directory, name)
            os.remove(module_file)
    except:
        print(f"No files to remove in {directory}")
        
    # remove api files to avoid showing duplicated section
    autosummary_directory = os.path.join(directory, '_autosummary')
    autosummary_files = []
    try:
        for filename in os.listdir(autosummary_directory):
            if filename.endswith(".rst"):
                # filepath = os.path.join(autosummary_directory, filename)
                autosummary_files.append(filename)
        
        for filename in os.listdir(directory):
            if filename.endswith(".rst"):
                filepath = os.path.join(directory, filename)
                if filename in autosummary_files: # if it is duplicated in autosummary, remove it
                    os.remove(filepath)
                    print(f"{filepath} is removed")
    except:
        print(f"{directory}/_autosummary not existing")

remove_file("./source/apis/components")
remove_file("./source/apis/core")
remove_file("./source/apis/eval")
remove_file("./source/apis/prompts")
remove_file("./source/apis/utils")
                
            
