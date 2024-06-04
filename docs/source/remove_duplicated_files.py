import os


def remove_duplication(directory: str):
    """Remove duplicated API references

    Args:
        directory (str): directory that contains duplicated files
    """
    
    try:
        for filename in os.listdir(directory):
            module_file = os.path.join(directory, "modules.rst")
            os.remove(module_file)
    except:
        print(f"No files to remove in {directory}")
        
    try:
        for filename in os.listdir(directory):
            name = directory.split('/')[-1] + ".rst"
            module_file = os.path.join(directory, name)
            os.remove(module_file)
    except:
        print(f"No files to remove in {directory}")
        
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
                if filename in autosummary_files:
                    os.remove(filepath)
                    print(f"{filepath} is removed")
    except:
        print(f"_autosummary not existing")

remove_duplication("./source/apis/components")
remove_duplication("./source/apis/core")
remove_duplication("./source/apis/eval")
remove_duplication("./source/apis/prompts")
remove_duplication("./source/apis/utils")
                
            
