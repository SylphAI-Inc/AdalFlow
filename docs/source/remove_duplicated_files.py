import os


def remove_duplication(directory: str):
    """Remove duplicated API references

    Args:
        directory (str): directory that contains duplicated files
    """
    autosummary_directory = os.path.join(directory, '_autosummary')
    autosummary_files = []
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

remove_duplication("./source/apis/components")
                
            
