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
        for _ in os.listdir(directory):
            module_file = os.path.join(directory, "modules.rst")
            os.remove(module_file)
    except Exception:
        print(f"No modules.rst to remove in {directory}")

    # remove components.rst, core.rst, prompt.rst, ...
    try:
        for _ in os.listdir(directory):
            name = directory.split("/")[-1] + ".rst"
            module_file = os.path.join(directory, name)
            os.remove(module_file)
    except Exception:
        print(f"No {name} to remove in {directory}")

remove_file("./source/apis/components")
remove_file("./source/apis/core")
remove_file("./source/apis/eval")
remove_file("./source/apis/utils")
remove_file("./source/apis/tracing")
remove_file("./source/apis/optim")
