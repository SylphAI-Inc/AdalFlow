import os
import inspect
import importlib.util


def list_all_files(directory):
    try:
        files = set()
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                middle = os.path.relpath(root, directory)
                print(f"middle: {middle}")
                # files.append(os.path.join(root, filename))
                if middle != ".":
                    files.add(os.path.join(middle, filename))
                else:
                    files.add(filename)
        return list(files)
    except Exception as e:
        print(f"Error listing files in directory {directory}: {e}")
        return []


def generate_rst_for_module(module_full_name, module, output_dir):
    # Initialize the content with the automodule directive
    # content = f".. automodule:: {module_name}\n"
    # content += "   :members:\n"
    # content += "   :undoc-members:\n"
    # content += "   :show-inheritance:\n\n"
    # read the initial file and save the first five lines
    rst_filepath = os.path.join(output_dir, module_full_name + ".rst")
    # first_five_lines = []
    # with open(rst_filepath, "r") as file:
    #     for i in range(5):
    #         first_five_lines.append(file.readline())

    content = "\n\n"

    # Collect functions
    functions = [
        func_name
        for func_name, func in inspect.getmembers(module, inspect.isfunction)
        if not func_name.startswith("_") and func.__module__ == module_full_name
    ]
    if functions:
        content += "   .. rubric:: Functions\n\n"
        content += "   .. autosummary::\n\n"
        for func_name in functions:
            content += f"      {func_name}\n"
        content += "\n"

    # Collect classes
    classes = [
        class_name
        for class_name, cls in inspect.getmembers(module, inspect.isclass)
        if not class_name.startswith("_") and cls.__module__ == module_full_name
    ]
    if classes:
        content += "   .. rubric:: Classes\n\n"
        content += "   .. autosummary::\n\n"
        for class_name in classes:
            content += f"      {class_name}\n"
        content += "\n"

    with open(rst_filepath, "a") as rst_file:
        rst_file.write(content)


def generate_autosummary_docs(src_dir, dest_dir):
    print(f"Generating autosummary docs for {src_dir} to {dest_dir}")
    files = os.listdir(src_dir)
    module = src_dir.split("/")[-1]
    print(f"files: {files}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file in list_all_files(src_dir):
        if (
            file.endswith(".py")
            and not file.endswith("__init__.py")
            and not file.startswith("__")
        ):
            print(f"valid src_dir: {src_dir}, file: {file}")
            module = src_dir.replace(".py", "").split("/")[-1]
            print(f"module: {module}")
            code_path = os.path.join(src_dir, file)
            print(f"code_path: {code_path}")
            file_to_module = file.replace(".py", "").replace("/", ".")
            module_full_name = f"{module}.{file_to_module}"
            print(f"module_full_name: {module_full_name}")
            module_dir = dest_dir
            print(f"module_dir: {module_dir}")
            # spec and load the module
            spec = importlib.util.spec_from_file_location(module_full_name, code_path)
            print(f"spec: {spec}")
            module = importlib.util.module_from_spec(spec)
            print(f"module: {module}")
            spec.loader.exec_module(module)
            generate_rst_for_module(module_full_name, module, module_dir)
        #         module_path = os.path.relpath(
        #             os.path.join(root, file), src_dir
        #         ).replace(os.sep, ".")
        #         submodules = module_path.replace(".py", "")
        #         sub
        #         # module_name should be
        #         module_name = f"{module}.{file.}

        #         # Load the module
        #         spec = importlib.util.spec_from_file_location(
        #             module_name, os.path.join(root, file)
        #         )
        #         module = importlib.util.module_from_spec(spec)
        #         spec.loader.exec_module(module)

        #         # Generate the .rst file for the module
        #         generate_rst_for_module(module_name, module, dest_dir)


# if __name__ == "__main__":
#     src_dir = "path/to/your/python/modules"  # Adjust this path
#     dest_dir = "path/to/your/docs/source/_autosummary"  # Adjust this path

#     generate_autosummary_docs(src_dir, dest_dir)


if __name__ == "__main__":

    source_root_dir = "../lightrag/lightrag"

    source_directories = [
        "core",
        "components",
        "eval",
        "utils",
        "tracing",
        "optim",
    ]
    dest_directories = [
        "./source/apis/core",
        "./source/apis/components",
        "./source/apis/eval",
        "./source/apis/utils",
        "./source/apis/tracing",
        "./source/apis/optim",
    ]

    for source_dir, dest_dir in zip(source_directories, dest_directories):
        print(f"source_dir: {source_dir}, dest_dir: {dest_dir}")
        source_path = os.path.join(source_root_dir, source_dir)
        generate_autosummary_docs(source_path, dest_dir)
