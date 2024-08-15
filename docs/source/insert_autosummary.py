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
                if middle != ".":
                    files.add(os.path.join(middle, filename))
                else:
                    files.add(filename)
        return list(files)
    except Exception as e:
        print(f"Error listing files in directory {directory}: {e}")
        return []


def generate_rst_for_module(module_full_name, module, output_dir):

    rst_filepath = os.path.join(output_dir, module_full_name + ".rst")

    # insert the following summary after the automodule directive

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

    # Collect constants from __all__
    constants = []
    if hasattr(module, "__all__"):
        all_members = getattr(module, "__all__")
        for const_name in all_members:
            const_value = getattr(module, const_name, None)
            if (
                const_value is not None
                and not inspect.isfunction(const_value)
                and not inspect.isclass(const_value)
            ):
                constants.append((const_name, const_value))

    if constants:
        content += "   .. rubric:: Constants\n\n"
        for const_name, const_value in constants:
            content += f"   .. autodata:: {module_full_name}.{const_name}\n"
        content += "\n"

    with open(rst_filepath, "a") as rst_file:
        rst_file.write(content)


def generate_autosummary_docs(src_dir, dest_dir):
    print(f"Generating autosummary docs for {src_dir} to {dest_dir}")
    module = src_dir.split("/")[-1]
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
            # spec and load the module
            try:
                spec = importlib.util.spec_from_file_location(
                    module_full_name, code_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                generate_rst_for_module(module_full_name, module, module_dir)
            except Exception as e:
                print(f"Error loading module {module_full_name}: {e}")
                continue


if __name__ == "__main__":

    source_root_dir = "../lightrag/lightrag"

    source_directories = [
        "core",
        "components",
        "eval",
        "datasets",
        "utils",
        "tracing",
        "optim",
    ]
    dest_directories = [
        "./source/apis/core",
        "./source/apis/components",
        "./source/apis/eval",
        "./source/apis/datasets",
        "./source/apis/utils",
        "./source/apis/tracing",
        "./source/apis/optim",
    ]

    for source_dir, dest_dir in zip(source_directories, dest_directories):
        print(f"source_dir: {source_dir}, dest_dir: {dest_dir}")
        source_path = os.path.join(source_root_dir, source_dir)
        generate_autosummary_docs(source_path, dest_dir)
