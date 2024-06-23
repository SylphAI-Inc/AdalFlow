import os
import ast
import pydot


def get_class_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Subscript):
        return get_class_name(node.value)
    elif isinstance(node, ast.Call):
        return get_class_name(node.func)
    return None


def get_classes_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            tree = ast.parse(file.read(), filename=file_path)

        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        classes = []
    print(f"Found {len(classes)} classes in {file_path}")
    return classes


def get_class_hierarchy(classes):
    graph = pydot.Dot(graph_type="digraph")
    excluded_bases = {
        "ComponentWithSuperInit",
        "ComponentMissSuperInit",
        "Memory",
        "ObjectTypes",
    }

    for cls in classes:
        for base in cls.bases:
            try:
                base_name = get_class_name(base)
                if base_name not in excluded_bases and cls.name not in excluded_bases:
                    print(f"Adding edge from {base_name} to {cls.name}")
                    graph.add_edge(pydot.Edge(base_name, cls.name))
            except AttributeError as e:
                print(f"Error processing {cls.name}: {base}, {e}")
                # base_name = base.value
                pass

    return graph


def visualize_class_hierarchy(graph, filename="class_hierarchy.png"):
    dpi = 800
    graph.set_graph_defaults(dpi=str(dpi))

    graph.write_png(filename, prog="dot")
    graph.write_dot(filename.replace(".png", ".dot"))
    graph.write_svg(filename.replace(".png", ".svg"))
    print(f"Class hierarchy saved as {filename}")


def process_directory(directory):
    all_classes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                classes = get_classes_from_file(file_path)
                all_classes.extend(classes)

    return all_classes


# Directory containing your Python files
def light_rag_paths():
    Light_rag_directory = "/Users/liyin/Documents/test/LightRAG/lightrag"

    paths = ["core", "components", "utils", "eval", "optim", "tracing"]
    for path in paths:
        yield os.path.join(Light_rag_directory, path)

    global graph_name
    graph_name = "lightrag_class_hierarchy.png"


def llama_index_paths():
    Llama_index_directory = "/Users/liyin/Documents/test/LightRAG/.venv/lib/python3.11/site-packages/llama_index"
    paths = ["core"]
    for path in paths:
        yield os.path.join(Llama_index_directory, path)
    yield Llama_index_directory

    global graph_name
    graph_name = "llama_index_class_hierarchy.png"


def lang_chain_paths():
    Lang_chain_directory = "/Users/liyin/Documents/test/LightRAG/.venv/lib/python3.11/site-packages/langchain"
    # paths = ["core"]
    # for path in paths:
    #     yield os.path.join(Lang_chain_directory, path)
    yield Lang_chain_directory

    global graph_name
    graph_name = "lang_chain_class_hierarchy.png"


if __name__ == "__main__":
    paths = light_rag_paths()
    # paths = llama_index_paths()
    # paths = lang_chain_paths()

    # Get all classes from the directory
    all_classes = []
    for path in paths:
        all_classes.extend(process_directory(path))

    # Generate the class hierarchy graph
    graph = get_class_hierarchy(all_classes)

    # Visualize the class hierarchy
    visualize_class_hierarchy(graph, filename=graph_name)
