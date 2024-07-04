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
    except Exception:
        print(f"No modules.rst to remove in {directory}")

    # remove components.rst, core.rst, prompt.rst, ...
    try:
        for filename in os.listdir(directory):
            name = directory.split("/")[-1] + ".rst"
            module_file = os.path.join(directory, name)
            os.remove(module_file)
    except Exception:
        print(f"No {name} to remove in {directory}")

    # remove api files to avoid showing duplicated section

    target_files = [
        "components.model_client.openai_client.rst",
        "components.retriever.faiss_retriever.rst",
        "components.reasoning.chain_of_thought.rst",
        "components.model_client.groq_client.rst",
        "components.retriever.bm25_retriever.rst",
        "components.model_client.google_client.rst",
        "components.model_client.transformers_client.rst",
        "components.model_client.utils.rst",
        "components.retriever.llm_retriever.rst",
        "components.agent.react.rst",
        "components.model_client.anthropic_client.rst",
        "components.output_parsers.outputs.rst",
        "components.model_client.cohere_client.rst",
        "components.retriever.reranker_retriever.rst",
        "components.data_process.data_components.rst",
        "components.data_process.text_splitter.rst",
        "components.memory.memory.rst",
        # "components.retriever.postgres_retriever.rst",
    ]
    try:
        for filename in os.listdir(directory):
            if filename in target_files:
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                print(f"{filepath} is removed")
    except Exception:
        print(f"No target files to remove in {directory}")


remove_file("./source/apis/components")
remove_file("./source/apis/core")
remove_file("./source/apis/eval")
remove_file("./source/apis/utils")
remove_file("./source/apis/tracing")
remove_file("./source/apis/optim")
