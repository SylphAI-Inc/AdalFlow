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
    target_files = ['components.api_client.openai_client.rst', 'components.retriever.faiss_retriever.rst', 'components.reasoning.chain_of_thought.rst', 'components.api_client.groq_client.rst', 'components.retriever.bm25_retriever.rst', 'components.api_client.google_client.rst', 'components.api_client.transformers_client.rst', 'components.retriever.llm_retriever.rst', 'components.agent.react_agent.rst', 'components.api_client.anthropic_client.rst']
    try:
        for filename in os.listdir(directory):
            if filename in target_files:
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                print(f"{filepath} is removed")
    except:
        print(f"{filepath} not existing")
            
    
remove_file("./source/apis/components")
remove_file("./source/apis/core")
remove_file("./source/apis/eval")
remove_file("./source/apis/prompts")
remove_file("./source/apis/utils")
                
            
