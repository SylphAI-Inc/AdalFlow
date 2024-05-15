from typing import Dict, List, Union, Any, Callable
from core.data_classes import RetrieverOutput, Document

from core.component import Component

# TODO: import all other  functions into this single file to be exposed to users


def compose_model_kwargs(default_model_kwargs: Dict, model_kwargs: Dict) -> Dict:
    r"""
    The model configuration exclude the input itself.
    Combine the default model, model_kwargs with the passed model_kwargs.
    Example:
    model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
    self.model_kwargs = {"model": "gpt-3.5"}
    combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

    """
    pass_model_kwargs = default_model_kwargs.copy()

    if model_kwargs:
        pass_model_kwargs.update(model_kwargs)
    return pass_model_kwargs


def retriever_output_to_context_str(
    retriever_output: Union[RetrieverOutput, List[RetrieverOutput]],
    deduplicate: bool = False,
) -> str:
    r"""The retrieved documents from one or mulitple queries.
    Deduplicate is especially helpful when you used query expansion.
    """
    """
    How to combine your retrieved chunks into the context is highly dependent on your use case.
    If you used query expansion, you might want to deduplicate the chunks.
    """
    chunks_to_use: List[Document] = []
    context_str = ""
    sep = " "
    if isinstance(retriever_output, RetrieverOutput):
        chunks_to_use = retriever_output.documents
    else:
        for output in retriever_output:
            chunks_to_use.extend(output.documents)
    if deduplicate:
        unique_chunks_ids = set([chunk.id for chunk in chunks_to_use])
        # id and if it is used, it will be True
        used_chunk_in_context_str: Dict[Any, bool] = {
            id: False for id in unique_chunks_ids
        }
        for chunk in chunks_to_use:
            if not used_chunk_in_context_str[chunk.id]:
                context_str += sep + chunk.text
                used_chunk_in_context_str[chunk.id] = True
    else:
        context_str = sep.join([chunk.text for chunk in chunks_to_use])
    return context_str


import hashlib
import json


def generate_component_key(component: Component) -> str:
    """
    Generates a unique key for a Component instance based on its type,
    version, configuration, and nested components.
    """
    # Start with the basic information: class name and version
    key_parts = {
        "class_name": component._get_name(),
        "version": getattr(component, "_version", 0),
    }

    # If the component stores configuration directly, serialize this configuration
    if hasattr(component, "get_config"):
        config = (
            component.get_config()
        )  # Ensure this method returns a serializable dictionary
        key_parts["config"] = json.dumps(config, sort_keys=True)

    # If the component contains other components, include their keys
    if hasattr(component, "_components") and component._components:
        nested_keys = {}
        for name, subcomponent in component._components.items():
            if subcomponent:
                nested_keys[name] = generate_component_key(subcomponent)
        key_parts["nested"] = nested_keys

    # Serialize key_parts to a string and hash it
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def generate_readable_key_for_function(fn: Callable) -> str:

    module_name = fn.__module__
    function_name = fn.__name__
    return f"{module_name}.{function_name}"


"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


# https://en.wikipedia.org/wiki/Okapi_BM25
# word can be a token or a real word
# Trotmam et al, Improvements to BM25 and Language Models Examined
"""
Retrieval is highly dependent on the database.

db-> transformer -> (index) should be a pair
LocalDocumentDB:  [Local Document RAG]
(1) algorithm, (2) index, build_index_from_documents (3) retrieve (top_k, query)

What algorithm will do for LocalDocumentDB:
(1) Build_index_from_documents (2) retrieval initialization (3) retrieve (top_k, query), potentially with score.

InMemoryRetriever: (Component)
(1) load_documents (2) build_index_from_documents (3) retrieve (top_k, query)

PostgresDB:
(1) sql_query for retrieval (2) pg_vector for retrieval (3) retrieve (top_k, query)

MemoryDB:
(1) chat_history (2) provide different retrieval methods, allow specify retrievel method at init.

Generator:
(1) prompt
(2) api_client (model)
(3) output_processors

Retriever
(1) 
"""
