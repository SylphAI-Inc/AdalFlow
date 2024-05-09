from typing import Dict, List, Union
from core.documents_data_class import RetrieverOutput, Chunk

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
    chunks_to_use: List[Chunk] = []
    context_str = ""
    sep = " "
    if isinstance(retriever_output, RetrieverOutput):
        chunks_to_use = retriever_output.chunks
    else:
        for output in retriever_output:
            chunks_to_use.extend(output.chunks)
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
