r"""Helper components for data types transformation.

It is commonly used as output_processors.
"""

from copy import deepcopy
from typing import Any, List, TypeVar, Sequence, Union, Dict, Any


from lightrag.core.component import Component
from lightrag.core.types import (
    EmbedderOutput,
    Embedding,
    Usage,
    Document,
    RetrieverOutput,
)
import lightrag.core.functional as F


T = TypeVar("T")


# TODO: move this to the ModelClient.
def parse_embedding_response(
    api_response,
) -> EmbedderOutput:
    r"""Parse embedding model output from the API response to EmbedderOutput.

    Follows the OpenAI API response pattern.
    """
    # Assuming `api_response` has `.embeddings` and `.usage` attributes
    # and that `embeddings` is a list of objects that can be converted to `Embedding` dataclass
    # TODO: check if any embedding is missing
    embeddings = [
        Embedding(embedding=e.embedding, index=e.index) for e in api_response.data
    ]
    usage = Usage(
        prompt_tokens=api_response.usage.prompt_tokens,
        total_tokens=api_response.usage.total_tokens,
    )  # Assuming `usage` is an object with a `count` attribute

    # Assuming the model name is part of the response or set statically here
    model = api_response.model

    return EmbedderOutput(data=embeddings, model=model, usage=usage)


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


# class ToEmbedderResponse(Component):
#     """Convert embedding model output to EmbedderOutput
#     .. note:
#        This might not apply to all Embedder models, this applys to one pattern
#     """

#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()

#     def __call__(self, input: Any) -> EmbedderOutput:
#         """
#         convert the model output to EmbedderOutput
#         """
#         try:
#             return parse_embedding_response(input)
#         except Exception as e:
#             raise ValueError(f"Error converting to EmbedderOutput: {e}")


"""
For now these are the data transformation components
"""


class ToEmbeddings(Component):
    r"""It transforms a Sequence of Chunks or Documents to a List of Embeddings.

    It operates on a copy of the input data, and does not modify the input data.
    """

    def __init__(self, vectorizer: Component, batch_size: int = 50) -> None:
        super().__init__()
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def __call__(self, input: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(input)
        for i in range(0, len(output), self.batch_size):
            batch = output[i : i + self.batch_size]
            embedder_output: EmbedderOutput = self.vectorizer(
                input=[chunk.text for chunk in batch]
            )
            vectors = embedder_output.data
            for j, vector in enumerate(vectors):
                output[i + j].vector = vector.embedding
            # update tracking
            # self.tracking["vectorizer"]["num_calls"] += 1
            # self.tracking["vectorizer"][
            #     "num_tokens"
            # ] += embedder_output.usage.total_tokens
        return output


class RetrieverOutputToContextStr(Component):
    r"""
    Wrap on functional F.retriever_output_to_context_str
    """

    def __init__(self, deduplicate: bool = False):
        super().__init__()
        self.deduplicate = deduplicate

    def __call__(
        self,
        input: Union[RetrieverOutput, List[RetrieverOutput]],
    ) -> str:
        return retriever_output_to_context_str(
            retriever_output=input, deduplicate=self.deduplicate
        )

    def _extra_repr(self) -> str:
        s = f"deduplicate={self.deduplicate}"
        return s
