"""Helper components for data transformation such as embeddings and document splitting."""

from copy import deepcopy
from typing import List, TypeVar, Sequence, Union, Dict, Any
from tqdm import tqdm


from lightrag.core.component import Component

from lightrag.core.types import (
    Document,
    RetrieverOutput,
)
from lightrag.core.embedder import (
    BatchEmbedder,
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
    Embedder,
)


T = TypeVar("T")
__all__ = [
    "ToEmbeddings",
    "RetrieverOutputToContextStr",
    "retriever_output_to_context_str",
]

# TODO: make the GeneratorOutput include the token usage too.


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


"""
For now these are the data transformation components
"""

ToEmbeddingsInputType = Sequence[Document]
ToEmbeddingsOutputType = Sequence[Document]


class ToEmbeddings(Component):
    r"""It transforms a Sequence of Chunks or Documents to a List of Embeddings.

    It operates on a copy of the input data, and does not modify the input data.
    """

    def __init__(self, embedder: Embedder, batch_size: int = 50) -> None:
        super().__init__(batch_size=batch_size)
        self.embedder = embedder
        self.batch_size = batch_size
        self.batch_embedder = BatchEmbedder(embedder=embedder, batch_size=batch_size)

    def __call__(self, input: ToEmbeddingsInputType) -> ToEmbeddingsOutputType:
        output = deepcopy(input)
        # convert documents to a list of strings
        embedder_input: BatchEmbedderInputType = [chunk.text for chunk in output]
        outputs: BatchEmbedderOutputType = self.batch_embedder(input=embedder_input)
        # n them back to the original order along with its query
        for batch_idx, batch_output in tqdm(
            enumerate(outputs), desc="Adding embeddings to documents from batch"
        ):
            for idx, embedding in enumerate(batch_output.data):
                output[batch_idx * self.batch_size + idx].vector = embedding.embedding
        return output

    def _extra_repr(self) -> str:
        s = f"batch_size={self.batch_size}"
        return s


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
