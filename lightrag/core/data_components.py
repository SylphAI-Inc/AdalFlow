r"""Helper components for data types transformation.

It is commonly used as output_processors.
"""

from copy import deepcopy
from typing import Any, List, TypeVar, Sequence, Union, Dict, Any
from tqdm import tqdm


from lightrag.core.component import Component

from lightrag.core.types import (
    EmbedderOutput,
    Embedding,
    Usage,
    Document,
    RetrieverOutput,
)
from lightrag.core.embedder import (
    BatchEmbedder,
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
    Embedder,
)
import lightrag.core.functional as F

T = TypeVar("T")


# TODO: make the GeneratorOutput include the token usage too.
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


"""
For now these are the data transformation components
"""


class ToEmbeddings(Component):
    r"""It transforms a Sequence of Chunks or Documents to a List of Embeddings.

    It operates on a copy of the input data, and does not modify the input data.
    """

    def __init__(self, vectorizer: Embedder, batch_size: int = 50) -> None:
        super().__init__(batch_size=batch_size)
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.batch_embedder = BatchEmbedder(embedder=vectorizer, batch_size=batch_size)

    def __call__(self, input: Sequence[Document]) -> Sequence[Document]:
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
