r"""These helps parse model or api's output to commonly used data structures.
It is commonly used as output_processors.
"""

from copy import deepcopy
import dataclasses
from typing import Any, Dict, List, Type, TypeVar, Optional, Sequence, Union


from core.component import Component
from core.data_classes import (
    EmbedderResponse,
    Embedding,
    Usage,
    Document,
    RetrieverOutput,
)
import core.functional as F


T = TypeVar("T")


# TODO: move this to a separate file
def from_dict_to_dataclass(klass: Type[T], d: Dict[str, Any]) -> T:
    fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
    return klass(
        **{
            f: (
                from_dict_to_dataclass(fieldtypes[f], d[f])
                if dataclasses.is_dataclass(fieldtypes[f])
                else d[f]
            )
            for f in d
        }
    )


# from openai.types import CreateEmbeddingResponse


def convert_to_embedder_response(
    api_response,
) -> EmbedderResponse:
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

    return EmbedderResponse(data=embeddings, model=model, usage=usage)


class ToEmbedderResponse(Component):
    """
    TODO: this might not apply to all Embedder models, this applys to one pattern
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(self, input: Any) -> EmbedderResponse:
        """
        convert the model output to EmbedderResponse
        """
        return convert_to_embedder_response(input)
        # return from_dict_to_dataclass(EmbedderResponse, input)


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
            embedder_output: EmbedderResponse = self.vectorizer(
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


# TODO: a helper factory that converts any given function to a component
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
        return F.retriever_output_to_context_str(
            retriever_output=input, deduplicate=self.deduplicate
        )

    def extra_repr(self) -> str:
        s = f"deduplicate={self.deduplicate}"
        return s
