r"""These helps parse model or api's output to commonly used data structures.
It is commonly used as output_processors.
"""

from core.component import Component
from core.data_classes import EmbedderResponse, Embedding, Usage
import dataclasses
from typing import Any, Dict, List, Type, TypeVar

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


def convert_to_embedder_response(api_response) -> EmbedderResponse:
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
