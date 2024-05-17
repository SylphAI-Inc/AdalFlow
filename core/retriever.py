from typing import List, Optional, Union, Callable, Any, Sequence, Any

import faiss
import numpy as np

from core.component import Component
from core.data_classes import (
    RetrieverOutput,
)

RetrieverInputType = Union[str, List[str]]  # query
RetrieverDocumentType = Any  # Documents
RetrieverOutputType = List[RetrieverOutput]


class Retriever(Component):
    """
    Retriever will manage its own index and retrieve in format of RetrieverOutput
    It takes a list of Document and builds index used for retrieval use anyway formed content from fields of the document
    using the input_field_map_func
    """

    indexed = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_index(self):
        raise NotImplementedError(f"reset_index is not implemented")

    # def _get_inputs(self, documents: List[Any], input_field_map_func: Callable):
    #     return [input_field_map_func(document) for document in documents]

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentType,
        input_field_map_func: Callable[[Any], Any] = lambda x: x.text,
    ):
        r"""Built index from the `text` field of each document in the list of documents.
        input_field_map_func: a function that maps the document to the input field to be used for indexing
        You can use _get_inputs to get a standard format fits for this retriever or you can write your own
        """
        raise NotImplementedError(
            f"build_index_from_documents and input_field_map_func is not implemented"
        )

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> Any:
        raise NotImplementedError(f"retrieve is not implemented")

    def __call__(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> Any:
        raise NotImplementedError(f"__call__ is not implemented")
