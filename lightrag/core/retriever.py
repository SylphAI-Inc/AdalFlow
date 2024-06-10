r"""
The base class for all retrievers who in particular retrieve documents from a given database.
"""

from typing import List, Optional, Union, Any


from lightrag.core.component import Component
from lightrag.core.types import RetrieverOutput

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

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentType,
        **kwargs,
    ):
        r"""Built index from the `text` field of each document in the list of documents.
        input_field_map_func: a function that maps the document to the input field to be used for indexing
        You can use _get_inputs to get a standard format fits for this retriever or you can write your own
        """
        raise NotImplementedError(
            f"build_index_from_documents and input_field_map_func is not implemented"
        )

    def retrieve(
        self,
        query_or_queries: RetrieverInputType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"retrieve is not implemented")

    def __call__(
        self,
        query_or_queries: RetrieverInputType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"__call__ is not implemented")
