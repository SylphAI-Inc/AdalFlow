r"""
The base class for all retrievers who in particular retrieve documents from a given database.
"""

from typing import (
    List,
    Optional,
    Generic,
    Dict,
    Any,
    Tuple,
    Callable,
)
import numpy as np

from lightrag.core.component import Component
from lightrag.core.types import (
    RetrieverInputType,
    RetrieverDocumentType,
    RetrieverDocumentsType,
    RetrieverOutputType,
)


def get_top_k_indices_scores(
    scores: List[float], top_k: int
) -> Tuple[List[int], List[float]]:
    scores_np = np.array(scores)
    top_k_indices = np.argsort(scores_np)[-top_k:][::-1]
    top_k_scores = scores_np[top_k_indices]
    return top_k_indices.tolist(), top_k_scores.tolist()


class Retriever(Component, Generic[RetrieverDocumentType, RetrieverInputType]):
    __doc__ = r"""The base class for all retrievers.

    Retriever will manage its own index and retrieve in format of RetrieverOutput
    
    Args:
        indexed (bool, optional): whether the retriever has an index. Defaults to False.
        index_keys (List[str], optional): attributes that define the index that can be used to restore the retriever. Defaults to [].
    
    The key method :meth:`build_index_from_documents` is the method to build the index from the documents.
    ``documents`` is a sequence of any type of document. With ``document_map_func``, you can map the document
    of Any type to the specific type ``RetrieverDocumentType`` that the retriever expects.
    
    note:
    To get the state of the retriever, leverage the :methd: "from_dict" and "to_dict" methods of the base class Component.


    """

    indexed: bool = False
    index_keys: List[str] = []  # attributes that define the index

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_index(self):
        raise NotImplementedError(f"reset_index is not implemented")

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentsType,
        document_map_func: Optional[Callable[[Any], RetrieverDocumentType]] = None,
        **kwargs,
    ):
        r"""Built index from the [document_map_func(doc) for doc in documents]."""
        raise NotImplementedError(
            f"build_index_from_documents and input_field_map_func is not implemented"
        )

    # TODO: this might not apply for cloud storage with built-in search engine
    # def load_index(self, index: Dict[str, Any]):
    #     r"""Load the index from a dictionary with expected index keys. Once loaded, turn the indexed to True"""
    #     if not all(key in index for key in self.index_keys):
    #         raise ValueError(
    #             f"Index keys are not complete. Expected keys: {self.index_keys}"
    #         )
    #     for key, value in index.items():
    #         setattr(self, key, value)
    #     self.indexed = True

    # def get_index(self) -> Dict[str, Any]:
    #     r"""Return the index as a dictionary. It is up to users to decide where and how to persist it."""
    #     if not self.indexed:
    #         raise ValueError(
    #             "Index is not built or loaded. Please either build the index or load it first."
    #         )
    #     return {key: getattr(self, key) for key in self.index_keys}

    def save_to_file(self, path: str):
        r"""Save the state including the index to a file.

        Optional for subclass to implement a default persistence method.
        """
        pass

    @classmethod
    def load_from_file(cls, path: str):
        r"""Load the index from a file.

        Optional for subclass to implement a default persistence method.
        """
        pass

    def retrieve(
        self,
        input: RetrieverInputType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"retrieve is not implemented")

    def __call__(
        self,
        input: RetrieverInputType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        return self.retrieve(input, top_k, **kwargs)
