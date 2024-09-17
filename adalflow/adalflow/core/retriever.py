r"""The base class for all retrievers who in particular retrieve documents from a given database."""

from typing import List, Optional, Generic, Any, Callable, TYPE_CHECKING, Union
import logging

from adalflow.core.types import (
    RetrieverQueriesType,
    RetrieverQueryType,
    RetrieverDocumentType,
    RetrieverDocumentsType,
    RetrieverOutputType,
)
from adalflow.optim.grad_component import GradComponent

if TYPE_CHECKING:
    from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.optim.types import ParameterType

log = logging.getLogger(__name__)


# TODO: tracing retriever in the diagnose files using callback manager
class Retriever(GradComponent, Generic[RetrieverDocumentType, RetrieverQueryType]):
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
    name: str = "Retriever"
    top_k: int

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_index(self):
        r"""Initialize/reset any attributes/states for the index."""
        raise NotImplementedError("reset_index is not implemented")

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentsType,
        document_map_func: Optional[Callable[[Any], RetrieverDocumentType]] = None,
        **kwargs,
    ):
        r"""Built index from the [document_map_func(doc) for doc in documents]."""
        raise NotImplementedError(
            "build_index_from_documents and input_field_map_func is not implemented"
        )

    def save_to_file(self, path: str):
        r"""Save the state, including the index to a file.

        Optional for subclass to implement a default persistence method.
        Subclass can leverge component's `to_dict` method to get the states and choose to save them in any file format.
        """
        pass

    @classmethod
    def load_from_file(cls, path: str):
        r"""Load the state, including index from a file to restore the retriever.

        Subclass can leverge component's `from_dict` method to restore the states from the file.
        """
        pass

    def call(
        self,
        input: RetrieverQueriesType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError("retrieve is not implemented")

    async def acall(
        self,
        input: RetrieverQueriesType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError("Async retrieve is not implemented")

    # TODO: adapt the generator to auto-track the prompt_kwargs as parameters
    def forward(
        self,
        input: Union[RetrieverQueriesType, Parameter],
        top_k: Optional[
            int
        ] = None,  # TODO: top_k can be trained in the future if its formulated as a parameter
        **kwargs,
    ) -> Parameter:
        r"""Customized forward on top of the GradComponent forward method.

        To track the input as Parameters and set the parameter type as RETRIEVER_OUTPUT in the response.
        """
        # convert input to parameter if it is not
        if not isinstance(input, Parameter):
            input = Parameter(
                data=input,
                name="input",
                requires_opt=True,
                param_type=ParameterType.INPUT,
            )
        # trace the top_k in the DAG graph
        top_k = Parameter(
            data=top_k or self.top_k,
            name="top_k",
            requires_opt=True,
            param_type=ParameterType.HYPERPARAM,
        )
        response = super().forward(input, top_k=top_k, **kwargs)
        response.param_type = (
            ParameterType.RETRIEVER_OUTPUT
        )  # be more specific about the type
        return response

    def backward(
        self,
        response: Parameter,
        id: Optional[str] = None,
        backward_engine: Optional["Generator"] = None,
    ):
        r"""Backward the response to pass the score to predecessors"""
        log.info(f"Retriever backward: {response}")
        pass
