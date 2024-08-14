r"""The base class for all retrievers who in particular retrieve documents from a given database."""

from typing import List, Optional, Generic, Any, Callable, Union, TYPE_CHECKING
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
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.optim.function import BackwardContext

log = logging.getLogger(__name__)


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

    def forward(
        self,
        input: Union[RetrieverQueriesType, Parameter],
        top_k: Optional[
            int
        ] = None,  # TODO: top_k can be trained in the future if its formulated as a parameter
        id: Optional[str] = None,
        **kwargs,
    ) -> Parameter:
        r"""Training mode which will deal with parameter as predecessors"""
        input_args = {"input": input, "top_k": top_k, "id": id}
        predecessors = [p for p in [input, top_k, id] if isinstance(p, Parameter)]

        input_args_values = {}
        for k, v in input_args.items():
            if isinstance(v, Parameter):
                input_args_values[k] = v.data
            else:
                input_args_values[k] = v

        retriever_reponse = self.call(**input_args_values)

        response = Parameter(
            data=retriever_reponse,
            name=self.name + "_output",
            role_desc="Retriever response",
            input_args=input_args,
            full_response=retriever_reponse,
        )
        response.set_predecessors(predecessors)
        response.trace_forward_pass(
            input_args=input_args, full_response=retriever_reponse
        )
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                id=id,
            )
        )
        return response

    def backward(
        self,
        response: Parameter,
        id: Optional[str] = None,
        backward_engine: Optional["Generator"] = None,
    ):
        r"""Backward the response to pass the score to predecessors"""
        log.info(f"Retriever backward: {response}")
        children_params = response.predecessors
        if not self.tracing:
            return
        # backward score to the demo parameter
        for pred in children_params:
            if pred.requires_opt:
                # pred._score = float(response._score)
                pred.set_score(response._score)
                log.debug(
                    f"backpropagate the score {response._score} to {pred.name}, is_teacher: {self.teacher_mode}"
                )
                if pred.param_type == ParameterType.DEMOS:
                    # Accumulate the score to the demo
                    pred.add_score_to_trace(
                        trace_id=id, score=response._score, is_teacher=self.teacher_mode
                    )
                    log.debug(f"Pred: {pred.name}, traces: {pred._traces}")

    # def __call__(self, *args, **kwargs) -> Union[RetrieverOutputType, Any]:
    #     if self.training:
    #         log.debug("Training mode")
    #         return self.forward(*args, **kwargs)
    #     else:
    #         log.debug("Inference mode")
    #         return self.call(*args, **kwargs)
