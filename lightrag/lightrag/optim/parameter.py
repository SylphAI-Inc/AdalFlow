"""WIP"""

from typing import Generic, TypeVar, Any, List, Set, Dict, TYPE_CHECKING
from collections import defaultdict
import logging

# from lightrag.core import Generator

from lightrag.optim.text_grad.backend_engine_prompt import GRADIENT_TEMPLATE

if TYPE_CHECKING:
    pass  # Import Generator for type checking only

T = TypeVar("T")  # covariant set to False to allow for in-place updates

log = logging.getLogger(__name__)


# tensor has the backward function
# Future direction: potentially subclass DataClass, especially when parameter becomes more complex
class Parameter(Generic[T]):
    r"""A data container to represent a component parameter.

    A parameter enforce a specific data type and can be updated in-place.
    When parameters are used in a component - when they are assigned as Component attributes
    they are automatically added to the list of its parameters, and  will
    appear in the :meth:`~Component.parameters` iterator.

    Args:
        data (T): the data of the parameter
        requires_opt (bool, optional): if the parameter requires optimization. Default: `True`

    Examples:

    * Specify the type explicitly via Generic:

    .. code-block:: python

        int_param = Parameter[int](data=123)
        str_param = Parameter[str](data="hello")
        # update the value in-place
        int_param.update_value(456)
        # expect a TypeError as a string is not an integer
        int_param.update_value("a string")

    * Specify the type implicitly via the first data assignment:

    .. code-block:: python

        # infer the type from the provided data
        param = Parameter(data=123)
        param.update_value(456)
        # expect a TypeError if the type is incorrect
        param.update_value("a string")

    References:

    1. https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    def __init__(
        self,
        data: T = None,
        requires_opt: bool = True,
        role_desc: str = None,
        predecessors: List["Parameter"] = None,
        alias: str = None,  # alias is used to refer to the parameter in the prompt, easier to read for humans
    ):
        if predecessors is None:
            predecessors = []
        _predecessors_requires_grad = [v for v in predecessors if v.requires_opt]
        self.data = data
        self.requires_opt = requires_opt
        self.data_type = type(
            data
        )  # Dynamically infer the data type from the provided data
        self.role_desc = role_desc
        self.predecessors = set(predecessors)
        self.gradients: Set[Parameter] = set()
        self.gradients_context: Dict[Parameter, str] = defaultdict(lambda: None)
        self.grad_fn = None
        self.alias = alias
        self.proposed_data: T = None  # this is for the optimizer to propose a new value

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn

    def get_grad_fn(self):
        return self.grad_fn

    def _check_data_type(self, new_data: Any):
        """Check the type of new_data against the expected data type."""
        if self.data is not None and not isinstance(new_data, self.data_type):
            raise TypeError(
                f"Expected data type {self.data_type.__name__}, got {type(new_data).__name__}"
            )

    def update_value(self, data: T):
        """Update the parameter's value in-place, checking for type correctness."""
        self._check_data_type(data)
        if self.data is None and data is not None:
            self.data_type = type(data)
        self.data = data

    def reset_gradients(self):
        self.gradients = set()

    def get_gradient_text(self) -> str:
        """Aggregates and returns the gradients."""
        return "\n".join([g.data for g in self.gradients])

    def get_gradient_and_context_text(self) -> str:
        """Aggregates and returns:
        1. the gradients
        2. the context text for which the gradients are computed
        """
        from lightrag.core.prompt_builder import Prompt

        gradients: List[str] = []
        for g in self.gradients:
            if (
                self.gradients_context[g] is None
            ):  # no context function provided, directly use the data
                gradients.append(g.data)

            else:
                criticism_and_context = Prompt(
                    template=GRADIENT_TEMPLATE,
                    prompt_kwargs={"feedback": g.data, **self.gradients_context[g]},
                )
                gradients.append(criticism_and_context())
        gradient_text = "\n".join(gradients)

        return gradient_text

    def get_short_value(self, n_words_offset: int = 10) -> str:
        """
        Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
        This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
        :param n_words_offset: The number of words to show from the beginning and the end of the value.
        :type n_words_offset: int
        """
        words = self.data.split(" ")
        if len(words) <= 2 * n_words_offset:
            return self.data
        short_value = (
            " ".join(words[:n_words_offset])
            + " (...) "
            + " ".join(words[-n_words_offset:])
        )
        return short_value

    def backward(self):  # engine should be the llm
        # if engine is None:
        #     raise ValueError("Engine is not provided.")
        # topological sort of all the predecessors of the current parameter in the graph
        log.debug(f"Backward pass for {self.data}, backward function: {self.grad_fn}")
        topo: List[Parameter] = []
        visited = set()

        def build_topo(node: Parameter):
            if node in visited:
                return
            visited.add(node)
            for pred in node.predecessors:
                build_topo(pred)
            topo.append(node)

        build_topo(self)
        # backpropagation

        self.gradients = set()
        for node in reversed(topo):
            if not node.requires_opt:
                log.debug(f"Skipping {node.data} as it does not require optimization")
                continue
            node.gradients = _check_and_reduce_gradients(node)
            log.debug(f"v: {node.data}, grad_fn: {node.grad_fn}, {node.get_grad_fn()}")
            if node.get_grad_fn() is not None:  # gradient function takes in the engine
                log.debug(f"Calling gradient function for {node.data}")
                node.grad_fn()

    # def to_dict(self):
    #     return {
    #         "data": self.data,
    #         "requires_opt": self.requires_opt,
    #         "role_desc": self.role_desc,
    #         "predecessors": self.predecessors,
    #     }

    # @classmethod
    # def from_dict(cls, data: dict):
    #     return cls(data=data["data"], requires_opt=data["requires_opt"])

    def to_dict(self):
        return {
            "alias": self.alias,
            "data": self.data,
            "requires_opt": self.requires_opt,
            "role_desc": self.role_desc,
            "predecessors": [pred.to_dict() for pred in self.predecessors],
            "gradients": [grad.to_dict() for grad in self.gradients],
            "proposed_data": self.proposed_data,
            "gradients_context": [
                (k.to_dict(), v) for k, v in self.gradients_context.items()
            ],
            "grad_fn": str(
                self.grad_fn
            ),  # Simplify for serialization, modify as needed
        }

    @classmethod
    def from_dict(cls, data: dict):
        predecessors = [cls.from_dict(pred) for pred in data["predecessors"]]
        param = cls(
            data=data["data"],
            requires_opt=data["requires_opt"],
            role_desc=data["role_desc"],
            predecessors=predecessors,
            alias=data["alias"],
            gradients=[cls.from_dict(grad) for grad in data["gradients"]],
            proposed_data=data["proposed_data"],
        )
        # Reconstruct gradients_context from the list of tuples
        param.gradients_context = defaultdict(
            lambda: None, {cls.from_dict(k): v for k, v in data["gradients_context"]}
        )
        return param

    # TODO: very hard to read directly, need to simplify and let users use to_dict for better readability
    def __repr__(self):
        return f"Parameter(alias={self.alias}, data={self.data}, requires_opt={self.requires_opt}, role_desc={self.role_desc}, predecessors={self.predecessors}, gradients={self.gradients})"


def _check_and_reduce_gradients(variable: Parameter) -> Set[Parameter]:

    if variable.get_gradient_text() == "":
        log.debug(f"No gradients detected for {variable.data}")
        return variable.gradients
    if len(variable.gradients) == 1:
        log.debug(f"Only one gradient, no need to reduce: {variable.gradients}")
        return variable.gradients
    else:
        log.warning(
            f"Multiple gradients detected for {variable.data}. Reducing the gradients."
        )
        return variable.gradients

    # TODO: Implement the reduction logic later
