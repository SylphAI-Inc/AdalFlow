"""WIP"""

from typing import Generic, TypeVar, Any, List, Set, Dict
from collections import defaultdict

from lightrag.core.prompt_builder import Prompt
from lightrag.optim.text_grad.prompt_template import GRADIENT_TEMPLATE

T = TypeVar("T")  # covariant set to False to allow for in-place updates


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

    def backward(self):
        pass

    def to_dict(self):
        return {
            "data": self.data,
            "requires_opt": self.requires_opt,
            "role_desc": self.role_desc,
            "predecessors": self.predecessors,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data=data["data"], requires_opt=data["requires_opt"])

    def __repr__(self):
        return f"Parameter(data={self.data}, requires_opt={self.requires_opt}, role_desc={self.role_desc}, predecessors={self.predecessors})"
