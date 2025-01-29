"""GradientContext and Gradient"""

import uuid
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from adalflow.optim.parameter import Parameter
from dataclasses import dataclass, field
from adalflow.core.base_data_class import DataClass

__all__ = ["GradientContext", "Gradient"]


@dataclass
class GradientContext(DataClass):
    """GradientContext is used to describe the component's function and trace its input and output.

    To get the component's function desc, use GradientContext.to_yaml_signature()
    To get the data: use instance.to_yaml()
    """

    variable_desc: str = field(
        metadata={"desc": "The description of the target parameter"}
    )

    input_output: str = field(
        metadata={
            "desc": "The context of the gradient in form of a conversation indicating \
                the relation of the current parameter to the response parameter"
        }
    )
    response_desc: str = field(
        metadata={"desc": "The description of the response parameter"}
    )
    # input: Dict[str, Any] = field(
    #     metadata={"desc": "The input to the whole system"}, default=None
    # )

    # ground_truth: Any = field(
    #     metadata={"desc": "The ground truth of the response parameter"}, default=None
    # )


@dataclass
class Gradient(DataClass):
    __doc__ = r"""It will handle gradients and feedbacks.

    It tracks the d_from_response_id / d_to_pred_id and the score of the whole response.

    if two gradients have the same data_id, different from_response_id, and same from_response_component_id, this is a cycle component structure.
    """
    data_id: Optional[str] = None  # the id of the response from data in the dataset
    from_response_component_id: str = (
        None  # the id of the component from which the gradient is calculated
    )
    order: Optional[int] = None  # the order of the gradient in the list of gradients

    from_response_id: str = (
        None  # the id of the response from which the gradient is calculated
    )

    to_pred_id: str = (
        None  # the id of the parameter to which the gradient is calculated and attached to d(from_response_id) / d(to_pred_id)
    )

    score: Optional[float] = None

    context: GradientContext = None
    data: Any = None
    prompt: Optional[str] = None  # the LLM prompt to generate the gradient

    is_default_copy: bool = False  # whether the gradient is a default copy

    def __init__(
        self,
        *,
        from_response: "Parameter",
        to_pred: "Parameter",
        id: Optional[str] = None,  # the id of the gradient
        score: Optional[float] = None,
        data_id: Optional[str] = None,
        data: Any = None,
    ):
        self.id = id or str(uuid.uuid4())
        self._generate_name(from_response, to_pred)
        self.from_response_component_id = from_response.component_trace.id
        if not self.from_response_component_id:
            raise ValueError(
                "The from_response_component_id should not be None. Please ensure the component_trace is set."
            )
        self.from_response_id = from_response.id
        self.to_pred_id = to_pred.id
        self.score = score
        self.data_id = data_id
        if self.data_id is None:
            raise ValueError("The data_id should not be None.")
        self.data = data
        self.order = None

    def _generate_name(self, response: "Parameter", pred: "Parameter"):
        self.name = f"d_{response.name}_/_{pred.name}({response.id}_/_{pred.id})"
        self.role_desc = f"Gradient from {response.name} to {pred.name}"

    def add_context(self, context: GradientContext):
        self.context = context

    def add_data(self, data: Any):
        self.data = data

    def update_from_to(self, from_response: "Parameter", to_pred: "Parameter"):
        self.from_response_id = from_response.id
        self.to_pred_id = to_pred.id
        self._generate_name(from_response, to_pred)
        self.from_response_component_id = from_response.component_trace.id

    def add_prompt(self, prompt: str):
        self.prompt = prompt

    def __hash__(self):
        # Use immutable and unique attributes to compute the hash
        return hash((self.id, self.data_id, self.from_response_id, self.to_pred_id))

    def __eq__(self, other):
        # Ensure equality comparison is based on the same unique attributes
        if not isinstance(other, Gradient):
            return False
        return (
            self.id == other.id
            and self.data_id == other.data_id
            and self.from_response_id == other.from_response_id
            and self.to_pred_id == other.to_pred_id
        )
