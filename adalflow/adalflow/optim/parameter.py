"""Parameter is used by Optimizer, Trainers, AdalComponent to auto-optimizations"""

from typing import (
    Generic,
    TypeVar,
    Any,
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    Literal,
    Callable,
    TYPE_CHECKING,
)
from collections import defaultdict
import logging
import os
from dataclasses import dataclass, field
import uuid
from adalflow.optim.types import ParameterType
from adalflow.core.base_data_class import DataClass

import html
from adalflow.optim.gradient import Gradient


if TYPE_CHECKING:
    from adalflow.optim.text_grad.tgd_optimizer import TGDData, TGDOptimizerTrace

T = TypeVar("T")  # covariant set to False to allow for in-place updates

log = logging.getLogger(__name__)

__all__ = ["Parameter", "ComponentNode", "ComponentTrace", "ScoreTrace"]


@dataclass
class ComponentTrace(DataClass):
    name: str = field(metadata={"desc": "The name of the component"}, default=None)
    id: str = field(metadata={"desc": "The unique id of the component"}, default=None)
    input_args: Dict[str, Any] = field(
        metadata={"desc": "The input arguments of the GradComponent forward"},
        default=None,
    )
    full_response: object = field(
        metadata={"desc": "The full response of the GradComponent output"}, default=None
    )
    raw_response: str = field(
        metadata={"desc": "The raw response of the generator"}, default=None
    )
    api_kwargs: Dict[str, Any] = field(
        metadata={
            "desc": "The api_kwargs for components like Generator and Retriever that pass to the model client"
        },
        default=None,
    )

    def to_context_str(self):
        output = f"""<INPUT>: {self.input_args}. <OUTPUT>: {self.full_response}"""
        return output


# TODO: use this to better trace the score
@dataclass
class ScoreTrace:
    score: float = field(metadata={"desc": "The score of the data point"}, default=None)
    eval_comp_id: str = field(
        metadata={"desc": "The id of the evaluation component"}, default=None
    )
    eval_comp_name: str = field(
        metadata={"desc": "The name of the evaluation component"}, default=None
    )


@dataclass(frozen=True)
class ComponentNode(DataClass):
    """Used to represent a node in the component graph."""

    id: str = field(metadata={"desc": "The unique id of the component"})
    name: str = field(metadata={"desc": "The name of the component"})
    type: Literal["INPUT", "COMPONENT"] = field(
        metadata={"desc": "The type of the node"}, default="COMPONENT"
    )


COMBINED_GRADIENTS_TEMPLATE = r"""
{% if component_schema %}
<COMPONENT_SCHEMA>
Gradients are from {{ component_schema | length }} components.
{% for component_id, schema in component_schema.items() %}
{{ schema }}
{% endfor %}
</COMPONENT_SCHEMA>
{% endif %}

{% if combined_gradients %}
{% for group in combined_gradients %}
<DataID: {{ loop.index }} >
<AVERAGE_SCORE>{{ group.average_score|round(2) }}</AVERAGE_SCORE>
{% for gradient in group.gradients %}
{{ loop.index }}.
INPUT_OUTPUT: {{ gradient.context }}
{% if gradient.score is not none %}
<SCORE>{{ gradient.score | round(3) }}</SCORE>
{% endif %}
{% if gradient.gradient is not none %}
<FEEDBACK>{{ gradient.gradient }}</FEEDBACK>
{% endif %}
{% endfor %}
</DataID>
{% endfor %}
{% endif %}
"""

# id: {{ component_id }}, remove using component id


class Parameter(Generic[T]):
    r"""A data container to represent a parameter used for optimization.

    A parameter enforce a specific data type and can be updated in-place.
    When parameters are used in a component - when they are assigned as Component attributes
    they are automatically added to the list of its parameters, and  will
    appear in the :meth:`~Component.parameters` or :meth:`~Component.named_parameters` method.

    Args:

    End users only need to create the Parameter with four arguments and pass it to the
    prompt_kwargs in the Generator.
        - data (str): the data of the parameter
        - requires_opt (bool, optional): if the parameter requires optimization. Default: `True`
        - role_desc.
        - param_type, incuding ParameterType.PROMPT for instruction optimization, ParameterType.DEMOS
        for few-shot optimization.
        - instruction_to_optimizer (str, optional): instruction to the optimizer. Default: `None`
        - instruction_to_backward_engine (str, optional): instruction to the backward engine. Default: `None`

    The parameter users created will be automatically assigned to the variable_name/key in the prompt_kwargs
    for easy reading and debugging in the trace_graph.

    References:

    1. https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    allowed_types = {
        ParameterType.NONE,
        ParameterType.PROMPT,
        ParameterType.DEMOS,
        ParameterType.HYPERPARAM,
        ParameterType.INPUT,
    }

    id: str = None  # Unique id of the parameter
    name: str = None  # Name of the parameter, easier to read for humans
    role_desc: str = ""  # Description of the role of the parameter
    data: T = None  # Data of the parameter
    data_id: str = (
        None  # Id of the data from the training set, used only for input_type
    )
    param_type: ParameterType

    proposing: bool = False  # State of the parameter
    predecessors: Set["Parameter"] = set()  # Predecessors of the parameter
    peers: Set["Parameter"] = set()  # Peers of the parameter
    eval_input: object = None  # Eval input passing to the eval_fn or evaluator you use
    successor_map_fn: Dict[str, Callable] = (
        None  # Map function to get the data from the output
    )

    tgd_optimizer_trace: "TGDOptimizerTrace" = None  # Trace of the TGD optimizer

    data_in_prompt: Callable = (
        None  # Callable to get the str of the data to be used in the prompt
    )
    gt: object = None  # Ground truth of the parameter

    def __init__(
        self,
        *,
        id: Optional[str] = None,  # unique id of the parameter
        data: T = None,
        data_id: str = None,  # for tracing the data item in the training/val/test set
        requires_opt: bool = True,
        role_desc: str = "",
        param_type: ParameterType = ParameterType.NONE,
        name: str = None,  # name is used to refer to the parameter in the prompt, easier to read for humans
        instruction_to_optimizer: str = None,
        instruction_to_backward_engine: str = None,
        score: Optional[float] = None,
        eval_input: object = None,
        successor_map_fn: Optional[Dict[str, Callable]] = None,
        data_in_prompt: Callable = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.data_id = data_id

        self.name = name
        self.role_desc = role_desc

        if not self.name:
            self.name = (
                self.role_desc.capitalize().replace(" ", "_")[0:10]
                if self.role_desc
                else f"param_{self.id}"
            )
        self.param_type = param_type
        # allow subclasses to override allowed_types dynamically
        allowed_types = getattr(self.__class__, "allowed_types", set())
        if param_type not in allowed_types:
            raise ValueError(
                f"{param_type.name} is not allowed for {self.__class__.__name__}"
            )

        self.data = data  # often string and will be used in the prompts
        self.requires_opt = requires_opt
        self.data_type = type(data)

        self.set_eval_fn_input(eval_input=data)
        self.gradients: Set[Gradient] = set()

        self.grad_fn = None

        self.previous_data = None  # used to store the previous data
        # context of the forward pass

        self.instruction_to_optimizer: str = instruction_to_optimizer
        self.instruction_to_backward_engine: str = instruction_to_backward_engine

        # here are used for demo parameter, filled by generator.forward
        self._traces: Dict[str, DataClass] = {}  # id to data items (DynamicDataClass)
        self._student_traces: Dict[str, DataClass] = {}  # id

        self.score: float = (
            score  # end to end evaluation score, TODO: might have multiple scores if using multiple eval fns  # score is set in the gradients in the backward pass
        )

        self._demos: List[DataClass] = (
            []
        )  # used for the optimizer to save the proposed demos
        self._previous_demos: List[DataClass] = []
        self.eval_input = eval_input

        self.successor_map_fn = successor_map_fn or {}

        def default_prompt_map_fn(param: Parameter):
            # if isinstance(param.data, GeneratorOutput):
            #     return param.data.raw_response
            return param.data

        self.data_in_prompt = data_in_prompt or default_prompt_map_fn
        self.gt = None

    def map_to_successor(self, successor: object) -> T:
        """Apply the map function to the successor based on the successor's id."""
        successor_id = id(successor)
        if successor_id not in self.successor_map_fn:
            default_map_fn = lambda x: x.data  # noqa: E731
            return default_map_fn(self)

        return self.successor_map_fn[successor_id](self)

    def add_successor_map_fn(self, successor: object, map_fn: Callable):
        """Add or update a map function of the value for a specific successor using its id.
        succssor will know the value of the current parameter."""
        self.successor_map_fn[id(successor)] = map_fn

    def check_if_already_computed_gradient_respect_to(self, response_id: str) -> bool:
        from_response_ids = [g.from_response_id for g in self.gradients]
        return response_id in from_response_ids

    ############################################################################################################
    # Handle gt
    ############################################################################################################
    def set_gt(self, gt: object):

        self.gt = gt

    def get_gt(self) -> object:
        return self.gt

    # ############################################################################################################
    # Handle gradients and context
    # ############################################################################################################
    def add_gradient(self, gradient: "Gradient"):
        # if gradient.param_type != ParameterType.GRADIENT:
        #     raise ValueError("Cannot add non-gradient parameter to gradients list.")

        if gradient.from_response_id is None:
            raise ValueError("Gradient must have a from_response_id.")

        start_order = len(self.gradients)
        gradient.order = start_order

        self.gradients.add(gradient)
        # sort the gradients by the data_id, response_component_id, and score
        self.sort_gradients()

    def reset_gradients(self):
        self.gradients = set()

    def get_gradients_names(self) -> str:
        names = [g.name for g in self.gradients]
        names = ", ".join(names)
        return names

    def get_prompt_data(self) -> str:
        return self.data_in_prompt(self)

    def get_gradients_str(self) -> str:
        if not self.gradients:
            return ""

        gradients_str = ""
        for i, g in enumerate(self.gradients):
            gradients_str += f"{i}. {g.data}\n"

        return gradients_str

    def get_gradient_and_context_text(self, skip_correct_sample: bool = False) -> str:
        """Aggregates and returns:
        1. the gradients
        2. the context text for which the gradients are computed

        Sort the gradients from the lowest score to the highest score.
        Highlight the gradients with the lowest score to the optimizer.
        """
        from adalflow.core.prompt_builder import Prompt

        if not self.gradients:
            return ""

        # print the score for the sorted gradients
        lowest_score_gradients = []
        for i, g in enumerate(self.gradients):
            if skip_correct_sample:
                if g.score > 0.5:
                    continue
            lowest_score_gradients.append(g)

        gradient_context_combined_str = ""
        if lowest_score_gradients and len(lowest_score_gradients) > 0:

            # group gradients by data_id and calculate average scores
            grouped_gradients = defaultdict(
                lambda: {"gradients": [], "score_sum": 0, "count": 0}
            )
            for g in lowest_score_gradients:
                group = grouped_gradients[g.data_id]
                group["gradients"].append(
                    {
                        "gradient": g.data,
                        "context": g.context.input_output,
                        "score": g.score,
                    }
                )
                group["score_sum"] += g.score if g.score is not None else 0
                group["count"] += 1

            # Calculate average scores and sort groups
            grouped_list = []
            for data_id, group in grouped_gradients.items():
                average_score = (
                    group["score_sum"] / group["count"] if group["count"] > 0 else 0
                )
                grouped_list.append(
                    {
                        "data_id": data_id,
                        "average_score": average_score,
                        "gradients": group["gradients"],
                    }
                )
            sorted_groups = sorted(grouped_list, key=lambda x: x["average_score"])

            gradient_context_combined_str = Prompt(
                template=COMBINED_GRADIENTS_TEMPLATE,
                prompt_kwargs={"combined_gradients": sorted_groups},
            )().strip()

        # get component id: gradient
        component_id_to_gradient: Dict[str, Gradient] = {}
        for g in lowest_score_gradients:
            component_id_to_gradient[g.from_response_component_id] = g

        componend_id_to_schema: Dict[str, str] = {}
        for id, g in component_id_to_gradient.items():
            componend_id_to_schema[id] = g.context.to_yaml(exclude={"input_output"})

        # if there are multiple successors, there will be multiple component schemas

        return gradient_context_combined_str

    def get_gradients_component_schema(self, skip_correct_sample: bool = False) -> str:
        """Aggregates and returns:
        1. the gradients
        2. the context text for which the gradients are computed

        Sort the gradients from the lowest score to the highest score.
        Highlight the gradients with the lowest score to the optimizer.
        """
        from adalflow.core.prompt_builder import Prompt

        # print(
        #     f"len of gradients: {len(self.gradients)}, scores: {[g._score for g in self.gradients]} for {self.name}"
        # )

        if not self.gradients:
            return ""

        # sore gradients by the _score from low to high
        # self.gradients = sorted(
        #     self.gradients, key=lambda x: x.score if x.score is not None else 1
        # )
        # print the score for the sorted gradients
        lowest_score_gradients = []
        for i, g in enumerate(self.gradients):
            if skip_correct_sample:
                if g.score > 0.5:
                    continue
            lowest_score_gradients.append(g)

        # Group gradients by `data_id` and calculate average scores
        grouped_gradients = defaultdict(
            lambda: {"gradients": [], "score_sum": 0, "count": 0}
        )
        for g in lowest_score_gradients:
            group = grouped_gradients[g.data_id]
            group["gradients"].append(
                {
                    "gradient": g.data,
                    "context": g.context.input_output,
                    "score": g.score,
                }
            )
            group["score_sum"] += g.score if g.score is not None else 0
            group["count"] += 1

        # Calculate average scores and sort groups
        grouped_list = []
        for data_id, group in grouped_gradients.items():
            average_score = (
                group["score_sum"] / group["count"] if group["count"] > 0 else 0
            )
            grouped_list.append(
                {
                    "data_id": data_id,
                    "average_score": average_score,
                    "gradients": group["gradients"],
                }
            )
        sorted_groups = sorted(grouped_list, key=lambda x: x["average_score"])

        # get component id: gradient
        component_id_to_gradient: Dict[str, Gradient] = {}
        for g in lowest_score_gradients:
            component_id_to_gradient[g.from_response_component_id] = g

        componend_id_to_schema: Dict[str, str] = {}
        for id, g in component_id_to_gradient.items():
            componend_id_to_schema[id] = g.context.to_yaml(exclude=["input_output"])

        # parse the gradients and context.
        gradients_and_context: List[Dict[str, Any]] = (
            []
        )  # {gradient: data, context: GradientContext.input_output}
        for g in lowest_score_gradients:
            gradients_and_context.append(
                {
                    "data_id": g.data_id,
                    "gradient": g.data,
                    "context": g.context.input_output,
                    "score": g.score,
                }
            )

        gradient_context_combined_str = Prompt(
            template=COMBINED_GRADIENTS_TEMPLATE,
            prompt_kwargs={
                "combined_gradients": sorted_groups,
                "component_schema": componend_id_to_schema,
            },
        )().strip()

        # if there are multiple successors, there will be multiple component schemas

        return gradient_context_combined_str

    def merge_gradients_for_cycle_components(self):
        """Merge data_id, from_response_component_id into the same gradient"""

    def sort_gradients(self):
        """With rules mentioned in Graient class, we will track the gradients by data_id, then response_component_id, then score"""

        self.gradients = sorted(
            self.gradients,
            key=lambda x: (
                x.data_id,
                x.from_response_component_id,
                -x.order if x.order is not None else 0,
                x.from_response_id,
                x.score,
            ),
        )
        # make it a set again
        self.gradients = set(self.gradients)

    ############################################################################################################
    # Setters and getters
    ############################################################################################################

    def set_predecessors(self, predecessors: List["Parameter"] = None):
        if predecessors is None:
            self.predecessors = set()
        else:
            for pred in self.predecessors:
                if not isinstance(pred, Parameter):
                    raise TypeError(
                        f"Expected a list of Parameter instances, got {type(pred).__name__}, {pred}"
                    )
            self.predecessors = set(predecessors)

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn

    def get_param_info(self):
        """Used to represent the parameter in the prompt."""
        return {
            "name": self.name,
            "role_desc": self.role_desc,
            "prompt_data": self.get_prompt_data(),  # default to use all data
            "param_type": self.param_type,
            "requires_opt": self.requires_opt,
            "eval_input": self.eval_input,  # for output passing to the eval_fn
        }

    def set_peers(self, peers: List["Parameter"] = None):
        if peers is None:
            self.peers = set()
        else:
            for peer in peers:
                if not isinstance(peer, Parameter):
                    raise TypeError(
                        f"Expected a list of Parameter instances, got {type(peer).__name__}, {peer}"
                    )
            self.peers = set(peers)

    #############################################################################################################
    # Trace the tgd optimizer data
    ############################################################################################################
    def trace_optimizer(self, api_kwargs: Dict[str, Any], response: "TGDData"):
        r"""Trace the inputs and output of a TGD optimizer."""
        from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizerTrace

        self.tgd_optimizer_trace = TGDOptimizerTrace(
            api_kwargs=api_kwargs, output=response
        )

    def set_eval_fn_input(self, eval_input: object):
        r"""Set the input for the eval_fn."""
        self.eval_input = eval_input

    ###################################################################################################################
    #   Used for demo optimizer (forward and backward pass) to accumlate the traces on both score and DynamicDataClass
    ###################################################################################################################
    def set_score(self, score: float):
        r"""Set the score of the parameter in the backward pass
        For intermediate nodes, there is only one score per each eval fn behind this node.
        For leaf nodes, like DEMO or PROMPT, it will have [batch_size] of scores.

        But this score is only used to relay the score to the demo parametr.
        """
        score = float(score)
        if not isinstance(score, float):
            raise ValueError(
                f"score is not float, but {type(score)}, parameter name: {self.name}"
            )
        self.score = score

    def add_dataclass_to_trace(self, trace: DataClass, is_teacher: bool = True):
        r"""Called by the generator.forward to add a trace to the parameter.

        It is important to allow updating to the trace, as this will give different sampling weight.
        If the score increases as the training going on, it will become less likely to be sampled,
        allowing the samples to be more diverse. Or else, it will keep sampling failed examples.
        """
        target = self._traces if is_teacher else self._student_traces
        if not hasattr(trace, "id"):
            raise ValueError("Trace must have an id attribute.")
        if trace.id in target:
            print(f"Trace with id {trace.id} already exists. Updating the trace.")
        target[trace.id] = trace

    def add_score_to_trace(self, trace_id: str, score: float, is_teacher: bool = True):
        r"""Called by the generator.backward to add the eval score to the trace."""

        target = self._traces if is_teacher else self._student_traces

        if trace_id not in target:
            raise ValueError(
                f"Trace with id {trace_id} does not exist. Current traces: {target.keys()}"
            )

        setattr(target[trace_id], "score", score)
        log.debug(f"Adding score {score} to trace {trace_id}")

    ############################################################################################################
    #   Used for optimizer to propose new data
    ############################################################################################################
    def propose_data(self, data: T, demos: Optional[List[DataClass]] = None):
        r"""Used by optimizer to put the new data, and save the previous data in case of revert."""
        if self.proposing:
            raise ValueError("Cannot propose a new data when it is already proposing.")
        self.previous_data = self.data
        self.data = data
        self.proposing = True
        if demos is not None:
            self._previous_demos = self._demos
            self._demos = demos

    def revert_data(self, include_demos: bool = False):
        r"""Revert the data to the previous data."""
        if not self.proposing:
            raise ValueError("Cannot revert data without proposing first.")

        self.data = self.previous_data
        self.previous_data = None
        self.proposing = False

        # cant reset gradients yet for the loss
        if include_demos:
            self._demos = self._previous_demos
            self._previous_demos = []

    def step_data(self, include_demos: bool = False):
        r"""Use PyTorch's optimizer syntax to finalize the update of the data."""
        if not self.proposing:
            raise ValueError("Cannot set data without proposing first.")

        self.previous_data = None
        self.proposing = False

        if include_demos:
            self._previous_demos = []

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

    # TODO: dont use short value
    def get_short_value(self, n_words_offset: int = 10) -> str:
        """
        Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
        This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
        :param n_words_offset: The number of words to show from the beginning and the end of the value.
        :type n_words_offset: int
        """
        # 1. ensure the data is a string
        # data = self.data
        data = self.get_prompt_data()
        if not isinstance(self.data, str):
            data = str(self.data)
        words = data.split(" ")
        if len(words) <= 2 * n_words_offset:
            return data
        short_value = (
            " ".join(words[:n_words_offset])
            + " (...) "
            + " ".join(words[-n_words_offset:])
        )
        return short_value

    def reset_all_gradients(self):
        """Traverse the graph and reset the gradients for all nodes."""
        nodes, _ = Parameter.trace_graph(self)
        for node in nodes:
            node.reset_gradients()

    @staticmethod
    def trace_graph(
        root: "Parameter",
    ) -> Tuple[Set["Parameter"], Set[Tuple["Parameter", "Parameter"]]]:
        nodes, edges = set(), set()

        def build_graph(node: "Parameter"):
            if node in nodes:
                return
            if node is None:
                raise ValueError("Node is None")
            nodes.add(node)
            for pred in node.predecessors:
                edges.add((pred, node))
                build_graph(pred)

        build_graph(root)
        return nodes, edges

    def backward(
        self,
    ):
        """
        Apply backward pass for for all nodes in the graph by reversing the topological order.
        """
        # engine should be the llm or customized backwards function to pass feedback

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
                log.debug(f"Skipping {node.name} as it does not require optimization")
                continue
            component_name = None
            if hasattr(node, "component_trace"):
                component_name = node.component_trace.name
            log.debug(
                f"node: {node.name}, component: {component_name}, grad_fn: {node.grad_fn}."
            )
            if node.get_grad_fn() is not None:  # gradient function takes in the engine
                log.debug(f"Calling gradient function for {node.name}")
                node.grad_fn()

    @staticmethod
    def generate_node_html(node: "Parameter", output_dir="node_pages"):
        """Generate an HTML page for a specific node."""
        import json

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{output_dir}/{node.name}.html"

        # Gather gradients as JSON objects
        gradients = []
        for i, g in enumerate(node.gradients):
            gradient = g.to_json_obj()
            for k, v in gradient.items():
                if isinstance(v, str):
                    gradient[k] = v.replace("<", "&lt;").replace(">", "&gt;")
            gradients.append(gradient)

        data_json = None
        node_data_type = str(type(node.data)).replace("<", "&lt;").replace(">", "&gt;")
        if isinstance(node.data, dict):
            data_json = data_json
        elif isinstance(node.data, DataClass):
            try:
                data_json = node.data.to_json_obj()
            except Exception:

                data_json = str(node.data)

        else:
            data_json = str(node.data)
            data_json = {"data": data_json}

        gradients_json = json.dumps(gradients, indent=4, ensure_ascii=False)

        optimizer_trace = None
        if node.tgd_optimizer_trace:
            optimizer_trace = node.tgd_optimizer_trace.to_json_obj()
            optimizer_trace = json.dumps(optimizer_trace, indent=4, ensure_ascii=False)

        with open(filename, "w") as file:
            file.write(
                f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{node.name}</title>
                <style>
                    pre {{
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                </style>
            </head>
            <body>
                <h1>Details for Node: {node.name}</h1>
                <p><b>ID:</b> {node.id}</p>
                <p><b>Role:</b> {node.role_desc}</p>
                <p><b>DataType:</b> {node_data_type}</p>
                <pre><b>Data:</b> \n{json.dumps(data_json, indent=4)}</pre>
                <p><b>Data ID:</b> {node.data_id}</p>
                <p><b>Previous Value:</b> {node.previous_data}</p>
                <p><b>Requires Optimization:</b> {node.requires_opt}</p>
                <p><b>Type:</b> {node.param_type.value} ({node.param_type.description})</p>
                <pre><b>Gradients:</b>\n{gradients_json}</pre>
                <pre><b>TGD Optimizer Trace:</b>\n{optimizer_trace}</pre>

            </body>
            </html>
            """
            )
        print(f"Generated HTML for node: {node.name} at {filename}")

    def draw_interactive_html_graph(
        self,
        filepath: Optional[str] = None,
        nodes: List["Parameter"] = None,
        edges: List[Tuple["Parameter", "Parameter"]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an interactive graph with pyvis and save as an HTML file.

        Args:
            nodes (list): A list of Parameter objects.
            edges (list): A list of edges as tuples (source, target).
            filepath (str, optional): Path to save the graph file. Defaults to None.

        Returns:
            dict: A dictionary containing the graph file path.
        """
        from jinja2 import Template
        from pyvis.network import Network

        output_file = "interactive_graph.html"
        filepath = filepath or "output"
        os.makedirs(filepath, exist_ok=True)
        final_file = os.path.join(filepath, output_file)

        net = Network(height="750px", width="100%", directed=True)

        node_colors = {
            ParameterType.PROMPT: "lightblue",
            ParameterType.DEMOS: "orange",
            ParameterType.INPUT: "gray",
            ParameterType.OUTPUT: "green",
            ParameterType.GENERATOR_OUTPUT: "purple",
            ParameterType.RETRIEVER_OUTPUT: "red",
            ParameterType.LOSS_OUTPUT: "pink",
            ParameterType.SUM_OUTPUT: "blue",
        }

        # Add nodes to the graph
        node_ids = set()
        for node in nodes:
            self.generate_node_html(node, output_dir=filepath)

            node_id = node.id
            node_show_name = node.name.replace(f"_{node_id}", "")
            label = (
                f"""<div style="max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background: white; position: relative; font-family: Arial, sans-serif;">"""
                f"<b>Name:</b> {node_show_name}<br>"
                f"<b>Role:</b> {node.role_desc.capitalize()}<br>"
                f"<b>Value:</b> {node.data}<br>"
                f"<b>Data ID:</b> {node.data_id}<br>"
            )
            if node.proposing:
                label += "<b>Proposing:</b> Yes<br>"
                label += f"<b>Previous Value:</b> {node.previous_data}<br>"
            label += f"<b>Requires Optimization:</b> {node.requires_opt}<br>"
            if node.param_type:
                label += f"<b>Type:</b> {node.param_type.value}<br>"

            net.add_node(
                n_id=node.id,
                label=node_show_name,
                title=label,
                color=node_colors.get(node.param_type, "gray"),
                url=f"./{node.name}.html",  # Relative path
            )
            node_ids.add(node.id)

        # Add edges to the graph
        for source, target in edges:
            if source.id in node_ids and target.id in node_ids:
                net.add_edge(source.id, target.id)
            else:
                print(
                    f"Skipping edge from {source.name} to {target.name} as one of the nodes does not exist."
                )

        net.toggle_physics(True)
        net.template = Template(
            """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis-network.min.css" rel="stylesheet" />
        <style>
            #tooltip {
                display: none;
                position: absolute;
                max-width: 300px;
                border: 1px solid #ccc;
                padding: 10px;
                background: white;
                z-index: 1000;
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
            #tooltip button {
                display: block;
                margin-top: 10px;
            }
             /* Simple styling for the legend */
            #legend {
                margin-top: 20px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            .legend-item {
                margin-bottom: 5px;
            }
            .legend-color-box {
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
                border: 1px solid #000;
                vertical-align: middle;
            }
        </style>
    </head>
    <body>
        <div id="tooltip">
            <div id="tooltip-content"></div>
            <button onclick="document.getElementById('tooltip').style.display='none'">Close</button>
        </div>
         <!-- Legend Section -->
        <div id="legend">
            <strong>Legend:</strong>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: lightblue;"></span>PROMPT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: orange;"></span>DEMOS
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: gray;"></span>INPUT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: green;"></span>OUTPUT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: purple;"></span>GENERATOR_OUTPUT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: red;"></span>RETRIEVER_OUTPUT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: pink;"></span>LOSS_OUTPUT
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background-color: blue;"></span>SUM_OUTPUT
            </div>
        </div>
        <!-- End Legend Section -->
        <div id="mynetwork" style="height: {{ height }};"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({{ nodes | safe }});
            var edges = new vis.DataSet({{ edges | safe }});
            var container = document.getElementById('mynetwork');
            var data = { nodes: nodes, edges: edges };
            var options = {{ options | safe }};
            var network = new vis.Network(container, data, options);

            // Handle node click to open a link
            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = nodes.get(nodeId);
                    if (node.url) {
                        window.open(node.url, '_blank');
                    }
                }
            });
        </script>
    </body>
    </html>
    """
        )

        net.show(final_file)
        print(f"Interactive graph saved to {final_file}")

        return {"graph_path": final_file}

    @staticmethod
    def wrap_and_escape(text, width=40):
        r"""Wrap text to the specified width, considering HTML breaks, and escape special characters."""
        try:
            import textwrap
        except ImportError as e:
            raise ImportError(
                "Please install textwrap using 'pip install textwrap' to use this feature"
            ) from e

        def wrap_text(text, width):
            """Wrap text to the specified width, considering HTML breaks."""
            lines = textwrap.wrap(
                text, width, break_long_words=False, replace_whitespace=False
            )
            return "<br/>".join(lines)

        if not isinstance(text, str):
            text = str(text)
        text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
            .replace(
                "\n", "<br/>"
            )  # Convert newlines to HTML line breaks if using HTML labels
        )
        return wrap_text(text, width)

    def draw_graph(
        self,
        add_grads: bool = True,
        full_trace: bool = False,
        format: Literal["png", "svg"] = "png",
        rankdir: Literal["LR", "TB"] = "TB",
        filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draw the graph of the parameter and its gradients.

        Args:
            add_grads (bool, optional): Whether to add gradients to the graph. Defaults to True.
            format (str, optional): The format of the output file. Defaults to "png".
            rankdir (str, optional): The direction of the graph. Defaults to "TB".
            filepath (str, optional): The path to save the graph. Defaults to None.
            full_trace (bool, optional): Whether to include more detailed trace such as api_kwargs. Defaults to False.
        """
        from adalflow.utils import save_json
        from adalflow.utils.global_config import get_adalflow_default_root_path

        try:
            from graphviz import Digraph

        except ImportError as e:
            raise ImportError(
                "Please install graphviz using 'pip install graphviz' to use this feature"
            ) from e

        assert rankdir in ["LR", "TB"]

        root_path = get_adalflow_default_root_path()

        filename = f"trace_graph_{self.name}_id_{self.id}"
        filepath = (
            os.path.join(filepath, filename)
            if filepath
            else os.path.join(root_path, "graphs", filename)
        )
        # final_path = f"{filepath}.{format}"
        print(f"Saving graph to {filepath}.{format}")

        nodes, edges = self.trace_graph(self)
        dot = Digraph(format=format, graph_attr={"rankdir": rankdir, "dpi": "300"})
        node_names = set()
        for n in nodes:
            label_color = "darkblue"

            node_label = (
                f"<table border='0' cellborder='1' cellspacing='0'>"
                f"<tr><td><b><font color='{label_color}'>Name: </font></b></td><td>{self.wrap_and_escape(n.id)}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Name: </font></b></td><td>{self.wrap_and_escape(n.name)}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Role: </font></b></td><td>{self.wrap_and_escape(n.role_desc.capitalize())}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Value: </font></b></td><td>{self.wrap_and_escape(n.data)}</td></tr>"
            )
            if n.data_id is not None:
                node_label += f"<tr><td><b><font color='{label_color}'>Data ID: </font></b></td><td>{self.wrap_and_escape(n.data_id)}</td></tr>"
            if n.proposing:
                node_label += f"<tr><td><b><font color='{label_color}'>Proposing</font></b></td><td>{{'Yes'}}</td></tr>"
                node_label += f"<tr><td><b><font color='{label_color}'>Previous Value: </font></b></td><td>{self.wrap_and_escape(n.previous_data)}</td></tr>"
            if n.requires_opt:
                node_label += f"<tr><td><b><font color='{label_color}'>Requires Optimization: </font ></b></td><td>{{'Yes'}}</td></tr>"
            if n.param_type:
                node_label += f"<tr><td><b><font color='{label_color}'>Type: </font></b></td><td>{self.wrap_and_escape(n.param_type.name)}</td></tr>"
            if (
                full_trace
                and hasattr(n, "component_trace")
                and n.component_trace.api_kwargs is not None
            ):
                node_label += f"<tr><td><b><font color='{label_color}'> API kwargs: </font></b></td><td>{self.wrap_and_escape(str(n.component_trace.api_kwargs))}</td></tr>"

            # show the score for intermediate nodes
            if n.score is not None and len(n.predecessors) > 0:
                node_label += f"<tr><td><b><font color='{label_color}'>Score: </font></b></td><td>{str(n.score)}</td></tr>"
            if add_grads:
                node_label += f"<tr><td><b><font color='{label_color}'>Gradients: </font></b></td><td>{self.wrap_and_escape(n.get_gradients_names())}</td></tr>"
                # add a list of each gradient with short value
                # combine the gradients and context
                # combined_gradients_contexts = zip(
                #     n.gradients, [n.gradients_context[g] for g in n.gradients]
                # )
                # if "output" in n.name:
                for g in n.gradients:
                    gradient_context = g.context
                    log.info(f"Gradient context display: {gradient_context}")
                    log.info(f"data: {g.data}")
                    node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.name} Feedback: </font></b></td><td>{self.wrap_and_escape(g.data)}</td></tr>"
                    # if gradient_context != "":
                    #     node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.name} Context: </font></b></td><td>{wrap_and_escape(gradient_context)}</td></tr>"
                    # if g.prompt:
                    #     node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.name} Prompt: </font></b></td><td>{wrap_and_escape(g.prompt)}</td></tr>"
            if len(n._traces.values()) > 0:
                node_label += f"<tr><td><b><font color='{label_color}'>Traces: keys: </font></b></td><td>{self.wrap_and_escape(str(n._traces.keys()))}</td></tr>"
                node_label += f"<tr><td><b><font color='{label_color}'>Traces: values: </font></b></td><td>{self.wrap_and_escape(str(n._traces.values()))}</td></tr>"
            if n.tgd_optimizer_trace is not None:
                node_label += f"<tr><td><b><font color='{label_color}'>TGD Optimizer Trace: </font></b></td><td>{self.wrap_and_escape(str(n.tgd_optimizer_trace))}</td></tr>"

            # show component trace, id and name
            if hasattr(n, "component_trace") and n.component_trace.id is not None:
                node_label += f"<tr><td><b><font color='{label_color}'>Component Trace ID: </font></b></td><td>{self.wrap_and_escape(str(n.component_trace.id))}</td></tr>"
            if hasattr(n, "component_trace") and n.component_trace.name is not None:
                node_label += f"<tr><td><b><font color='{label_color}'>Component Trace Name: </font></b></td><td>{self.wrap_and_escape(str(n.component_trace.name))}</td></tr>"

            node_label += "</table>"
            # check if the name exists in dot
            if n.name in node_names:
                n.name = f"{n.name}_{n.id}"
            node_names.add(n.name)
            dot.node(
                name=n.name,
                label=f"<{node_label}>",
                shape="plaintext",
            )
            # writer.add_text(n.name, str(n.to_dict()))
            log.info(f"Node: {n.name}, {n.to_dict()}")
            # track gradients
            for g in n.gradients:

                log.info(f"Gradient: {g.name}, {g.to_dict()}")
                log.info(f"Gradient prompt: {g.prompt}")
        for n1, n2 in edges:
            dot.edge(n1.name, n2.name)

        # dot.render(filepath, format=format, cleanup=True)

        save_json(self.to_dict(), f"{filepath}_root.json")

        # draw interactive graph
        graph_file: Dict[str, str] = self.draw_interactive_html_graph(
            filepath=filepath, nodes=nodes, edges=edges
        )
        output = {
            # "graph_path": final_path,
            "root_path": f"{filepath}_root.json",
            "interactive_html_graph": graph_file["graph_path"],
        }
        print(f"Graph saved as {filepath}.{format}")
        return output

    def draw_output_subgraph(
        self,
        add_grads: bool = True,
        format: str = "png",
        rankdir: str = "TB",
        filepath: str = None,
    ) -> Dict:
        """
        Build and visualize a subgraph containing only OUTPUT parameters.

        Args:
            add_grads (bool): Whether to include gradient edges.
            format (str): Format for output (e.g., png, svg).
            rankdir (str): Graph layout direction ("LR" or "TB").
            filepath (str): Path to save the graph.
        """

        assert rankdir in ["LR", "TB"]
        from adalflow.utils.global_config import get_adalflow_default_root_path

        try:
            from graphviz import Digraph

        except ImportError as e:
            raise ImportError(
                "Please install graphviz using 'pip install graphviz' to use this feature"
            ) from e

        root_path = get_adalflow_default_root_path()

        filename = f"trace_component_output_graph_{self.name}_id_{self.id}.{format}"
        filepath = (
            os.path.join(filepath, filename)
            if filepath
            else os.path.join(root_path, "graphs", filename)
        )

        # Step 1: Collect OUTPUT nodes and edges
        nodes, edges = self._collect_output_subgraph()

        # Step 2: Render using Graphviz
        print(f"Saving OUTPUT subgraph to {filepath}")

        dot = Digraph(format=format, graph_attr={"rankdir": rankdir})
        node_ids = set()

        for node in nodes:

            node_label = f"""
            <table border="0" cellborder="1" cellspacing="0">
                <tr><td><b>Name:</b></td><td>{self.wrap_and_escape(node.name)}</td></tr>
                <tr><td><b>Type:</b></td><td>{self.wrap_and_escape(node.param_type.name)}</td></tr>
                <tr><td><b>Value:</b></td><td>{self.wrap_and_escape(node.get_short_value())}</td></tr>"""
            # add the component trace id and name
            if hasattr(node, "component_trace") and node.component_trace.id is not None:
                escaped_ct_id = html.escape(str(node.component_trace.id))
                node_label += f"<tr><td><b>Component Trace ID:</b></td><td>{escaped_ct_id}</td></tr>"
            if (
                hasattr(node, "component_trace")
                and node.component_trace.name is not None
            ):
                escaped_ct_name = html.escape(str(node.component_trace.name))
                node_label += f"<tr><td><b>Component Trace Name:</b></td><td>{escaped_ct_name}</td></tr>"

            node_label += "</table>"
            dot.node(
                name=node.id,
                label=f"<{node_label}>",
                shape="plaintext",
                color="lightblue" if node.requires_opt else "gray",
            )
            node_ids.add(node.id)

        for source, target in edges:
            if source.id in node_ids and target.id in node_ids:
                dot.edge(source.id, target.id)

        # Step 3: Save and render
        dot.render(filepath, cleanup=True)
        print(f"Graph saved as {filepath}")
        return {"output_subgraph": filepath}

    def draw_component_subgraph(
        self,
        format: str = "png",
        rankdir: str = "TB",
        filepath: str = None,
    ):
        """
        Build and visualize a subgraph containing only OUTPUT parameters.

        Args:
            format (str): Format for output (e.g., png, svg).
            rankdir (str): Graph layout direction ("LR" or "TB").
            filepath (str): Path to save the graph.
        """
        assert rankdir in ["LR", "TB"]
        from adalflow.utils.global_config import get_adalflow_default_root_path

        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "Please install graphviz using 'pip install graphviz' to use this feature"
            ) from e

        # Step 1: Collect OUTPUT nodes and edges
        component_nodes, edges, component_nodes_orders = (
            self._collect_component_subgraph()
        )
        root_path = get_adalflow_default_root_path()

        # Step 2: Setup graph rendering
        filename = f"output_component_{self.name}_{self.id}.{format}"
        filepath = filepath or f"./{filename}"

        filepath = (
            os.path.join(filepath, filename)
            if filepath
            else os.path.join(root_path, "graphs", filename)
        )
        print(f"Saving OUTPUT subgraph to {filepath}")

        dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

        # Add nodes
        for node in component_nodes:
            node_label = """
            <table border="0" cellborder="1" cellspacing="0">"""

            if node.name:
                node_label += """<tr><td><b>Name:</b></td><td>{node.name}</td></tr>"""
            if node.type:
                node_label += """<tr><td><b>TYPE:</b></td><td>{node.type}</td></tr>"""

            # add the list of orders
            if node.id in component_nodes_orders:
                node_label += f"<tr><td><b>Order:</b></td><td>{component_nodes_orders[node.id]}</td></tr>"
            node_label += "</table>"
            dot.node(
                name=node.id if node.id else "id missing",
                label=f"<{node_label}>",
                shape="plaintext",
                color="lightblue",
            )

        # Add edges with order labels
        for source_id, target_id, edge_order in edges:
            dot.edge(source_id, target_id)  # , label=str(edge_order), color="black")

        # Step 3: Save and render
        dot.render(filepath, cleanup=True)
        print(f"Graph saved as {filepath}")
        return {"component_graph": f"{filepath}"}

    def _collect_output_subgraph(
        self,
    ) -> Tuple[Set["Parameter"], List[Tuple["Parameter", "Parameter"]]]:
        """
        Collect nodes of type OUTPUT and their relationships.

        Returns:
            nodes (Set[Parameter]): Set of OUTPUT nodes.
            edges (List[Tuple[Parameter, Parameter]]): Edges between OUTPUT nodes.
        """
        output_nodes = set()
        edges = []

        visited = set()  # check component_trace.id and name

        def traverse(node: "Parameter"):
            if node in visited:
                return
            visited.add(node)

            # Add OUTPUT nodes to the set
            if (
                node.param_type == ParameterType.OUTPUT
                or "OUTPUT" in node.param_type.name
            ):
                output_nodes.add(node)

            # Traverse predecessors and add edges
            for pred in node.predecessors:
                if (
                    pred.param_type == ParameterType.OUTPUT
                    or "OUTPUT" in pred.param_type.name
                ):
                    edges.append((pred, node))
                traverse(pred)

        traverse(self)
        return output_nodes, edges

    def _collect_component_subgraph(
        self,
    ) -> Tuple[Set[ComponentNode], List[Tuple[str, str]]]:
        """
        Collect OUTPUT nodes and their relationships as ComponentNodes.

        Returns:
            component_nodes (Set[ComponentNode]): Set of component nodes (id and name only).
            edges (List[Tuple[str, str]]): Edges between component IDs.
        """
        component_nodes = set()  # To store component nodes as ComponentNode
        component_nodes_orders: Dict[str, List[int]] = (
            {}
        )  # To store component nodes order
        edges = []  # To store edges between component IDs

        visited = set()  # Track visited parameters to avoid cycles
        edge_counter = [0]  # Mutable counter for edge order tracking

        def traverse(node: "Parameter"):
            if node in visited:
                return
            visited.add(node)

            # Check if node is of OUTPUT type
            if (
                node.param_type == ParameterType.OUTPUT
                or "OUTPUT" in node.param_type.name
            ):
                component_id = node.component_trace.id or f"unknown_id_{uuid.uuid4()}"
                component_name = node.component_trace.name or "Unknown Component"

                # Create a ComponentNode and add to the set
                component_node = ComponentNode(id=component_id, name=component_name)
                component_nodes.add(component_node)

                # Traverse predecessors and add edges
                for pred in node.predecessors:
                    # if pred.param_type != ParameterType.OUTPUT:
                    #     continue
                    pred_id = f"unknown_id_{uuid.uuid4()}"
                    pred_name = "Unknown Component"

                    if hasattr(pred, "component_trace") and pred.component_trace.id:
                        pred_id = pred.component_trace.id
                        pred_name = pred.component_trace.name

                    # Add edge if predecessor is also of OUTPUT type
                    if (
                        pred.param_type == ParameterType.OUTPUT
                        or "OUTPUT" in pred.param_type.name
                    ):
                        edges.append((pred_id, component_id, edge_counter[0]))
                        component_nodes.add(ComponentNode(id=pred_id, name=pred_name))
                        edge_counter[0] += 1

                    if pred.param_type == ParameterType.INPUT:
                        pred_id = pred.id
                        pred_name = pred.name
                        pred_node = ComponentNode(
                            id=pred_id, name=pred_name, type="INPUT"
                        )
                        component_nodes.add(pred_node)
                        # add an edge from input to the first output
                        edges.append((pred_id, component_id, edge_counter[0]))
                        edge_counter[0] += 1

                    traverse(pred)

        # Start traversal from the current parameter
        traverse(self)
        # Reverse the edge order
        # total_edges = len(edges)
        # edges = [
        #     (source, target, (total_edges - 1) - edge_number)
        #     for idx, (source, target, edge_number) in enumerate(edges)
        # ]

        return component_nodes, edges, component_nodes_orders

    def to_dict(self):
        return {
            "name": self.name,
            "id": self.id,
            "role_desc": self.role_desc,
            "data": str(self.data),
            "requires_opt": self.requires_opt,
            "param_type": str(self.param_type),
            # others
            "predecessors": [pred.to_dict() for pred in self.predecessors],
            "gradients": [grad.to_dict() for grad in self.gradients],
            "previous_data": self.previous_data,
            "grad_fn": str(
                self.grad_fn
            ),  # Simplify for serialization, modify as needed
            "score": self.score,
            "traces": {k: v.to_dict() for k, v in self._traces.items()},
            # demos
            "demos": [d.to_dict() for d in self._demos],
        }

    @classmethod
    def from_dict(cls, data: dict):
        predecessors = [cls.from_dict(pred) for pred in data["predecessors"]]
        param = cls(
            name=data["name"],
            role_desc=data["role_desc"],
            data=data["data"],
            requires_opt=data["requires_opt"],
            param_type=ParameterType(data["param_type"]),
            # others
            predecessors=predecessors,
            gradients=[cls.from_dict(grad) for grad in data["gradients"]],
            previous_data=data["previous_data"],
            score=data["score"],
            # demos
            demos=[DataClass.from_dict(d) for d in data["demos"]],
        )
        # Reconstruct gradients_context from the list of tuples
        param._traces = {k: DataClass.from_dict(v) for k, v in data["traces"].items()}
        return param

    # TODO: very hard to read directly, need to simplify and let users use to_dict for better readability
    def __repr__(self):
        return f"Parameter(name={self.name}, requires_opt={self.requires_opt}, param_type={self.param_type}, role_desc={self.role_desc}, data={self.data}, predecessors={self.predecessors}, gradients={self.gradients},\
            traces={self._traces})"


# TODO: separate the Parameter class into different classes and each class will have its own methods instead of all in one class
# class InputParameter(Parameter):
#     """One of the simplest types of parameters, representing an input to the system.
#     Input parameter will not be trainable, but serves a tracing purpose in the computation graph.
#     """

#     def __init__(
#         self,
#         name: str,
#         role_desc: str,
#         data: Any,
#         requires_opt: bool = False,
#         param_type: ParameterType = ParameterType.INPUT,
#     ):
#         super().__init__(
#             name=name,
#             role_desc=role_desc,
#             data=data,
#             requires_opt=requires_opt,
#             param_type=param_type,
#         )


# class HyperParameter(Parameter):
#     """One of the simplest types of parameters, representing a hyperparameter to the system."""

#     def __init__(
#         self,
#         name: str,
#         role_desc: str,
#         data: Any,
#         requires_opt: bool = False,
#         param_type: ParameterType = ParameterType.HYPERPARAM,
#     ):
#         super().__init__(
#             name=name,
#             role_desc=role_desc,
#             data=data,
#             requires_opt=requires_opt,
#             param_type=param_type,
#         )


# class PromptParameter(Parameter):

#     def __init__(
#         self,
#         name: str,
#         role_desc: str,
#         data: Any,
#         requires_opt: bool = True,
#         param_type: ParameterType = ParameterType.PROMPT,
#     ):
#         super().__init__(
#             name=name,
#             role_desc=role_desc,
#             data=data,
#             requires_opt=requires_opt,
#             param_type=param_type,
#         )


# class DemoParameter(Parameter):

#     def __init__(
#         self,
#         name: str,
#         role_desc: str,
#         data: Any,
#         requires_opt: bool = True,
#         param_type: ParameterType = ParameterType.DEMOS,
#     ):
#         super().__init__(
#             name=name,
#             role_desc=role_desc,
#             data=data,
#             requires_opt=requires_opt,
#             param_type=param_type,
#         )


class OutputParameter(Parameter):
    __doc__ = r"""The output parameter is the most complex type of parameter in the system.

    It will trace the predecessors, set up a grad_fn, store gradients, and trace the forward pass by tracking the component_trace.
    """
    allowed_types = {
        ParameterType.OUTPUT,
        ParameterType.LOSS_OUTPUT,
        ParameterType.GENERATOR_OUTPUT,
        ParameterType.SUM_OUTPUT,
    }
    component_trace: ComponentTrace = (
        None  # Trace of the component that produced this output
    )
    full_response: object = None  # The full response from the component

    def __init__(
        self,
        *,
        id: Optional[str] = None,  # unique id of the parameter
        data: T = None,  # for generator output, the data will be set up as raw_response
        data_id: str = None,  # for tracing the data item in the training/val/test set
        requires_opt: bool = True,
        role_desc: str = "",
        param_type: ParameterType = ParameterType.OUTPUT,
        name: str = None,  # name is used to refer to the parameter in the prompt, easier to read for humans
        instruction_to_optimizer: str = None,
        instruction_to_backward_engine: str = None,
        score: Optional[float] = None,
        eval_input: object = None,
        successor_map_fn: Optional[Dict[str, Callable]] = None,
        data_in_prompt: Optional[
            Callable
        ] = None,  # how will the data be displayed in the prompt
        full_response: Optional[Any] = None,
    ):
        super().__init__(
            id=id,
            data=data,
            data_id=data_id,
            requires_opt=requires_opt,
            role_desc=role_desc,
            param_type=param_type,
            name=name,
            instruction_to_optimizer=instruction_to_optimizer,
            instruction_to_backward_engine=instruction_to_backward_engine,
            score=score,
            eval_input=eval_input,
            successor_map_fn=successor_map_fn,
            data_in_prompt=data_in_prompt,
        )

        self.component_trace = ComponentTrace()
        self.full_response = full_response

    ############################################################################################################
    #  Trace component, include trace_forward_pass & trace_api_kwargs for now
    ############################################################################################################
    def trace_forward_pass(
        self,
        input_args: Dict[str, Any],
        full_response: object,
        id: str = None,
        name: str = None,
    ):
        r"""Trace the forward pass of the parameter. Adding the component information to the trace"""
        self.input_args = input_args
        self.full_response = full_response
        # TODO: remove the input_args and full_response to use component_trace
        self.component_trace.input_args = input_args
        self.component_trace.full_response = full_response
        self.component_trace.id = id
        self.component_trace.name = name
        # just for convenience to trace full response separately
        self.full_response = full_response

    def trace_api_kwargs(self, api_kwargs: Dict[str, Any]):
        r"""Trace the api_kwargs for components like Generator and Retriever that pass to the model client."""
        self.component_trace.api_kwargs = api_kwargs

    def to_dict(self):
        super_dict = super().to_dict()
        super_dict.update(
            {
                "component_trace": self.component_trace.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, data: dict):
        component_trace = ComponentTrace.from_dict(data["component_trace"])
        return super().from_dict(data).update({"component_trace": component_trace})

    def __repr__(self):
        super_repr = super().__repr__()
        start = super_repr.find("Parameter")
        if start == 0:
            end = start + len("Parameter")
            super_repr = super_repr[:start] + "OutputParameter" + super_repr[end:]
        return super_repr


if __name__ == "__main__":

    # test gradient hash and to_dict
    from_response = OutputParameter(
        name="p1",
        role_desc="role1",
        data=1,
    )
    from_response.component_trace = ComponentTrace(id="1")
    g1 = Gradient(
        from_response=from_response,
        to_pred=Parameter(name="p2", role_desc="role2", data=2),
        data_id="1",
    )
    g2 = Gradient(
        from_response=from_response,
        to_pred=Parameter(name="p2", role_desc="role2", data=2),
        data_id="1",
    )
    print(g1 == g2)
    print(g1.__hash__())
    print(g2.__hash__())
    print(isinstance(g1, Gradient))  # Should print True

    print(g1.to_dict())
