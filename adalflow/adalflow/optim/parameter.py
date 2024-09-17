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
)
from collections import defaultdict
import logging
from dataclasses import dataclass, field
import uuid
from adalflow.optim.types import ParameterType
from adalflow.core.base_data_class import DataClass


T = TypeVar("T")  # covariant set to False to allow for in-place updates

log = logging.getLogger(__name__)


@dataclass
class GradientContext:
    variable_desc: str = field(
        metadata={"desc": "The description of the target parameter"}
    )
    response_desc: str = field(
        metadata={"desc": "The description of the response parameter"}
    )
    context: str = field(
        metadata={
            "desc": "The context of the gradient in form of a conversation indicating \
                the relation of the current parameter to the response parameter (gradient)"
        }
    )


COMBINED_GRADIENTS_TEMPLATE = r"""
{% for g in combined_gradients %}
{% set gradient = g[0] %}
{% set gradient_context = g[1] %}

{% if gradient_context %}
{{loop.index}}.
<CONTEXT>{{gradient_context.context}}</CONTEXT>
{% endif %}

{% if gradient.data %}
  {% if gradient_context %}
{#The output is used as <{{gradient_context.response_desc}}>#}
<FEEDBACK>{{gradient.data}}</FEEDBACK>
{% else %}
<FEEDBACK>{{gradient.data}}</FEEDBACK>
{% endif %}
{% endif %}
{% endfor %}"""


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

    id: str = None  # Unique id of the parameter
    name: str = None  # Name of the parameter, easier to read for humans
    role_desc: str = ""  # Description of the role of the parameter
    data: T = None  # Data of the parameter
    param_type: ParameterType

    proposing: bool = False  # State of the parameter
    predecessors: Set["Parameter"] = set()  # Predecessors of the parameter
    peers: Set["Parameter"] = set()  # Peers of the parameter
    # TODO: input_args should be OrderedDict to keep the order of args
    input_args: Dict[str, Any] = None  # Input arguments of the GradComponent forward
    full_response: object = None  # Full response of the GradComponent output
    eval_input: object = None  # Eval input passing to the eval_fn or evaluator you use
    successor_map_fn: Dict[str, Callable] = (
        None  # Map function to get the data from the output
    )
    from_response_id: str = (
        None  # for parameterType GRADIENT, the id of the response parameter
    )
    backward_engine_disabled: bool = (
        False  # Disable the backward engine for the parameter
    )

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        data: T = None,  # for generator output, the data will be set up as raw_response
        requires_opt: bool = True,
        role_desc: str = "",
        param_type: ParameterType = ParameterType.PROMPT,
        name: str = None,  # name is used to refer to the parameter in the prompt, easier to read for humans
        gradient_prompt: str = None,
        raw_response: str = None,  # use this to track the raw response of generator instead of the data (can be parsed)
        instruction_to_optimizer: str = None,
        instruction_to_backward_engine: str = None,
        score: Optional[float] = None,
        eval_input: object = None,
        from_response_id: Optional[str] = None,
        successor_map_fn: Optional[Dict[str, Callable]] = None,
    ):
        self.id = id or str(uuid.uuid4())

        self.name = name
        self.role_desc = role_desc

        if not self.name:
            self.name = (
                self.role_desc.capitalize().replace(" ", "_")[0:10]
                if self.role_desc
                else f"param_{self.id}"
            )
        self.param_type = param_type
        self.data = data  # often string and will be used in the prompts
        self.requires_opt = requires_opt
        self.data_type = type(data)

        self.set_eval_fn_input(eval_input=data)
        self.gradients: List[Parameter] = []  # <FEEDBACK>gradient.data</FEEDBACK>
        self.gradient_prompt: str = (
            gradient_prompt  # the whole llm prompt to compute the gradient
        )
        self.gradients_context: Dict[Parameter, GradientContext] = defaultdict(
            lambda: None
        )  # input and output from an operator, each operator should have a template
        # <CONVERSATION>...</CONVERSATION>
        self.grad_fn = None

        self.previous_data = None  # used to store the previous data
        # context of the forward pass
        self.raw_response = raw_response

        self.instruction_to_optimizer: str = instruction_to_optimizer
        self.instruction_to_backward_engine: str = instruction_to_backward_engine

        # here are used for demo parameter, filled by generator.forward
        self._traces: Dict[str, DataClass] = {}  # id of the data points
        self._score: float = score  # end to end evaluation score

        self._student_traces: Dict[str, DataClass] = {}  # id
        self._demos: List[DataClass] = (
            []
        )  # used for the optimizer to save the proposed demos
        self._previous_demos: List[DataClass] = []
        self.eval_input = eval_input

        self.from_response_id = from_response_id  # for gradient parameter
        self.successor_map_fn = successor_map_fn or {}

    def map_to_successor(self, successor: object) -> T:
        """Apply the map function to the successor based on the successor's id."""
        successor_id = id(successor)
        if successor_id not in self.successor_map_fn:
            default_map_fn = lambda x: x.data  # noqa: E731
            return default_map_fn(self)

        return self.successor_map_fn[successor_id](self)

    def add_successor_map_fn(self, successor: object, map_fn: Callable):
        """Add or update a map function for a specific successor using its id."""
        self.successor_map_fn[id(successor)] = map_fn

    def check_if_already_computed_gradient_respect_to(self, response_id: str) -> bool:
        from_response_ids = [g.from_response_id for g in self.gradients]
        return response_id in from_response_ids

    def add_gradient(self, gradient: "Parameter"):
        if gradient.param_type != ParameterType.GRADIENT:
            raise ValueError("Cannot add non-gradient parameter to gradients list.")

        if gradient.from_response_id is None:
            raise ValueError("Gradient must have a from_response_id.")

        self.gradients.append(gradient)

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
        return {
            "name": self.name,
            "role_desc": self.role_desc,
            "data": self.data,
            "param_type": self.param_type,
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

    def trace_forward_pass(self, input_args: Dict[str, Any], full_response: object):
        r"""Trace the forward pass of the parameter."""
        self.input_args = input_args
        self.full_response = full_response

    def set_eval_fn_input(self, eval_input: object):
        r"""Set the input for the eval_fn."""
        self.eval_input = eval_input

    def set_score(self, score: float):
        self._score = score

    def add_to_trace(self, trace: DataClass, is_teacher: bool = True):
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
        target[trace_id].score = score

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

        # reset the gradients and context
        # self.reset_gradients()
        # self.reset_gradients_context()

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

        # reset the gradients and context
        # self.reset_gradients()
        # self.reset_gradients_context()
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

    def reset_gradients(self):
        self.gradients = []

    def reset_gradients_context(self):
        self.gradients_context = defaultdict(lambda: None)

    def get_gradients_names(self) -> str:
        names = [g.name for g in self.gradients]
        names = ", ".join(names)
        return names

    def get_gradient_and_context_text(self) -> str:
        """Aggregates and returns:
        1. the gradients
        2. the context text for which the gradients are computed
        """
        from adalflow.core.prompt_builder import Prompt

        # print(
        #     f"len of gradients: {len(self.gradients)}, scores: {[g._score for g in self.gradients]} for {self.name}"
        # )

        # sore gradients by the _score from low to high
        self.gradients = sorted(
            self.gradients, key=lambda x: x._score if x._score else 1
        )

        gradient_context_combined = zip(
            self.gradients,
            [self.gradients_context[g] for g in self.gradients],
        )

        gradient_context_combined_str = Prompt(
            template=COMBINED_GRADIENTS_TEMPLATE,
            prompt_kwargs={"combined_gradients": gradient_context_combined},
        )().strip()

        return gradient_context_combined_str

    # TODO: dont use short value
    def get_short_value(self, n_words_offset: int = 10) -> str:
        """
        Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
        This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
        :param n_words_offset: The number of words to show from the beginning and the end of the value.
        :type n_words_offset: int
        """
        # 1. ensure the data is a string
        data = self.data
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

    @staticmethod
    def trace_graph(
        root: "Parameter",
    ) -> Tuple[Set["Parameter"], Set[Tuple["Parameter", "Parameter"]]]:
        nodes, edges = set(), set()

        def build_graph(node: "Parameter"):
            if node in nodes:
                return
            nodes.add(node)
            for pred in node.predecessors:
                edges.add((pred, node))
                build_graph(pred)

        build_graph(root)
        return nodes, edges

    def backward(
        self,
    ):  # engine should be the llm or customized backwards function to pass feedback

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
            node.gradients = _check_and_reduce_gradients(node)
            log.debug(f"v: {node.data}, grad_fn: {node.grad_fn}, {node.get_grad_fn()}")
            if node.get_grad_fn() is not None:  # gradient function takes in the engine
                log.debug(f"Calling gradient function for {node.name}")
                node.grad_fn()

    def draw_graph(
        self,
        add_grads: bool = True,
        format: Literal["png", "svg"] = "png",
        rankdir: Literal["LR", "TB"] = "TB",
        filepath: Optional[str] = None,
    ):
        """Draw the graph of the parameter and its gradients.

        Args:
            add_grads (bool, optional): Whether to add gradients to the graph. Defaults to True.
            format (str, optional): The format of the output file. Defaults to "png".
            rankdir (str, optional): The direction of the graph. Defaults to "TB".
            filepath (str, optional): The path to save the graph. Defaults to None.
        """
        from adalflow.utils import save_json
        from adalflow.utils.global_config import get_adalflow_default_root_path
        import os

        try:
            from graphviz import Digraph

        except ImportError as e:
            raise ImportError(
                "Please install graphviz using 'pip install graphviz' to use this feature"
            ) from e

        # try:
        #     from tensorboardX import SummaryWriter
        # except ImportError as e:
        #     raise ImportError(
        #         "Please install tensorboardX using 'pip install tensorboardX' to use this feature"
        #     ) from e
        assert rankdir in ["LR", "TB"]
        try:
            import textwrap
        except ImportError as e:
            raise ImportError(
                "Please install textwrap using 'pip install textwrap' to use this feature"
            ) from e

        root_path = get_adalflow_default_root_path()
        # # prepare the log directory
        # log_dir = os.path.join(root_path, "logs")

        # # Set up TensorBoard logging
        # writer = SummaryWriter(log_dir)

        filename = f"trace_graph_{self.name}"
        filepath = (
            os.path.join(filepath, filename)
            if filepath
            else os.path.join(root_path, "graphs", filename)
        )
        print(f"Saving graph to {filepath}.{format}")

        def wrap_text(text, width):
            """Wrap text to the specified width, considering HTML breaks."""
            lines = textwrap.wrap(
                text, width, break_long_words=False, replace_whitespace=False
            )
            return "<br/>".join(lines)

        def wrap_and_escape(text, width=40):
            r"""Wrap text to the specified width, considering HTML breaks, and escape special characters."""
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

        nodes, edges = self.trace_graph(self)
        dot = Digraph(format=format, graph_attr={"rankdir": rankdir})
        node_names = set()
        for n in nodes:
            label_color = "darkblue"

            node_label = (
                f"<table border='0' cellborder='1' cellspacing='0'>"
                f"<tr><td><b><font color='{label_color}'>Name: </font></b></td><td>{wrap_and_escape(n.name)}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Role: </font></b></td><td>{wrap_and_escape(n.role_desc.capitalize())}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Value: </font></b></td><td>{wrap_and_escape(n.data)}</td></tr>"
            )
            if n.proposing:
                node_label += f"<tr><td><b><font color='{label_color}'>Proposing</font></b></td><td>{{'Yes'}}</td></tr>"
                node_label += f"<tr><td><b><font color='{label_color}'>Previous Value: </font></b></td><td>{wrap_and_escape(n.previous_data)}</td></tr>"
            if n.requires_opt:
                node_label += f"<tr><td><b><font color='{label_color}'>Requires Optimization: </font ></b></td><td>{{'Yes'}}</td></tr>"
            if n.param_type:
                node_label += f"<tr><td><b><font color='{label_color}'>Type: </font></b></td><td>{wrap_and_escape(n.param_type.name)}</td></tr>"
            if add_grads:
                node_label += f"<tr><td><b><font color='{label_color}'>Gradients: </font></b></td><td>{wrap_and_escape(n.get_gradients_names())}</td></tr>"
                # add a list of each gradient with short value
                # combine the gradients and context
                combined_gradients_contexts = zip(
                    n.gradients, [n.gradients_context[g] for g in n.gradients]
                )
                for g, context in combined_gradients_contexts:
                    gradient_context = context
                    log.info(f"Gradient context display: {gradient_context}")
                    log.info(f"data: {g.data}")
                    node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.name} Feedback: </font></b></td><td>{wrap_and_escape(g.data)}</td></tr>"
                    if gradient_context != "":
                        node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.name} Context: </font></b></td><td>{wrap_and_escape(gradient_context)}</td></tr>"
            if len(n._traces.values()) > 0:
                node_label += f"<tr><td><b><font color='{label_color}'>Traces: keys: </font></b></td><td>{wrap_and_escape(str(n._traces.keys()))}</td></tr>"
                node_label += f"<tr><td><b><font color='{label_color}'>Traces: values: </font></b></td><td>{wrap_and_escape(str(n._traces.values()))}</td></tr>"

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
                # writer.add_text(g.name, str(g.to_dict()))
                # if g.gradient_prompt:
                #     writer.add_text(f"{g.name}_prompt", g.gradient_prompt)
                log.info(f"Gradient: {g.name}, {g.to_dict()}")
                log.info(f"Gradient prompt: {g.gradient_prompt}")
        for n1, n2 in edges:
            dot.edge(n1.name, n2.name)

        dot.render(filepath, format=format, cleanup=True)
        # from PIL import Image
        # try:
        #     import matplotlib.pyplot as plt
        # except ImportError as e:
        #     raise ImportError(
        #         "Please install matplotlib using 'pip install matplotlib' to use this feature"
        #     ) from e
        # from io import BytesIO
        # import numpy as np

        # # Read the rendered image file into memory using matplotlib
        # with open(f"{filepath}.{format}", "rb") as f:
        #     image_bytes = f.read()

        # # Use matplotlib to read the image from bytes
        # image = plt.imread(BytesIO(image_bytes), format=format)

        # # Ensure the image is in the format [H, W, C]
        # if image.ndim == 2:  # Grayscale image
        #     image = np.expand_dims(image, axis=2)

        # Read the rendered image file
        # writer.add_image("graph", image, dataformats="HWC", global_step=1)
        # writer.close()

        # filename = f"{filepath}_prompts.json"
        # prompts = {}
        # for n in nodes:
        #     prompts[n.name] = {
        #         "raw_response": n.raw_response,
        #     }
        #     for g in n.gradients:
        #         prompts[g.name] = {
        #             "gradient_prompt": g.gradient_prompt,
        #         }

        # save_json(prompts, filename)
        # save root node to_dict to json
        save_json(self.to_dict(), f"{filepath}_root.json")
        return dot

    def to_dict(self):
        return {
            "name": self.name,
            "role_desc": self.role_desc,
            "data": str(self.data),
            "requires_opt": self.requires_opt,
            "param_type": str(self.param_type),
            # others
            "predecessors": [pred.to_dict() for pred in self.predecessors],
            "gradients": [grad.to_dict() for grad in self.gradients],
            "previous_data": self.previous_data,
            "gradients_context": [
                (k.name, v) for k, v in self.gradients_context.items()
            ],
            "grad_fn": str(
                self.grad_fn
            ),  # Simplify for serialization, modify as needed
            "gradient_prompt": str(self.gradient_prompt),
            "raw_response": self.raw_response,
            "score": self._score,
            "traces": {k: v.to_dict() for k, v in self._traces.items()},
            "input_args": self.input_args,
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
            gradient_prompt=data["gradient_prompt"],
            raw_response=data["raw_response"],
            input_args=data["input_args"],
            score=data["score"],
            # demos
            demos=[DataClass.from_dict(d) for d in data["demos"]],
        )
        # Reconstruct gradients_context from the list of tuples
        param.gradients_context = defaultdict(
            lambda: None, {cls.from_dict(k): v for k, v in data["gradients_context"]}
        )
        param._traces = {k: DataClass.from_dict(v) for k, v in data["traces"].items()}
        return param

    # TODO: very hard to read directly, need to simplify and let users use to_dict for better readability
    def __repr__(self):
        return f"Parameter(name={self.name}, requires_opt={self.requires_opt}, param_type={self.param_type}, role_desc={self.role_desc}, data={self.data}, predecessors={self.predecessors}, gradients={self.gradients},\
            raw_response={self.raw_response}, input_args={self.input_args}, traces={self._traces})"


def _check_and_reduce_gradients(variable: Parameter) -> Set[Parameter]:

    if variable.get_gradient_and_context_text() == "":
        log.debug(f"No gradients detected for {variable.data}")
        return variable.gradients
    if len(variable.gradients) == 1:
        log.debug(f"Only one gradient, no need to reduce: {variable.gradients}")
        return variable.gradients
    else:
        log.debug(
            f"Multiple gradients detected for {variable.data}. But we are not reducting them."
        )
        return variable.gradients

    # TODO: Implement the reduction logic later
