"""WIP"""

from typing import Generic, TypeVar, Any, List, Set, Dict, TYPE_CHECKING, Tuple
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
        data: T = None,  # for generator output, the data will be set up as raw_response
        requires_opt: bool = True,
        role_desc: str = None,
        predecessors: List["Parameter"] = None,
        alias: str = None,  # alias is used to refer to the parameter in the prompt, easier to read for humans
        gradient_prompt: str = None,
        raw_response: str = None,  # use this to track the raw response of generator instead of the data (can be parsed)
    ):
        if predecessors is None:
            predecessors = []
        self.predecessors = set(predecessors)

        for pred in self.predecessors:
            if not isinstance(pred, Parameter):
                raise TypeError(
                    f"Expected a list of Parameter instances, got {type(pred).__name__}, {pred}"
                )
        self._predecessors_requires_grad = [v for v in predecessors if v.requires_opt]
        self.data = data
        self.requires_opt = requires_opt
        self.data_type = type(
            data
        )  # Dynamically infer the data type from the provided data
        self.role_desc = role_desc
        self.gradients: Set[Parameter] = set()
        self.gradient_prompt: str = (
            gradient_prompt  # the whole llm prompt to compute the gradient
        )
        self.gradients_context: Dict[Parameter, str] = defaultdict(
            lambda: None
        )  # TODO: what is the context
        self.grad_fn = None
        self.alias = alias
        if not self.alias:
            self.alias = (
                self.role_desc.capitalize().replace(" ", "_")[0:10]
                if self.role_desc
                else f"param_{id(self)}"
            )
        self.proposed_data: T = None  # this is for the optimizer to propose a new value
        self.raw_response = raw_response

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

    def get_gradients_alias(self) -> str:
        alias = [g.alias for g in self.gradients]
        ", ".join(alias)
        return alias

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
                gradients.append(f"data: {g.data}\n")  # how come gradient has no data

            else:
                criticism_and_context = Prompt(
                    template=GRADIENT_TEMPLATE,
                    prompt_kwargs={"feedback": g.data, **self.gradients_context[g]},
                )
                gradients.append(f"criticism_and_context: {criticism_and_context()}")
        gradient_text = "\n".join(gradients)

        return gradient_text

    # TODO: dont use short value
    # def get_short_value(self, n_words_offset: int = 10) -> str:
    #     """
    #     Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
    #     This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
    #     :param n_words_offset: The number of words to show from the beginning and the end of the value.
    #     :type n_words_offset: int
    #     """
    #     # 1. ensure the data is a string
    #     data = self.data
    #     if not isinstance(self.data, str):
    #         data = str(self.data)
    #     words = data.split(" ")
    #     if len(words) <= 2 * n_words_offset:
    #         return data
    #     short_value = (
    #         " ".join(words[:n_words_offset])
    #         + " (...) "
    #         + " ".join(words[-n_words_offset:])
    #     )
    #     return short_value

    @staticmethod
    def trace(
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

    def draw_graph(
        self,
        add_grads: bool = True,
        format="png",
        rankdir="TB",
        filepath: str = "gout",
    ):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """

        try:
            from graphviz import Digraph

        except ImportError as e:
            raise ImportError(
                "Please install graphviz using 'pip install graphviz' to use this feature"
            ) from e

        try:
            from tensorboardX import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "Please install tensorboardX using 'pip install tensorboardX' to use this feature"
            ) from e
        assert rankdir in ["LR", "TB"]
        import textwrap

        # Set up TensorBoard logging
        writer = SummaryWriter(log_dir="logs/simple_net")

        filepath = filepath + "." + format

        def wrap_text(text, width):
            # Wrap text to the specified width
            return "<br/>".join(textwrap.wrap(text, width))

        def wrap_and_escape(text, width=40):
            if not isinstance(text, str):
                text = str(text)
            text = (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            return wrap_text(text, width)

        nodes, edges = self.trace(self)
        dot = Digraph(
            format=format, graph_attr={"rankdir": rankdir}
        )  # , node_attr={'rankdir': 'TB'})

        for n in nodes:
            label_color = "darkblue"
            # node_label = (
            #     f"<b><font color='{label_color}'>Alias: </font></b> {wrap_and_escape(n.alias)}"
            #     f"<b><font color='{label_color}'>Role: </font></b> {wrap_and_escape(n.role_desc.capitalize())}"
            #     f"<b><font color='{label_color}'>Value: </font></b> {wrap_and_escape(n.data)}"
            # )
            node_label = (
                f"<table border='0' cellborder='1' cellspacing='0'>"
                f"<tr><td><b><font color='{label_color}'>Alias: </font></b></td><td>{wrap_and_escape(n.alias)}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Role: </font></b></td><td>{wrap_and_escape(n.role_desc.capitalize())}</td></tr>"
                f"<tr><td><b><font color='{label_color}'>Value: </font></b></td><td>{wrap_and_escape(n.data)}</td></tr>"
            )
            if add_grads:
                node_label += f"<tr><td><b><font color='{label_color}'>Gradients: </font></b></td><td>{wrap_and_escape(n.get_gradients_alias())}</td></tr>"
                # add a list of each gradient with short value
                for g in n.gradients:
                    node_label += f"<tr><td><b><font color='{label_color}'>Gradient {g.alias}: </font></b></td><td>{wrap_and_escape(g.data)}</td></tr>"
            node_label += "</table>"
            # if add_grads:
            #     # get the alias of the gradients
            #     node_label += f"<b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(n.get_gradients_alias())}"
            #     node_label += f"<br/><b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(n.get_gradient_and_context_text())}"
            dot.node(
                name=n.alias or n.role_desc,
                label=f"<{node_label}>",
                shape="plaintext",
            )
            writer.add_text(n.alias, str(n.to_dict()))
            log.info(f"Node: {n.alias}, {n.to_dict()}")
            # track gradients
            for g in n.gradients:
                writer.add_text(g.alias, str(g.to_dict()))
                if g.gradient_prompt:
                    writer.add_text(f"{g.alias}_prompt", g.gradient_prompt)
                log.info(f"Gradient: {g.alias}, {g.to_dict()}")
                log.info(f"Gradient prompt: {g.gradient_prompt}")
        for n1, n2 in edges:
            dot.edge(n1.alias, n2.alias)

        dot.render(filepath, format=format, cleanup=True)
        # from PIL import Image
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Please install matplotlib using 'pip install matplotlib' to use this feature"
            ) from e
        from io import BytesIO
        import numpy as np

        # Read the rendered image file into memory using matplotlib
        with open(f"{filepath}.{format}", "rb") as f:
            image_bytes = f.read()

        # Use matplotlib to read the image from bytes
        image = plt.imread(BytesIO(image_bytes), format=format)

        # Ensure the image is in the format [H, W, C]
        if image.ndim == 2:  # Grayscale image
            image = np.expand_dims(image, axis=2)

        # Read the rendered image file
        writer.add_image("graph", image, dataformats="HWC", global_step=1)
        writer.close()
        return dot

    def to_dict(self):
        return {
            "alias": self.alias,
            "data": str(self.data),
            "requires_opt": self.requires_opt,
            "role_desc": self.role_desc,
            "predecessors": [pred.to_dict() for pred in self.predecessors],
            "gradients": [grad.to_dict() for grad in self.gradients],
            "proposed_data": self.proposed_data,
            "gradients_context": [
                (k.alias, v) for k, v in self.gradients_context.items()
            ],
            "grad_fn": str(
                self.grad_fn
            ),  # Simplify for serialization, modify as needed
            "gradient_prompt": str(self.gradient_prompt),
            "raw_response": self.raw_response,
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
            gradient_prompt=data["gradient_prompt"],
            raw_response=data["raw_response"],
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
