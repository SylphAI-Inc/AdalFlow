"""Component is to LLM task pipelines what nn.Module is to PyTorch models."""

from collections import OrderedDict, namedtuple
from typing import (
    Callable,
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Iterable,
    Set,
    Union,
    overload,
    Mapping,
    TypeVar,
    Type,
)
import operator
from itertools import islice
import logging
import pickle
import inspect


from lightrag.core.parameter import Parameter
from lightrag.utils.serialization import default
from lightrag.utils.config import new_component

from lightrag.utils.registry import EntityMapping

# import networkx as nx
# from pyvis.network import Network

# import matplotlib.pyplot as plt
# import itertools
log = logging.getLogger(__name__)

T = TypeVar("T")


# class _IncompatibleKeys(
#     namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
# ):
#     def __repr__(self):
#         if not self.missing_keys and not self.unexpected_keys:
#             return "<All keys matched successfully>"
#         return super().__repr__()

#     __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Component:
    r"""
    Base class for all LLM task pipeline components.

    Such as ``Prompt``, ``ModelClient``, ``Embedder``, ``Retriever``, ``Generator``, etc.
    Your task pipeline should subclass this.

    Components can also contain other Components, allowing to nest them in
    a tree structure. You can assign the subcomponents as regular attributes::

    Example:

    .. code-block:: python

        from lightrag.core import Component, Generator
        from lightrag.components.model_client import OpenAIClient

        template_doc = r"<SYS> You are a doctor </SYS> User: {{input_str}}"

        class DocQA(Component):
        def __init__(self):
            super(DocQA, self).__init__()
            self.doc = Generator(
                template=template_doc,
                model_client=OpenAIClient(),
                model_kwargs={"model": "gpt-3.5-turbo"},
            )

        def call(self, query: str) -> str:
            return self.doc(query).data

        # instantiate the component
        doc_qa = DocQA()
        print(doc_qa)

    The print will be:

    .. code-block::

        DocQA(
        (doc): Generator(
            model_kwargs={'model': 'gpt-3.5-turbo'}, model_type=ModelType.LLM
            (system_prompt): Prompt(template: <SYS> You are a doctor </SYS> User: {{input_str}}, prompt_variables: ['input_str'])
            (model_client): OpenAIClient()
        )
        )

    We follow the same design pattern as PyTorch's ``nn.Module.``
    Instead of working with ``Tensor`` and ``Parameter`` to train models with weights and biases,
    our component works with any data, ``Parameter`` that can be any data type for LLM in-context learning, from manual to auto prompt engineering.
    Besides, (1) instead of `forward` and `backward` functions, we have `call` and `acall` functions for sync and async calls.
    (2) we provide `to_dict` to handle serialization of the whole component states on top of `state_dict`.

    We purposly avoid using the name "Module" to avoid confusion with PyTorch's nn.Module.
    As we consider 'Component' to be an extension to 'Moduble' as if you use a local llm model
    for the Generator, you might need the 'Module' within the 'Component'.
    """

    _version: int = 1  # Version of the component
    # TODO: the type of module, is it OrderedDict or just Dict?
    _components: Dict[str, Optional["Component"]]
    _init_args: Dict[str, Any] = {}  # Store the init arguments
    # _execution_graph: List[str] = []  # This will store the graph of execution.
    # _graph = nx.DiGraph()
    # _last_called = None  # Tracks the last component called
    _parameters: Dict[str, Optional[Parameter]]
    training: bool

    # def _generate_unique_name(self):
    #     # Generate a unique identifier that includes the class name
    #     return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_components", OrderedDict())
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("training", False)
        # only for tracking the init args
        super().__setattr__("_init_args", self._get_init_args(*args, **kwargs))

    def train(self, mode: bool = True):
        r"""Sets the component in training mode."""
        if not isinstance(mode, bool):
            raise ValueError("mode should be a boolean")
        self.training = mode
        for component in self.children():
            component.train(mode)
        return self

    def eval(self):
        r"""Sets the component in evaluation mode."""
        return self.train(False)

    def __dir__(self):
        r"""Useful to handle json serialization.

        Use dir() to get the list of attributes of the component.
        """
        component_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        components = list(self._components.keys())
        keys = component_attrs + attrs + parameters + components

        # Elimiate attrs that are not legal python variable names
        keys = [key for key in keys if not key[0].isdigit()]
        return sorted(keys)

    def is_picklable(self) -> bool:
        """
        Test if the given object is picklable.

        Args:
            obj: The object to test.

        Returns:
            bool: True if the object is picklable, False otherwise.
        """
        try:
            import io

            # Create a BytesIO buffer to simulate file I/O
            buffer = io.BytesIO()
            target = self.to_dict()
            # Try to serialize the object to the buffer
            pickle.dump(target, buffer)
            # Reset the buffer's position to the beginning
            buffer.seek(0)
            # Try to deserialize the object from the buffer
            pickle.load(buffer)
            return True
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            log.info(f"Object is not picklable: {e}")
            return False

    def pickle_to_file(self, filepath: str) -> None:
        """Pickle the component to a file."""
        with open(filepath, "wb") as file:
            pickle.dump(self.to_dict(), file)

    @classmethod
    def load_from_pickle(cls, filepath: str) -> None:
        """Load the component from a file."""
        with open(filepath, "rb") as file:
            return cls.from_dict(pickle.load(file))

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """Converts the component to a dictionary object for serialization, including more states of the component than state_dict.

        Each data if of format: {"type": type, "data": data}
        """
        exclude = exclude or []
        exclude = set(exclude)
        result: Dict[str, Any] = {
            "type": type(self).__name__,
            "data": {},
        }  # Add the type of the component
        data_dict = result["data"]
        for key, value in self.__dict__.items():
            key = str(key)
            if key in exclude:
                continue

            try:
                data_dict[key] = self._process_value(value)
            except TypeError:
                # Handle unserializable objects by pickling them
                data_dict[key] = {"_pickle_data": pickle.dumps(value).hex()}
        return result

    def _process_value(self, value):
        """Process values recursively for serialization."""
        # if isinstance(value, dict):
        #     # Recurse into dictionaries
        #     return {k: self._process_value(v) for k, v in sorted(value.items())}

        if isinstance(value, dict):
            if isinstance(value, OrderedDict):
                return {
                    "_ordered_dict": True,
                    "data": [(k, self._process_value(v)) for k, v in value.items()],
                }
            # return {k: self._process_value(v) for k, v in sorted(value.items())}
            # no sorting
            return {k: self._process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively process list items
            try:
                # return sorted(self._process_value(v) for v in value)
                return [self._process_value(v) for v in value]
            except TypeError:
                # If elements are not comparable, process them without sorting
                return [self._process_value(v) for v in value]
        elif hasattr(value, "to_dict"):
            # Check if the object has a to_dict method
            return self._attempt_to_dict(value)
        else:
            return value

    def _attempt_to_dict(self, obj):
        """Attempt to call to_dict on an object, handling both instances and class types."""
        try:
            # call our customized json serializer
            return default(obj)
        except Exception as e:
            log.error(f"Error calling to_dict for object {obj}: {e}")
            # Fallback to a simpler representation
            return {"type": type(obj).__name__, "data": str(obj)}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an instance from previously serialized data
        using :meth:`~to_dict` method.
        """
        # set the attributes of the class
        # Instantiate the class with _init_args

        # obj = cls(**data.get("_init_args", {}))
        obj = cls.__new__(cls)  # Create a new instance without calling __init__
        for key, value in data["data"].items():
            setattr(obj, key, cls._restore_value(value))
        return obj

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        """Create an instance of the component from a configuration dictionary."""
        kwargs = config.copy()
        # check arguments that are a component
        for key, value in kwargs.items():
            if "component_name" in value:
                kwargs[key] = new_component(value)
        return cls(**kwargs)

    @staticmethod
    def _restore_value(value: Any) -> Any:
        """Restore the original value from the serialized value."""
        if isinstance(value, dict):
            if "_pickle_data" in value:
                # Handle unpickling
                return pickle.loads(bytes.fromhex(value["_pickle_data"]))
            if "_ordered_dict" in value and value["_ordered_dict"]:
                return OrderedDict(
                    (Component._restore_value(k), Component._restore_value(v))
                    for k, v in value["data"]
                )
            if "type" in value and "data" in value:
                # this is likely a class instance, restore using from_dict
                class_name = value.get("type")
                class_type = EntityMapping.get(class_name) or globals().get(class_name)
                if class_type:
                    try:
                        log.info(
                            f"Restoring class using from_dict {class_name}, {value}"
                        )
                        return class_type.from_dict(value)
                    except Exception as e:
                        log.error(
                            f"Error restoring class using from_dict {class_name}: {e}, value{value}"
                        )
                        return class_type()
                else:
                    log.error(f"Unknown class type: {class_name}")
                    return value
            else:  # Recursively process the dictionary
                return {k: Component._restore_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [Component._restore_value(v) for v in value]
        return value

    @classmethod
    def load_state_pickle(cls, filepath: str) -> "Component":
        """Load the state of the component from a file using pickle."""
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def register_parameter(self, name: str, param: Optional[Parameter] = None) -> None:
        r"""Add a parameter to the component.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed using this name.
            param (Parameter): parameter to be added.

        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cant assign parameter before Component.__init__() call"
            )
        elif "." in name:
            raise ValueError('parameter name can\'t contain "."')
        elif name == "":
            raise ValueError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign'{type(param)}' object to parameter '{name}'(Parameter or None required)"
            )
        else:
            self._parameters[name] = param

    def parameters(self, recursive: bool = True) -> Iterable[Parameter]:
        r"""Returns an iterator over module parameters.

        Args:
            recursive (bool): if True, then yields parameters of this component and all subcomponents.
                Otherwise, yields only parameters that are direct members of this component.

        Yields:
            Parameter: module parameter

        Examples:
            >>> for param in model.parameters():
            >>>     print(param)
        """
        for name, param in self.named_parameters(recursive=recursive):
            yield param

    def _named_members(
        self,
        get_members_fn,
        prefix: str = "",
        recursive: bool = True,
        remove_duplicate: bool = True,
    ):
        r"""Helper method for yielding various names + members of the component.

        Args:
            get_members_fn (Callable): callable to extract the members from the component.
            prefix (str): prefix to prepend to all parameter names.
            recursive (bool): if True, then yields parameters of this component and all subcomponents.
                Otherwise, yields only parameters that are direct members of this component.

        Yields:
            Tuple[str, Any]: Tuple containing the name and member

        Examples:
            >>> for name, param in model._named_members(model.named_parameters):
            >>>     print(name, param)
        """
        memo = set()
        components = (
            self.named_components(prefix=prefix, remove_duplicate=remove_duplicate)
            if recursive
            else [(prefix, self)]
        )
        for component_prefix, component in components:
            members = get_members_fn(component)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = component_prefix + ("." if component_prefix else "") + k
                yield name, v

    def named_parameters(
        self, prefix: str = "", recursive: bool = True, remove_duplicate: bool = True
    ) -> Iterable[Tuple[str, Parameter]]:
        r"""Returns an iterator over componenet parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recursive (bool): if True, then yields parameters of this component and all subcomponents.
                Otherwise, yields only parameters that are direct members of this component.
                are direct members of this component.
            remove_duplicate (bool): if True, then yields only unique parameters.

        Yields:
            Tuple[str, Parameter]: Tuple containing the name and parameter

        Examples:
            >>> for name, param in model.named_parameters():
            >>>     print(name, param)
        """
        gen = self._named_members(
            lambda component: component._parameters.items(),
            prefix=prefix,
            recursive=recursive,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    # @staticmethod
    # def visualize_graph_html(filename="graph.html"):
    #     nt = Network(directed=True)
    #     nt.from_nx(Component._graph)
    #     for edge in nt.edges:
    #         edge["title"] = edge["label"]
    #         edge["value"] = (
    #             10  # You can adjust the 'value' to set the width of the edges
    #         )
    #     nt.show_buttons(
    #         filter_=["physics"]
    #     )  # Optional: Show interactive buttons to adjust physics and other settings
    #     nt.show(
    #         filename, notebook=False
    #     )  # Make sure to set notebook=False for non-notebook environments

    # @staticmethod
    # def visualize_graph():
    #     pos = nx.spring_layout(Component._graph)
    #     nx.draw(
    #         Component._graph,
    #         pos,
    #         with_labels=True,
    #         node_color="lightblue",
    #         node_size=2000,
    #         edge_color="gray",
    #         linewidths=1,
    #         font_size=15,
    #     )
    #     plt.show()

    # TODO: do we need to disable this format of calling instead use call and acall extensively?
    def __call__(self, *args, **kwargs):
        r"""In default, we use sync call."""
        # Register the edge if this call follows another component's call
        # component_name = self._get_name()
        # input_repr = repr(args) + " " + repr(kwargs)

        # if Component._last_called is not None:
        #     Component._graph.add_edge(
        #         Component._last_called, component_name, label=input_repr[0:50]
        #     )

        # Component._last_called = component_name

        # self._execution_graph.append(
        #     f"{self._get_name()} called with input {input_repr}"
        # )
        output = self.call(*args, **kwargs)
        # Log output
        # self._execution_graph.append(f"{self._get_name()} output {repr(output)}")
        return output

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Component {type(self).__name__} is missing the required 'call' method."
        )

    async def acall(self, *args, **kwargs):
        r"""API call, file io."""
        pass

    def add_component(self, name: str, component: Optional["Component"]) -> None:
        r"Add a child component to the current component."
        if not isinstance(component, Component) and component is not None:
            raise TypeError(
                f"component should be an instance of Component, but got {type(component)}"
            )
        if not isinstance(name, str):
            raise TypeError(f"name should be a string, but got {type(name)}")
        elif hasattr(self, name) and name not in self._components:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise ValueError('component name can\'t contain "."')
        elif name == "":
            raise ValueError('component name can\'t be empty string ""')
        self._components[name] = component

    def register_component(self, name: str, component: Optional["Component"]) -> None:
        r"""
        Alias for add_component
        """
        self.add_component(name, component)

    def get_subcomponent(self, name: str) -> Optional["Component"]:
        return self._components.get(name)

    def named_children(self) -> Iterable[Tuple[str, "Component"]]:
        r"""
        Returns an iterator over immediate children components.
        """
        memo = set()
        for name, component in self._components.items():
            if component is not None and component not in memo:
                memo.add(component)
                yield name, component

    def children(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over immediate children components.
        """
        for name, component in self.named_children():
            yield component

    def components(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over all components in the Module.
        """
        for name, component in self.named_children():
            yield component

    def named_components(
        self,
        memo: Optional[Set["Component"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all components in the pipeline, yielding both the name of the component as well as the component itself.

        This can be used to represent the state of the component in a dictionary format.
        Args:
            memo (Optional[Set["Component"]]): a memo to store the set of components already added to the result
            prefix (str): a prefix to prepend to all component names
            remove_duplicate (bool): if True, then yields only unique components

        Yields:
            Tuple[str, "Component"]: Tuple containing the name and component

        Example:
            >>> qa = Generator(template="User {{input}}", model_client=GroqAPIClient(), model_kwargs={"model": "llama3-8b-8192"})
            >>> for idx, c in enumerate(qa.named_components()):
            ...     print(f"{idx} -> {c}")

            0 -> ('', Generator(
            model_kwargs={'model': 'llama3-8b-8192'}, model_type=ModelType.LLM
            (system_prompt): Prompt(template: User: {{input}}, prompt_variables: ['input'])
            (model_client): GroqAPIClient()
            ))
            1 -> ('system_prompt', Prompt(template: User: {{input}}, prompt_variables: ['input']))
            2 -> ('model_client', GroqAPIClient())
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, component in self._components.items():
                if component is None:
                    continue
                subcomponent_prefix = prefix + ("." if prefix else "") + name
                yield from component.named_components(
                    memo, subcomponent_prefix, remove_duplicate
                )

    def _save_to_state_dict(self, destination, prefix):
        r"""Saves the state of the component to a dictionary.

        Args:
            destination (Dict[str, Any]): the dictionary to which the state is saved.
            prefix (str): a prefix to add to the keys in the state_dict.
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param

    # TODO: test it + add example
    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: Optional[str] = ""
    ) -> Dict[str, Any]:
        r"""Returns a dictionary containing references to the whole state of the component.

        Parameters are included for now.

        ..note:
            The returned object is a shallow copy. It cantains references
            to the component's parameters and subcomponents.
        Args:
            destination (Dict[str, Any]): If provided, the state of component will be copied into it.
            And the same object is returned.
            Othersie, an ``OrderedDict`` will be created and returned.

            prefix (str): a prefix to add to the keys in the state_dict.

        Returns:
            Dict[str, Any]: a dictionary containing the state of the component.
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()  # type: ignore[attr-defined]
        local_metadata = dict(version=self._version)
        # to do when local data where be needed
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata  # type: ignore[index]

        # save its own state
        self._save_to_state_dict(destination, prefix=prefix)
        # save the state of all subcomponents
        for name, component in self._components.items():
            if component is not None:
                component.state_dict(destination=destination, prefix=f"{prefix}{name}.")
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        # local_metadata=None,
        strict=True,
        missing_keys: List[str] = [],
        unexpected_keys: List[str] = [],
        # error_msgs: List[str] = [],
    ):
        r"""Copies parameters from :attr:`state_dict` into this component but not its descendants.

        This is called on every subcomponent"""
        local_state = {k: v for k, v in self._parameters.items() if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if isinstance(input_param, Parameter):
                    input_param = input_param.data
                if input_param is not None:
                    param.update_value(input_param)  # update the value of the parameter
            elif strict:
                missing_keys.append(key)

        # deal with unexpected keys
        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :]
                    input_name = input_name.split(".", 1)[0]
                    if (
                        input_name not in self._components
                        and input_name not in local_state
                    ):
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Copy parameters from :attr:`state_dict` into this component and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys
        returned by this component's :meth:`~state_dict` function.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"state_dict should be a mapping, but got {type(state_dict)}"
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []
        # metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(
            state_dict
        )  # To prevent modification of the original state_dict
        # if metadata is not None:
        #     state_dict._metadata = metadata

        def load(component: Component, local_state_dict: Mapping[str, Any], prefix=""):
            component._load_from_state_dict(
                local_state_dict,
                prefix=prefix,
                strict=strict,
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            )

            for name, child in component._components.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, prefix=child_prefix)

        load(self, state_dict)
        # del load
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(missing_keys)
                    ),
                )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])
        # return _IncompatibleKeys(
        #     missing_keys=missing_keys, unexpected_keys=unexpected_keys
        # )

    # def apply(self: "Component", fn: Callable[["Component", Any], None]) -> None:
    #     r"""
    #     Applies a function to all subcomponents.
    #     # TODO: in what situation we need function apply?
    #     """
    #     for name, component in self.children():
    #         component.apply(fn)
    #     fn(self)
    #     return self

    def __setattr__(self, name: str, value: Any) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        # set parameter
        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cant assign parameter before Component.__init__() call"
                )
            remove_from(self.__dict__)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{type(value)}' object to parameter '{name}' (Parameter or None required)"
                )
            self.register_parameter(name, value)
        else:  # set component

            components = self.__dict__.get("_components")
            if isinstance(value, Component):
                if components is None:
                    raise AttributeError(
                        "cant assign component before Component.__init__() call"
                    )
                remove_from(self.__dict__)
                components[name] = value

            else:  # set attribute
                super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if "_parameters" in self.__dict__:
            parameters = self.__dict__["_parameters"]
            if name in parameters:
                return parameters[name]
        if "_components" in self.__dict__:
            components = self.__dict__["_components"]
            if name in components:
                return components[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        if name in self._components:
            del self._components[name]
        else:
            super().__delattr__(name)

    def _extra_repr(self) -> str:
        """
        Normally implemented by subcomponents to print additional positional or keyword arguments.
        # NOTE: Dont add components as it will have its own __repr__
        """
        return ""

    def _get_name(self):
        # return self._name
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self._extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, component in self._components.items():
            mod_str = repr(component)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def _get_init_args(self, *args, **kwargs) -> Dict[str, Any]:
        """Save the initialization arguments."""
        sig = inspect.signature(self.__class__.__init__)
        params = sig.parameters
        init_args = {}

        for i, (key, param) in enumerate(params.items()):
            if key == "self":
                continue
            if param.default == inspect.Parameter.empty:
                init_args[key] = args[i] if i < len(args) else kwargs.get(key)
            else:
                init_args[key] = kwargs.get(key, param.default)
        return init_args


T = TypeVar("T", bound=Component)


class Sequential(Component):
    __doc__ = r"""A sequential container.

    Components will be added to it in the order they are passed to the constructor.
    Output of the previous component is input to the next component as positional argument.

    Examples:
    1. Use list as input:
        >>> seq = Sequential([component1, component2])
    2. Use positional arguments:
        >>> seq = Sequential(component1, component2)
    3. Use both:
        >>> seq = Sequential(component1, [component2, component3])
    4. Add components:
        >>> seq.append(component4)
    5. Get a component:
        >>> seq[0]
    6. Delete a component:
        >>> del seq[0]
    7. Iterate over components:
        >>> for component in seq:
        >>>     print(component)
    """

    _components: Dict[str, Component]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Component) -> None: ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Component]") -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_component(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_component(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", Component]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._components.items())[idx]))
        else:
            return self._get_item_by_idx(self._components.values(), idx)

    def __setitem__(self, idx: int, component: Component) -> None:
        key: str = self._get_item_by_idx(self._components.keys(), idx)
        return setattr(self, key, component)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._components.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._components.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._components))]
        self._components = OrderedDict(
            list(zip(str_indices, self._components.values()))
        )

    def __iter__(self) -> Iterable[Component]:
        return iter(self._components.values())

    def __len__(self) -> int:
        return len(self._components)

    def append(self, component: Component) -> "Sequential":
        r"""Appends a component to the end of the Sequential."""
        idx = len(self._components)
        self.add_component(str(idx), component)
        return self

    # def __add__(self, other) -> "Sequential":
    #     if not isinstance(other, Sequential):
    #         ret = Sequential()
    #         for layer in self:
    #             ret.append(layer)
    #         for layer in other:
    #             ret.append(layer)
    #         return ret
    #     else:
    #         raise ValueError(
    #             "add operator supports only objects "
    #             f"of Sequential class, but {str(type(other))} is given."
    #         )

    def call(self, input: Any) -> Any:
        for component in self._components.values():
            input = component(input)
        return input


# TODO: support async call
class FunComponent(Component):
    r"""Component that wraps a function.

    Args:
        fun (Callable): The function to be wrapped.

    Examples:

    function = lambda x: x + 1
    fun_component = FunComponent(function)
    print(fun_component(1))  # 2
    """

    def __init__(self, fun: Optional[Callable] = None, afun: Optional[Callable] = None):
        super().__init__()
        self.fun_name = fun.__name__
        EntityMapping.register(self.fun_name, fun)

    def call(self, *args, **kwargs):
        fun = EntityMapping.get(self.fun_name)
        return fun(*args, **kwargs)

    def _extra_repr(self) -> str:
        return super()._extra_repr() + f"fun_name={self.fun_name}"


def fun_to_component(fun) -> FunComponent:
    r"""Helper function to convert a function into a Component with
    its own class name.

    Can be used as both a decorator and a function.

    Args:
        fun (Callable): The function to be wrapped.
    Returns:
        FunComponent: The component that wraps the function.

    Examples:
    1. As a decorator:
        >>> @fun_to_component
        >>> def my_function(x):
        >>>     return x + 1
        >>> # is equivalent to
        >>> class MyFunctionComponent(FunComponent):
        >>>     def __init__(self):
        >>>         super().__init__(my_function)

    2. As a function:
        >>> my_function_component = fun_to_component(my_function)
    """

    # Split the function name by underscores, capitalize each part, and join them back together
    class_name = (
        "".join(part.capitalize() for part in fun.__name__.split("_")) + "Component"
    )
    print(f"Class name: {class_name}, function name: {fun.__name__}")
    # register the function
    EntityMapping.register(fun.__name__, fun)
    # Define a new component class dynamically
    component_class = type(
        class_name,
        (FunComponent,),
        {"__init__": lambda self: FunComponent.__init__(self, fun)},
    )
    # register the component
    EntityMapping.register(class_name, component_class)

    return component_class()


# TODO: not used yet, will further investigate dict mode
# class ComponentDict(Component):
#     r"""
#     We directly used the code in PyTorch's ModuleDict.
#     See its design here: /torch/nn/modules/container.py

#     Holds submodules in a dictionary.

#     :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
#     but modules it contains are properly registered, and will be visible by all
#     :class:`~torch.nn.Module` methods.

#     :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

#     * the order of insertion, and

#     * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
#       ``OrderedDict``, ``dict`` (started from Python 3.6) or another
#       :class:`~torch.nn.ModuleDict` (the argument to
#       :meth:`~torch.nn.ModuleDict.update`).

#     Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
#     types (e.g., Python's plain ``dict`` before Python version 3.6) does not
#     preserve the order of the merged mapping.

#     Args:
#         modules (iterable, optional): a mapping (dictionary) of (string: module)
#             or an iterable of key-value pairs of type (string, module)

#     Example::

#         class MyModule(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.choices = nn.ModuleDict({
#                         'conv': nn.Conv2d(10, 10, 3),
#                         'pool': nn.MaxPool2d(3)
#                 })
#                 self.activations = nn.ModuleDict([
#                         ['lrelu', nn.LeakyReLU()],
#                         ['prelu', nn.PReLU()]
#                 ])

#             def forward(self, x, choice, act):
#                 x = self.choices[choice](x)
#                 x = self.activations[act](x)
#                 return x
#     """

#     _components: Dict[str, Component]  # type: ignore[assignment]

#     def __init__(self, modules: Optional[Mapping[str, Component]] = None) -> None:
#         super().__init__()
#         if modules is not None:
#             self.update(modules)

#     def __getitem__(self, key: str) -> Component:
#         return self._components[key]

#     def __setitem__(self, key: str, component: Component) -> None:
#         self.add_component(key, component)

#     def __delitem__(self, key: str) -> None:
#         del self._components[key]

#     def __len__(self) -> int:
#         return len(self._components)

#     def __iter__(self) -> Iterator[str]:
#         return iter(self._components)

#     def __contains__(self, key: str) -> bool:
#         return key in self._components

#     def clear(self) -> None:
#         """Remove all items from the ModuleDict."""
#         self._components.clear()

#     def pop(self, key: str) -> Component:
#         r"""Remove key from the ModuleDict and return its module.

#         Args:
#             key (str): key to pop from the ModuleDict
#         """
#         v = self[key]
#         del self[key]
#         return v

#     def keys(self) -> Iterable[str]:
#         r"""Return an iterable of the ModuleDict keys."""
#         return self._components.keys()

#     def items(self) -> Iterable[Tuple[str, Component]]:
#         r"""Return an iterable of the ModuleDict key/value pairs."""
#         return self._components.items()

#     def values(self) -> Iterable[Component]:
#         r"""Return an iterable of the ModuleDict values."""
#         return self._components.values()

#     def update(self, components: Mapping[str, Component]) -> None:
#         r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

#         .. note::
#             If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
#             an iterable of key-value pairs, the order of new elements in it is preserved.

#         Args:
#             modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
#                 or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
#         """
#         if not isinstance(components, container_abcs.Iterable):
#             raise TypeError(
#                 "ModuleDict.update should be called with an "
#                 "iterable of key/value pairs, but got " + type(components).__name__
#             )

#         if isinstance(components, (OrderedDict, ComponentDict, container_abcs.Mapping)):
#             for key, module in components.items():
#                 self[key] = module
#         else:
#             # modules here can be a list with two items
#             for j, m in enumerate(components):
#                 if not isinstance(m, container_abcs.Iterable):
#                     raise TypeError(
#                         "ModuleDict update sequence element "
#                         "#" + str(j) + " should be Iterable; is" + type(m).__name__
#                     )
#                 if not len(m) == 2:
#                     raise ValueError(
#                         "ModuleDict update sequence element "
#                         "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
#                     )
#                 # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
#                 # that's too cumbersome to type correctly with overloads, so we add an ignore here
#                 self[m[0]] = m[1]  # type: ignore[assignment]

#     # remove forward alltogether to fallback on Module's _forward_unimplemented
