from collections import OrderedDict
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
    Iterator,
    Mapping,
)
from collections import OrderedDict, abc as container_abcs
import operator
from itertools import islice
import networkx as nx
from pyvis.network import Network

import matplotlib.pyplot as plt
import uuid


# TODO: design hooks.
_global_pre_call_hooks: Dict[int, Callable] = OrderedDict()
__all__ = ["Component", "EmbedderOutput", "OpenAIEmbedder"]


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


def _call_unimplemented(self, *input: Any) -> None:
    r"""
    Define the call method for the component.
    Should be overriden by all subclasses.
    """
    raise NotImplementedError(
        f'Component {type(self).__name__} is missing the required "call" method.'
    )


class Component:
    r"""
    Component defines all functional base classes such as Embedder, Retriever, Generator.

    We purposly avoid using the name "Module" to avoid confusion with PyTorch's nn.Module.
    As we consider 'Component' to be an extension to 'Moduble' as if you use a local llm model
    for the Generator, you might need the 'Module' within the 'Component'.

    But 'Component' follows a similar design pattern to 'Module' in PyTorch.

    (1) 'Module' does not have async function because of GPU's inherent parallelism.
     But we need to support async functions.
     call and acall should call functions.
    (2) All components can be running local or APIs. 'Component' can deal with API calls, so we need support retries and rate limits.
    """

    _version: int = 0
    # TODO: the type of module, is it OrderedDict or just Dict?
    _components: Dict[str, Optional["Component"]]
    _execution_graph: List[str] = []  # This will store the graph of execution.
    _graph = nx.DiGraph()
    _last_called = None  # Tracks the last component called

    # def _generate_unique_name(self):
    #     # Generate a unique identifier that includes the class name
    #     return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_components", {})

    def __setattr__(self, name: str, value: Any) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        components = self.__dict__.get("_components")
        if isinstance(value, Component):
            if components is None:
                raise AttributeError(
                    "cant assign component before Component.__init__() call"
                )
            remove_from(self.__dict__)
            components[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if "_components" in self.__dict__:
            components = self.__dict__.get("_components")
            if name in components:
                return components[name]
        # else:
        #     super().__getattr__(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name in self._components:
            del self._components[name]
        else:
            super().__delattr__(name)

    def extra_repr(self) -> str:
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
        extra_repr = self.extra_repr()
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

    @staticmethod
    def visualize_graph_html(filename="graph.html"):
        nt = Network(directed=True)
        nt.from_nx(Component._graph)
        for edge in nt.edges:
            edge["title"] = edge["label"]
            edge["value"] = (
                10  # You can adjust the 'value' to set the width of the edges
            )
        nt.show_buttons(
            filter_=["physics"]
        )  # Optional: Show interactive buttons to adjust physics and other settings
        nt.show(
            filename, notebook=False
        )  # Make sure to set notebook=False for non-notebook environments

    @staticmethod
    def visualize_graph():
        pos = nx.spring_layout(Component._graph)
        nx.draw(
            Component._graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            edge_color="gray",
            linewidths=1,
            font_size=15,
        )
        plt.show()

    # TODO: do we need to disable this format of calling instead use call and acall extensively?
    def __call__(self, *args, **kwargs):
        r"""In default, we use sync call."""
        # Register the edge if this call follows another component's call
        component_name = self._get_name()
        input_repr = repr(args) + " " + repr(kwargs)

        if Component._last_called is not None:
            Component._graph.add_edge(
                Component._last_called, component_name, label=input_repr[0:50]
            )

        Component._last_called = component_name
        # Component._last_called_input = input_repr[0:50]
        # Update last called to current component
        # Default to sync call
        # Log input and the function
        # being called
        self._execution_graph.append(
            f"{self._get_name()} called with input {input_repr}"
        )
        output = self.call(*args, **kwargs)
        # Log output
        self._execution_graph.append(f"{self._get_name()} output {repr(output)}")
        return output

    # call: Callable[..., Any] = _call_unimplemented

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Component {type(self).__name__} is missing the required 'call' method."
        )

    async def acall(self, *args, **kwargs):
        r"""API call, file io."""
        pass

    def add_component(self, name: str, component: Optional["Component"]) -> None:
        self._components[name] = component

    def register_subcomponent(
        self, name: str, component: Optional["Component"]
    ) -> None:
        r"""
        Alias for add_component
        """
        self.add_component(name, component)

    def get_subcomponent(self, name: str) -> Optional["Component"]:
        return self._components.get(name)

    def named_children(self) -> Iterable[Tuple[str, "Component"]]:
        r"""
        Returns an iterator over immediate children modules.
        """
        memo = set()
        for name, component in self._components.items():
            if component is not None and component not in memo:
                memo.add(component)
                yield name, component

    def children(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over immediate children modules.
        """
        for name, component in self.named_children():
            yield component

    def named_components(
        self,
        memo: Optional[Set["Component"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._components.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_components(
                    memo, submodule_prefix, remove_duplicate
                )

    def components(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over all components in the Module.
        """
        for name, component in self.named_children():
            yield component

    def apply(self: "Component", fn: Callable[["Component", Any], None]) -> None:
        r"""
        Applies a function to all subcomponents.
        # TODO: in what situation we need function apply?
        """
        for name, component in self.children():
            component.apply(fn)
        fn(self)
        return self


class Sequential(Component):
    r"""A sequential container. Components will be added to it in the order they are passed to the constructor.

    Output of the previous component is input to the next component as positional argument.
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

    def _get_item_by_idx(self, iterator, idx) -> Component:  # type: ignore[misc, type-var]
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

    def __len__(self) -> int:
        return len(self._components)

    def append(self, component: Component) -> "Sequential":
        r"""Appends a component to the end of the Sequential."""
        idx = len(self._components)
        self.add_component(str(idx), component)
        return self

    def __add__(self, other) -> "Sequential":
        if not isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def call(self, input: Any) -> Any:
        for component in self._components.values():
            input = component(input)
        return input


class ComponentDict(Component):
    r"""
    We directly used the code in PyTorch's ModuleDict.
    See its design here: /torch/nn/modules/container.py

    Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _components: Dict[str, Component]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Component]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Component:
        return self._components[key]

    def __setitem__(self, key: str, component: Component) -> None:
        self.add_component(key, component)

    def __delitem__(self, key: str) -> None:
        del self._components[key]

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __contains__(self, key: str) -> bool:
        return key in self._components

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._components.clear()

    def pop(self, key: str) -> Component:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._components.keys()

    def items(self) -> Iterable[Tuple[str, Component]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._components.items()

    def values(self) -> Iterable[Component]:
        r"""Return an iterable of the ModuleDict values."""
        return self._components.values()

    def update(self, components: Mapping[str, Component]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(components, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(components).__name__
            )

        if isinstance(components, (OrderedDict, ComponentDict, container_abcs.Mapping)):
            for key, module in components.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(components):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented
