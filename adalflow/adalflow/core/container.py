"""
Container component for composing multiple components, such as Sequential
and ComponentList.

This design draws inspiration from PyTorch’s modular
container patterns, including `nn.Sequential` and `nn.ModuleList`. The
`Container` component allows for grouping several components into one, enabling
flexible and reusable model architectures.

Design Motivation:
-------------------
This implementation follows the same principles as PyTorch’s component-based
design, encouraging modularity, reusability, and extensibility. The `Container`
component provides an easy way to manage multiple layers or other components,
while ensuring that their parameters are properly registered and updated during
training.

Credits:
---------
The design of this component takes inspiration from the PyTorch project
(https://pytorch.org). PyTorch is an open-source deep learning framework,
licensed under a BSD-style license. Although this code is not part of the
official PyTorch library, it mirrors the same design principles.

For more details on PyTorch’s licensing, refer to:
https://github.com/pytorch/pytorch/blob/main/LICENSE

Usage Example:
--------------
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()

            self.model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed using ints
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
            return x

"""

from collections import OrderedDict, abc as container_abcs
import operator
from itertools import islice, chain
from typing import TypeVar, Dict, Union, Iterable, Iterator, Any, overload, Optional

from adalflow.core.component import Component

T = TypeVar("T", bound=Component)

__all__ = ["Sequential", "ComponentList"]


class Sequential(Component):
    __doc__ = r"""A sequential container.

    Adapted from PyTorch's ``nn.Sequential``.

    Components will be added to it in the order they are passed to the constructor.
    Alternatively, an ``OrderedDict`` of components can be passed in.
    It "chains" outputs of the previous component to the input of the next component sequentially.
    Output of the previous component is input to the next component as positional argument.

    Benefits of using Sequential:
    1. Convenient for data pipeline that often consists of multiple components. This allow users to encapsulate the pipeline in a single component.
    Examples:

    Without Sequential:

    .. code-block:: python

        class AddAB(Component):
            def call(self, a: int, b: int) -> int:
                return a + b


        class MultiplyByTwo(Component):
            def call(self, input: int) -> int:
                return input * 2

        class DivideByThree(Component):
            def call(self, input: int) -> int:
                return input / 3

        # Manually chaining the components
        add_a_b = AddAB()
        multiply_by_two = MultiplyByTwo()
        divide_by_three = DivideByThree()

        result = divide_by_three(multiply_by_two(add_a_b(2, 3)))



    With Sequential:

    .. code-block:: python

        seq = Sequential(AddAB(), MultiplyByTwo(), DivideByThree())
        result = seq(2, 3)

    .. note::
        Only the first component can receive arbitrary positional and keyword arguments.
        The rest of the components should have a single positional argument as input and have it to be exactly the same type as the output of the previous component.

    2. Apply a transformation or operation (like training, evaluation, or serialization) to the Sequential object, it automatically applies that operation to each component it contains.
    This can be useful for In-context learning training.


    Examples:

    1. Use positional arguments:
        >>> seq = Sequential(component1, component2)
    2. Add components:
        >>> seq.append(component4)
    3. Get a component:
        >>> seq[0]
    4. Delete a component:
        >>> del seq[0]
    5. Iterate over components:
        >>> for component in seq:
        >>>     print(component)
    6. Add two Sequentials:
        >>> seq1 = Sequential(component1, component2)
        >>> seq2 = Sequential(component3, component4)
        >>> seq3 = seq1 + seq2
    7. Use OrderedDict:
        >>> seq = Sequential(OrderedDict({"component1": component1, "component2": component2}))
    8. Index OrderDict:
        >>> seq = Sequential(OrderedDict({"component1": component1, "component2": component2}))
        >>> seq["component1"]
        # or
        >>> seq[0]
    9. Call with a single argument as input:
        >>> seq = Sequential(component1, component2)
        >>> result = seq.call(2)
    10. Call with multiple arguments as input:
        >>> seq = Sequential(component1, component2)
        >>> result = seq.call(2, 3)
    """

    _components: Dict[str, Component] = OrderedDict()  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Component) -> None: ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Component]") -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, component in args[0].items():
                self.add_component(key, component)
        else:
            for idx, component in enumerate(args):
                self.add_component(str(idx), component)

    def _get_item_by_idx(self, iterator: Iterator[Component], idx: int) -> Component:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(
        self, idx: Union[slice, int, str]
    ) -> Union["Sequential", Component]:
        """Get the idx-th and by-key component of the Sequential."""
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._components.items())[idx]))
        elif isinstance(idx, str):
            return self._components[idx]
        else:
            return self._get_item_by_idx(iter(self._components.values()), idx)

    def __setitem__(self, idx: Union[int, str], component: Component) -> None:
        """Set the idx-th component of the Sequential."""
        if isinstance(idx, str):
            self._components[idx] = component
        else:
            # key: str = self._get_item_by_idx(iter(self._components.keys()), idx)
            # self._components[key] = component
            key_list = list(self._components.keys())
            key = key_list[idx]
            self._components[key] = component

    def __delitem__(self, idx: Union[slice, int, str]) -> None:
        """Delete the idx-th component of the Sequential."""
        if isinstance(idx, slice):
            for key in list(self._components.keys())[idx]:
                delattr(self, key)
        elif isinstance(idx, str):
            del self._components[idx]
        else:
            # key = self._get_item_by_idx(iter(self._components.keys()), idx)
            key_list = list(self._components.keys())
            key = key_list[idx]

            delattr(self, key)

        # Reordering is needed if numerical keys are used to keep the sequence
        self._components = OrderedDict(
            (str(i), comp) for i, comp in enumerate(self._components.values())
        )

    def __iter__(self) -> Iterator[Component]:
        r"""Iterates over the components of the Sequential.

        Examples:
        1. Iterate over the components:

        .. code-block:: python

            for component in seq:
                print(component)
        """
        return iter(self._components.values())

    def __len__(self) -> int:
        return len(self._components)

    def __add__(self, other) -> "Sequential":
        r"""Adds two Sequentials.

        Creating a new Sequential with components of both the Sequentials.

        Examples:
        1. Add two Sequentials:

        .. code-block:: python

            seq1 = Sequential(component1, component2)
            seq2 = Sequential(component3, component4)
            seq3 = seq1 + seq2
        """
        if isinstance(other, Sequential):
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

    def __iadd__(self, other) -> "Sequential":
        r"""Inplace add two Sequentials.

        Adding components of the other Sequential to the current Sequential.

        Examples:
        1. Inplace add two Sequentials:

        .. code-block:: python

            seq1 = Sequential(component1, component2)
            seq2 = Sequential(component3, component4)
            seq1 += seq2
        """
        if not isinstance(other, Sequential):
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )
        for layer in other:
            self.append(layer)
        return self

    @overload
    def call(self, input: Any) -> object: ...

    @overload
    def call(self, *args: Any, **kwargs: Any) -> object: ...

    def call(self, *args: Any, **kwargs: Any) -> object:
        if len(args) == 1 and not kwargs:
            input = args[0]
            for component in self._components.values():
                input = component(input)
            return input
        else:
            for component in self._components.values():
                result = component(*args, **kwargs)
                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], dict)
                ):
                    args, kwargs = result
                else:
                    args = (result,)
                    kwargs = {}
            return args[0] if len(args) == 1 else (args, kwargs)

    @overload
    async def acall(self, input: Any) -> object: ...

    @overload
    async def acall(self, *args: Any, **kwargs: Any) -> object: ...

    async def acall(self, *args: Any, **kwargs: Any) -> object:
        r"""When you for loop or multiple await calls inside each component, use acall method can potentially speed up the execution."""
        if len(args) == 1 and not kwargs:
            input = args[0]
            for component in self._components.values():
                input = await component(input)
            return input
        else:
            for component in self._components.values():
                result = await component(*args, **kwargs)
                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], dict)
                ):
                    args, kwargs = result
                else:
                    args = (result,)
                    kwargs = {}
            return args[0] if len(args) == 1 else (args, kwargs)

    def append(self, component: Component) -> "Sequential":
        r"""Appends a component to the end of the Sequential."""
        idx = len(self._components)
        self.add_component(str(idx), component)
        return self

    def insert(self, idx: int, component: Component) -> None:
        r"""Inserts a component at a given index in the Sequential."""
        if not isinstance(component, Component):
            raise TypeError(
                f"component should be an instance of Component, but got {type(component)}"
            )
        n = len(self._components)
        if not (-n <= idx <= n):
            raise IndexError(
                f"index {idx} is out of range for Sequential with length {len(self._components)}"
            )
        if idx < 0:
            idx += n
        for i in range(n, idx, -1):
            self._components[str(i)] = self._components[str(i - 1)]
        self._components[str(idx)] = component

    def extend(self, components: Iterable[Component]) -> "Sequential":
        r"""Extends the Sequential with components from an iterable."""
        for component in components:
            self.append(component)
        return self


def _addindent(s_: str, numSpaces: int):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class ComponentList(Component):
    __doc__ = r"""Holds subcomponents in a list.

    :class:`adalflow.core.ComponentList` can be indexed like a regular Python list, but
    the components it holds are properly registered, and will be visible by all
    :class:`adalflow.core.Component` methods.

    Args:
        components (iterable, optional): an iterable of components to add

    Examples:

    .. code-block:: python

        # Example of how to use ComponentList
        class MyComponents(Component):
            def __init__(self):
                super().__init__()
                self.llms = ComponentList([adal.Generator() for i in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """
    _components: Dict[str, Component] = OrderedDict()

    def __init__(self, components: Optional[Iterable[Component]] = None) -> None:
        super().__init__()
        if components is not None:
            self += components

    def _get_abs_string_index(self, idx):
        """Get the absolute index as a string."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Component, "ComponentList"]:
        """Retrieve a component or a slice of components."""
        if isinstance(idx, slice):
            return self.__class__(list(self._components.values())[idx])
        else:
            return self._components[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, component: Component) -> None:
        """Set a component at the given index."""
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), component)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        """Delete a component or a slice of components."""
        if isinstance(idx, slice):
            for k in range(len(self._components))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._components is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._components))]
        self._components = OrderedDict(
            list(zip(str_indices, self._components.values()))
        )

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._components)

    def __iter__(self) -> Iterator[Component]:
        """Iterate over the components."""
        return iter(self._components.values())

    def __iadd__(self, components: Iterable[Component]) -> "ComponentList":
        """Add multiple components using the `+=` operator."""

        return self.extend(components)

    def __add__(self, other: Iterable[Component]) -> "ComponentList":
        """Concatenate two ComponentLists."""

        combined = ComponentList()
        for i, component in enumerate(chain(self, other)):
            combined.add_component(str(i), component)
        return combined

    def __repr__(self):
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, component: Component) -> None:
        """Insert a component at the specified index."""
        for i in range(len(self._components), index, -1):
            self._components[str(i)] = self._components[str(i - 1)]
        self._components[str(index)] = component

    def pop(self, index: Union[int, slice]) -> Component:
        """Remove and return a component at the given index."""
        component = self[index]
        del self[index]
        return component

    def append(self, component: Component) -> "ComponentList":
        """Append a component to the list."""
        # self._components[str(len(self))] = component
        self.add_component(str(len(self)), component)
        return self

    def extend(self, components: Iterable[Component]) -> "ComponentList":
        """Extend the list by appending multiple components."""
        # for component in components:
        #     self.append(component)
        # return self

        if not isinstance(components, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an "
                "iterable, but got " + type(components).__name__
            )
        offset = len(self)
        for i, component in enumerate(components):
            self.add_component(str(offset + i), component)
        return self


# TODO: need to do the same to ParameterList and ParameterDict, ModuleDict
