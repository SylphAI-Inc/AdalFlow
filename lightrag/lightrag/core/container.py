from collections import OrderedDict
import operator
from itertools import islice
from typing import TypeVar, Dict, Union, Iterable, Iterator, Any, overload

from lightrag.core.component import Component

T = TypeVar("T", bound=Component)


class Sequential(Component):
    __doc__ = r"""A sequential container.

    Follows the same design pattern as PyTorch's ``nn.Sequential``.

    Components will be added to it in the order they are passed to the constructor.
    Alternatively, an ``OrderedDict`` of components can be passed in.
    It "chains" outputs of the previous component to the input of the next component sequentially.
    Output of the previous component is input to the next component as positional argument.

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
    """

    _components: Dict[str, Component]  # = OrderedDict()

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

    def _get_item_by_idx(self, iterator: Iterator[T], idx: int) -> T:
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
            return self._get_item_by_idx(self._components.values(), idx)

    def __setitem__(self, idx: Union[int, str], component: Component) -> None:
        """Set the idx-th component of the Sequential."""
        if isinstance(idx, str):
            self._components[idx] = component
        else:
            key: str = self._get_item_by_idx(self._components.keys(), idx)
            return setattr(self, key, component)

    def __delitem__(self, idx: Union[slice, int, str]) -> None:
        """Delete the idx-th component of the Sequential."""
        if isinstance(idx, slice):
            for key in list(self._components.keys())[idx]:
                delattr(self, key)
        elif isinstance(idx, str):
            del self._components[idx]
        else:
            key = self._get_item_by_idx(self._components.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._components))]
        self._components = OrderedDict(
            list(zip(str_indices, self._components.values()))
        )

    def __iter__(self) -> Iterable[Component]:
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

    def call(self, input: Any) -> object:
        for component in self._components.values():
            input = component(input)
        return input

    def append(self, component: Component) -> "Sequential":
        r"""Appends a component to the end of the Sequential."""
        idx = len(self._components)
        self.add_component(str(idx), component)
        return self

    def insert(self, idx: int, component: Component) -> None:
        r"""Inserts a component at a given index in the Sequential."""
        if not isinstance(component, Component):
            raise AssertionError(
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
        return self

    def extend(self, components: Iterable[Component]) -> "Sequential":
        r"""Extends the Sequential with components from an iterable."""
        for component in components:
            self.append(component)
        return self
