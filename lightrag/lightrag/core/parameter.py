"""WIP"""
from typing import Generic, TypeVar, Any

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
    """

    def __init__(self, data: T = None, requires_opt: bool = True):
        self.data = data
        self.requires_opt = requires_opt
        self.data_type = type(
            data
        )  # Dynamically infer the data type from the provided data

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

    def to_dict(self):
        return {"data": self.data, "requires_opt": self.requires_opt}

    def __repr__(self):
        return f"Parameter: {self.data}"
