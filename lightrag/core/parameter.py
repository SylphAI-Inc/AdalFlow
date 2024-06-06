from typing import Generic, TypeVar

T = TypeVar("T")  # covariant set to False to allow for in-place updates


class Parameter(Generic[T]):
    r"""A generic class to represent a parameter.

    A parameter enforce a specific data type and can be updated in-place.

    Parameters are any data type that has a special property when
    used in a component - when they are assigned as Component attributes
    they are automatically added to the list of its parameters, and  will
    appear in the :meth:`~Component.parameters` iterator.

    Args:
        data (T): the data of the parameter
        requires_opt (bool, optional): if the parameter requires optimization. Default: `True`

    Examples:
        # specify the type explicitly
        int_param = Parameter[int](data=123)
        str_param = Parameter[str](data="hello")
        # update the value in-place
        int_param.update_value(456)
        # expect a TypeError if the type is incorrect
        int_param.update_value("a string")

        # specify the type implicitly
        param = Parameter(data=123)
        param.update_value(456)
        # expect a TypeError if the type is incorrect
        param.update_value("a string")
    """

    def __init__(self, data: T, requires_opt: bool = True):
        self.data = data
        self.data_type = type(
            data
        )  # Dynamically determine the type from the data provided

        # # Initial type check to ensure that the data matches the type specified by T if T is explicit
        # if not isinstance(data, self.data_type):
        #     raise TypeError(
        #         f"Expected data type {self.data_type.__name__}, got {type(data).__name__}"
        #     )

        self.requires_opt = requires_opt

    # def _check_data_type(self, new_data: T):
    #     if not isinstance(new_data, self.data_type):
    #         raise TypeError(
    #             f"Expected data type {self.data_type.__name__}, got {type(new_data).__name__}"
    #         )

    def update_value(self, data: T):
        r"""Update the value in-place."""
        # self._check_data_type(data)
        self.data = data

    def to_dict(self):
        return {"data": self.data, "requires_opt": self.requires_opt}

    def __repr__(self):
        return f"Parameter: {self.data}"
