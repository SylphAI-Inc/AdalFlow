import pytest

from adalflow.optim.parameter import Parameter


class TestParameter:
    # Test initialization with different types
    @pytest.mark.parametrize(
        "data",
        [
            (123, int),
            (12.34, float),
            ("test", str),
            ([1, 2, 3], list),
            ({"key": "value"}, dict),
            (True, bool),
        ],
    )
    def test_parameter_initialization_with_types(self, data):
        value, expected_type = data
        param = Parameter(data=value)
        assert isinstance(
            param.data, expected_type
        ), f"Data should be of type {expected_type}"

    def test_parameter_initialization(self):
        param = Parameter(data=10, requires_opt=False)
        assert param.data == 10, "The data should be initialized correctly"
        assert (
            param.requires_opt is False
        ), "requires_opt should be initialized correctly"

    @pytest.mark.parametrize(
        "data, new_data",
        [
            (123, 456),  # integers
            ("initial", "updated"),  # strings
            ([1, 2, 3], [4, 5, 6]),  # lists
            ({"key": "value"}, {"new_key": "new_value"}),  # dictionaries
            (True, False),  # booleans
        ],
    )
    def test_update_value(self, data, new_data):
        """Test updating the parameter's data."""
        param = Parameter(data=data)
        param.update_value(new_data)
        assert param.data == new_data, "Parameter data should be updated correctly"

    def test_data_in_prompt_callable(self):
        param = Parameter(
            data=10, requires_opt=False, data_in_prompt=lambda x: f"Data: {x.data}"
        )

        assert (
            param.data_in_prompt(param) == "Data: 10"
        ), "Data should be correctly formatted in the prompt"

        assert (
            param.get_prompt_data() == "Data: 10"
        ), "Data should be correctly formatted in the prompt"

    # def test_update_value_incorrect_type(self):
    #     """Test updating the parameter with an incorrect type."""
    #     param = Parameter[int](data=10)
    #     with pytest.raises(TypeError) as e:
    #         param.update_value("a string")
    #     assert "Expected data type int, got str" in str(
    #         e.value
    #     ), "TypeError should be raised with the correct message"

    # def test_to_dict(self):
    #     param = Parameter(data=10, requires_opt=True)
    #     expected_dict = {
    #         "data": 10,
    #         "requires_opt": True,
    #         "role_desc": None,
    #         "predecessors": set(),
    #     }
    #     output = param.to_dict()
    #     print(f"output: {output}")
    #     assert (
    #         param.to_dict() == expected_dict
    #     ), "to_dict should return the correct dictionary representation"

    # def test_repr(self):
    #     param = Parameter(data="test_param")
    #     assert (
    #         repr(param) == "Parameter: test_param"
    #     ), "The __repr__ should return the correct format"
