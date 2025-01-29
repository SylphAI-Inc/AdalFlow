import pytest
from dataclasses import dataclass

from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import FunctionDefinition


@dataclass
class User:
    id: int
    name: str


# Define some dummy functions for testing purposes
def sync_add(x, y, user: User = User(id=1, name="John")):
    return x + y


async def async_add(x, y):
    return x + y


metadata = FunctionDefinition(func_desc="A simple addition tool", func_name="add")


def test_function_tool_sync():
    tool = FunctionTool(definition=metadata, fn=sync_add)
    print(
        f"tool: {tool}, tool.metadata: {tool.definition}, tool.fn: {tool.fn}, tool.async: {tool._is_async}"
    )

    output = tool(1, 2)  # Using __call__ which proxies to call()
    assert output.output == 3
    assert output.name == "add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == (1, 2)


def test_function_tool_async():
    # use default metadata
    tool = FunctionTool(fn=async_add)

    import asyncio

    output = asyncio.run(tool.acall(3, 4))
    assert output.output == 7
    assert output.name == "async_add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == (3, 4)

    # call with sync call with raise ValueError
    with pytest.raises(ValueError):
        tool.call(3, 4)


# def test_invalid_function_tool_initialization():
#     # Test initialization without any function should raise ValueError
#     with pytest.raises(ValueError):
#         tool = FunctionTool(metadata=metadata)


# def test_from_defaults_uses_function_docstring():
#     def sample_function(x, y, user: User = User(id=1, name="John")):
#         """
#         Adds two numbers together and returns the sum.
#         """
#         return x + y

#     tool = FunctionTool(fn=sample_function)

#     expected_description = sample_function.__doc__.strip()
#     actual_description = tool.metadata.description
#     print(f"Expected: {expected_description}, Actual: {actual_description}")


# # Check if the metadata description matches the function's docstring
# assert (
#     actual_description == expected_description
# ), f"The description should automatically be set to the function's docstring. Expected: {expected_description}, Actual: {actual_description}"
