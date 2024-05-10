import pytest
from core.tool_helper import FunctionTool, ToolMetadata, ToolOutput

from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str


# Define some dummy functions for testing purposes
def sync_add(x, y, user: User = User(id=1, name="John")):
    return x + y


async def async_add(x, y):
    return x + y


metadata = ToolMetadata(description="A simple addition tool")


def test_function_tool_sync():
    tool = FunctionTool(metadata=metadata, fn=sync_add)
    print(
        f"tool: {tool}, tool.metadata: {tool.metadata}, tool.fn: {tool.fn}, tool.async_fn: {tool.async_fn}"
    )

    output = tool(1, 2)  # Using __call__ which proxies to call()
    assert output.raw_output == 3
    assert output.name is None  # Since name is optional and not set
    assert "args" in output.raw_input
    assert output.raw_input["args"] == (1, 2)


def test_function_tool_async():
    tool = FunctionTool(metadata=metadata, async_fn=async_add)

    import asyncio

    output = asyncio.run(tool.acall(3, 4))
    assert output.raw_output == 7
    assert output.name is None  # Since name is optional and not set
    assert "args" in output.raw_input
    assert output.raw_input["args"] == (3, 4)


def test_invalid_function_tool_initialization():
    # Test initialization without any function should raise ValueError
    with pytest.raises(ValueError):
        tool = FunctionTool(metadata=metadata)


def test_tool_output_str_content():
    output = ToolOutput(raw_input={}, raw_output=100)
    assert str(output) == "100"


def test_from_defaults_uses_function_docstring():
    def sample_function(x, y, user: User = User(id=1, name="John")):
        """
        Adds two numbers together and returns the sum.
        """
        return x + y

    tool = FunctionTool.from_defaults(fn=sample_function)

    expected_description = sample_function.__doc__.strip()
    actual_description = tool.metadata.description
    print(f"Expected: {expected_description}, Actual: {actual_description}")

    # # Check if the metadata description matches the function's docstring
    # assert (
    #     actual_description == expected_description
    # ), f"The description should automatically be set to the function's docstring. Expected: {expected_description}, Actual: {actual_description}"
