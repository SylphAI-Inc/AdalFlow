import pytest
from dataclasses import dataclass

from adalflow.core.func_tool import FunctionTool
from adalflow.core.tool_manager import ToolManager
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


from adalflow.optim.grad_component import GradComponent


class GradAdd(GradComponent):
    def __init__(self):
        super().__init__()
        print(f"training: {self.training}")

    def call(self, x, y):
        return x + y

    def forward(self, x, y):
        print(f"training: {self.training}")
        return f"{x + y} + forward"


class GradSub(GradComponent):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return x - y

    def forward(self, x, y):
        print(f"training: {self.training}")
        return f"{x - y} + forward"


class TestComponent(GradComponent):
    def __init__(self):
        super().__init__()

        self.add = GradAdd()
        self.sub = GradSub()

        print(f"sub_component: {self.sub.training}")

        print(f"add_component: {self.add.training}")

        def add_as_tool(x, y):
            return self.add(x, y)

        self.tools = [
            FunctionTool(fn=add_as_tool, component=self.add),
            FunctionTool(fn=self.sub.__call__, component=self.sub),
        ]


add = GradAdd()
sub = GradSub()


class TestComponnetInstanceOutsideComponent(GradComponent):
    def __init__(self):
        super().__init__()

        print(f"sub_component: {sub.training}")

        print(f"add_component: {add.training}")

        def add_as_tool(x, y):
            return add(x, y)

        self.tools = [
            FunctionTool(fn=add_as_tool, component=add),
            FunctionTool(fn=sub.__call__, component=sub),
        ]


class TestToolManagerComponent(GradComponent):

    def __init__(self):
        super().__init__()

        print(f"sub_component: {sub.training}")

        print(f"add_component: {add.training}")

        def add_as_tool(x, y):
            return add(x, y)

        self.tools = [
            FunctionTool(fn=add_as_tool, component=add),
            FunctionTool(fn=sub.__call__, component=sub),
        ]

        # manag by tool manager, and since the component is passed to tools_manager which is also a component, it will be in training mode
        self.tools_manager = ToolManager(tools=self.tools)


def test_function_tool_with_grad_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert test_com.add.training
    # ensure it is the forward method that is called
    output = test_com.tools[0](1, 2)
    assert output.output == "3 + forward"


def test_component_instance_outside_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestComponnetInstanceOutsideComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert not add.training  # the subcomponent is no longer in training mode
    # ensure it is the forward method that is called
    output = test_com.tools[0](1, 2)
    assert output.output == 3


def test_tool_manager_with_grad_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestToolManagerComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools_manager.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert (
        add.training
    )  # the subcomponent will change as it is managed by the tool manager
    # ensure it is the forward method that is called
    output = test_com.tools_manager.tools[0](1, 2)
    assert output.output == "3 + forward"
