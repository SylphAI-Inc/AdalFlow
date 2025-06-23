import pytest
import asyncio

from adalflow.core.mcp_tool import (
    MCPFunctionTool,
    mcp_session_context,
    MCPToolManager,
    MCPServerStdioParams,
)
from adalflow.core.types import FunctionDefinition


metadata = FunctionDefinition(func_desc="A simple addition tool", func_name="add")

mcp_server_path = "../tutorials/mcp_agent/mcp_calculator_server.py"


def test_function_tool_async():
    async def retrieve_tool():
        # use default metadata
        server_params = MCPServerStdioParams(
            command="python", args=[mcp_server_path], env=None
        )

        async with mcp_session_context(server_params) as session:
            tools = await session.list_tools()
            tools = tools.tools
            for tool in tools:
                if tool.name == "add":
                    break
            mcp_func_tool = MCPFunctionTool(server_params, tool)
        return mcp_func_tool

    mcp_func_tool = asyncio.run(retrieve_tool())
    output = asyncio.run(mcp_func_tool.acall(a=3, b=4))
    assert int(float(output.output)) == 7
    assert output.name == "add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == []
    assert output.input.kwargs["a"] == 3
    assert output.input.kwargs["b"] == 4

    # call with sync call with raise ValueError
    with pytest.raises(ValueError):
        mcp_func_tool.call(3, 4)


def test_mcp_client_manager():
    async def get_all_tools():
        manager = MCPToolManager()
        # Add servers. Here we used a local stdio server defined in `mcp_server.py`.
        manager.add_server(
            "calculator_server",
            MCPServerStdioParams(
                command="python",  # Command to run the server
                args=[mcp_server_path],  # Arguments (path to your server script)
                env=None,  # Optional environment variables
            ),
        )
        # to see all available resources/tools/prompts. But we only use tools.
        await manager.list_all_tools()
        return await manager.get_all_tools()

    tools = asyncio.run(get_all_tools())
    assert len(tools) > 0, "There should be at least one tool available"


# TODO Test MCP function tools with gradient components
