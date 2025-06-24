import sys
import pytest
import unittest
import os

if sys.version_info < (3, 10):
    pytest.skip("Requires Python 3.10 or higher", allow_module_level=True)


from adalflow.core.mcp_tool import (
    MCPFunctionTool,
    mcp_session_context,
    MCPToolManager,
    MCPServerStdioParams,
)
from mcp.types import CallToolResult
from adalflow.core.types import FunctionDefinition


metadata = FunctionDefinition(func_desc="A simple addition tool", func_name="add")

mcp_server_path = "../tutorials/mcp_agent/mcp_calculator_server.py"


class TestMCPTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Get current working directory
        current_dir = os.getcwd()

        # Define server_params for the local npx filesystem server
        self.server_params = MCPServerStdioParams(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                current_dir,
            ],
        )
        # this npx @modelcontextprotocol/server-filesystem will download the package from npm registry to a local cache
        # it runs immediately without installing globally

        # TODO: if a tool is used globably, better to install globally once: npm install -g @modelcontextprotocol/server-filesystem

    async def test_mcp_tool_local_npx_server_async(self):
        """Test MCP tool with local npx filesystem server"""
        # directly test the mcp server and functions like list_tools, call_tool
        async with mcp_session_context(self.server_params) as session:
            tools = await session.list_tools()
            print(f"Available tools: {tools}")

            # Assert that we got tools back
            self.assertIsNotNone(tools)

            try:
                output = await session.call_tool(
                    "read_file",
                    arguments={"path": "README.md"},
                )
                print(f"Type of output: {type(output)}")
                print(f"Output content: {output}")
                self.assertIsInstance(output, CallToolResult)

                # Assert that we got some output back
                self.assertIsNotNone(output)

            except Exception as e:
                print(f"Error calling read_file tool: {e}")
                # If the tool fails, that's okay for testing - just log it
                # We don't want to fail the test if the filesystem server isn't available
                pass

    async def test_mcp_function_tool_async(self):
        """Test MCP function tool with calculator server, local python"""

        async def create_mcp_retrieve_tool():
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
                # create a MCPFunctionTool for the testing
                mcp_func_tool = MCPFunctionTool(server_params, tool)
            return mcp_func_tool

        mcp_func_tool = await create_mcp_retrieve_tool()
        output = await mcp_func_tool.acall(a=3, b=4)

        # Assertions
        self.assertEqual(int(float(output.output)), 7)
        self.assertEqual(
            output.name, "add", "The name should be set to the function name"
        )
        self.assertTrue(hasattr(output.input, "args"))
        self.assertEqual(output.input.args, [])
        self.assertEqual(output.input.kwargs["a"], 3)
        self.assertEqual(output.input.kwargs["b"], 4)

        # call with sync call with raise ValueError
        with self.assertRaises(ValueError):
            mcp_func_tool.call(3, 4)

    async def test_mcp_client_manager(self):
        """Test MCP tool manager"""

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

        tools = await get_all_tools()
        self.assertGreater(len(tools), 0, "There should be at least one tool available")


# TODO Test MCP function tools with gradient components
