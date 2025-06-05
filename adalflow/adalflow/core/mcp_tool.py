"""
MCP (Modular Command Protocol) Tools are function tools defined in a unified standard sharing with different models and services. They are served by an MCP server.

Features:
- The MCPFunctionTool class, which wraps MCP tools as FunctionTool instances for use in agent workflows.
- The MCPClientManager class, which manages multiple MCP server connections, loads server configurations from JSON, lists available resources/tools/prompts, and provides all tools as FunctionTool instances.

The module enables dynamic discovery and invocation of MCP tools for agent-based workflows.
"""

import os
import json
from adalflow.core.func_tool import FunctionTool
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from contextlib import asynccontextmanager
import logging
from typing import Union, List, Any

from adalflow.core.component import Component
from adalflow.utils.logger import printc
from adalflow.core.types import FunctionDefinition, FunctionOutput, Function

log = logging.getLogger(__name__)

MCPServerParameters = Union[StdioServerParameters, str]


@asynccontextmanager
async def mcp_session_context(server_params: MCPServerParameters):
    """
    Asynchronous context manager for establishing an MCP (Modular Communication Protocol) session.

    Depending on the type of `server_params`, this function initializes a connection to an MCP server
    either via standard I/O or HTTP streaming, and yields an initialized `ClientSession` object.

    Args:
        server_params (MCPServerParameters or str): Parameters for connecting to the MCP server.
            - If an instance of `StdioServerParameters`, connects via standard I/O.
            - If a string (interpreted as a URL), connects via HTTP streaming.

    Yields:
        ClientSession: An initialized client session for communicating with the MCP server.

    Raises:
        ValueError: If `server_params` is not a supported type.
    """
    """"""
    if isinstance(server_params, StdioServerParameters):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                printc("📡 Initializing connection...", color="magenta")
                await session.initialize()
                printc("✅ Connection established!", color="magenta")
                yield session
    elif isinstance(server_params, str):  # URL
        async with streamablehttp_client(server_params) as (read, write, _):
            async with ClientSession(read, write) as session:
                printc("📡 Initializing connection...", color="magenta")
                await session.initialize()
                printc("✅ Connection established!", color="magenta")
                yield session
    else:
        raise ValueError(
            f"Unsupported server parameters type. Must be StdioServerParameters or URL string. But got {type(server_params)}"
        )


async def execute_mcp_op(
    server_params: MCPServerParameters, tool_name: str, id=None, **params: dict
) -> str:
    """
    Executes an operation using a specified MCP tool within a new client session.

    Args:
        server_params (MCPServerParameters): Parameters required to connect to the MCP server.
        tool_name (str): The name of the tool to execute.
        id (optional): An optional identifier for the operation.
        **params (dict): Additional parameters to pass to the tool.

    Returns:
        str: The textual result of the tool operation, or None if the operation failed.

    Raises:
        Exception: If an error occurs during the tool execution, it is caught and logged, but not re-raised.

    Side Effects:
        Prints the result of the tool operation or an error message to the console, with colored output for emphasis.

    Notes:
        - The function uses an asynchronous context manager to handle the MCP session lifecycle.
        - The result is expected to be accessible via `result.content[0].text`.
    """
    async with mcp_session_context(server_params) as session:
        try:
            result = await session.call_tool(tool_name, params)
            printc(f"{tool_name} {params} = {result.content[0].text}", color="magenta")
        except Exception as e:
            printc(f"❌ Error calling {tool_name} tool: {e}", color="magenta")
    return result.content[0].text if result else None


class MCPFunctionTool(FunctionTool):
    __doc__ = r"""A FunctionTool wrapper for MCP (Modular Command Protocol) tools.

    MCPFunctionTool enables seamless integration of MCP tools into agent workflows by exposing them as FunctionTool instances.
    It automatically translates the `mcp.types.Tool` into a `FunctionTool`.
    It allows dynamic discovery, description, and invocation of MCP tools, making them accessible to LLM-based agents or pipelines.

    Note:

        Different from FunctionTool, MCPFunctionTool only supports `acall` since all
        tools are executed asynchronously in the MCP protocol.

    Args:
        server_params (MCPServerParameters): The parameters required to connect to the MCP server. Could be a mcp.StdioServerParameters instance or a URL string.
        mcp_tool (mcp.types.Tool): The MCP tool instance to be used by this function tool.

    Usage Example:

    .. code-block:: python

        from adalflow.core.mcp_tool import MCPFunctionTool, mcp_session_context, StdioServerParameters

        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )

        async with mcp_session_context(server_params) as session:
            tools = await session.list_tools()
            tool = tools.tools[0]
            mcp_func_tool = MCPFunctionTool(server_params, tool)
            output = await mcp_func_tool.acall(param1="value1")

    Features:
    - Wraps an MCP tool (from an MCP server) as a FunctionTool, providing a standardized interface for agent-based workflows.
    - Automatically generates a FunctionDefinition based on the MCP tool's schema and description.
    - Supports asynchronous execution of the tool via the MCP protocol, using the provided server parameters.
    - Handles both stdio and HTTP-based MCP server connections.
    """

    def __init__(self, server_params: MCPServerParameters, mcp_tool: types.Tool):
        """
        Initialize the MCPFunctionTool with the specified server parameters and MCP tool.
        """
        # set params before calling super().__init__ such that _create_fn_definition can use these info.
        self.server_params = server_params
        self.mcp_tool = mcp_tool
        super().__init__(fn=execute_mcp_op, definition=self._create_fn_definition())

    def _create_fn_definition(self) -> FunctionDefinition:
        """
        Create a FunctionDefinition for the MCP tool.
        This overrides the base class method to customize the function signature and description based on the MCP tool's schema.
        """
        # remove 'title' from function parameters
        func_parameters = {
            k: v for k, v in self.mcp_tool.inputSchema.items() if k != "title"
        }
        func_parameters["properties"] = {
            arg_name: {k: v for k, v in props.items() if k != "title"}
            for arg_name, props in func_parameters["properties"].items()
        }

        # Build the description
        arg_list = [
            f'{arg_name}: {props["type"]}'
            for arg_name, props in func_parameters["properties"].items()
        ]
        signature_str = f"({', '.join(arg_list)})"
        description = f"{self.mcp_tool.name}{signature_str}\n"
        # signature_str: add(a: int, b: int, id=None) -> int

        # TODO I don't understand what the `class` is for in adalflow.
        cls_name = self.mcp_tool.name
        if cls_name:
            description += f"Belongs to class: {cls_name}\n"

        if self.mcp_tool.description:
            description += f"Docstring: {self.mcp_tool.description}\n"

        return FunctionDefinition(
            func_name=self.mcp_tool.name,
            func_desc=description,
            func_parameters=func_parameters,
            class_instance=None,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> FunctionOutput:
        r"""Execute the function asynchronously.

        Need to be called in an async function or using asyncio.run.

        Example:

        .. code-block:: python

            import asyncio
            server_params = StdioServerParameters(
                command="python",  # Command to run the server
                args=["mcp_server.py"],  # Arguments (path to your server script)
                env=None  # Optional environment variables
            )
            # Get tools from the server
            async with mcp_session_context(server_params) as session:
                tools = await session.list_tools()
            async def call_async_function():
                tool_1 = MCPFunctionTool(server_params, tools[0])
                output = await tool_1.acall()

            asyncio.run(call_async_function())
        """
        assert (
            len(args) == 0
        ), "FunctionTool does not support positional arguments, use keyword arguments only"
        args = [self.server_params, self.definition.func_name]
        # Call the parent method to handle common logic
        func_output = await super().acall(*args, **kwargs)

        return FunctionOutput(
            name=func_output.name,
            input=Function(
                name=self.definition.func_name, args=args[2:], kwargs=kwargs
            ),
            output=func_output.output,
            error=func_output.error,
        )

    def bicall(self, *args, **kwargs):
        """This function is not supported in MCPFunctionTool."""
        raise ValueError(
            "MCPFunctionTool does not support bicall. Use acall instead, which is designed for asynchronous execution."
        )

    def call(self, *args, **kwargs):
        """This function is not supported in MCPFunctionTool."""
        raise ValueError(
            "MCPFunctionTool does not support call. Use acall instead, which is designed for asynchronous execution."
        )


class MCPClientManager(Component):
    __doc__ = r"""Manage MCP client connections and resources.

    Example:

    .. code-block:: python

        from adalflow.core.mcp_tool import MCPClientManager, StdioServerParameters

        manager = MCPClientManager()
        # Add servers. Here we used a local stdio server defined in `mcp_server.py`.
        manager.add_server("calculator_server", StdioServerParameters(
            command="python",  # Command to run the server
            args=["mcp_server.py"],  # Arguments (path to your server script)
            env=None  # Optional environment variables
        ))
        await manager.list_all_tools()  # to see all available resources/tools/prompts. But we only use tools.
        tools = await manager.get_all_tools()
        # Add tools to agent
        react = ReActAgent(
            max_steps=6,
            add_llm_as_fallback=True,
            tools=tools,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
    """

    def __init__(self):
        self.server_params = {}

    def add_servers_from_json_file(self, json_path: str):
        """Read MCP server configurations from a JSON file and add them to the manager.

        Args:
            json_path (str): Path to the JSON file containing MCP server configurations.

        Example of JSON structure:

        .. code-block:: json

            {
                "mcpServers": {
                    "mcp_weather_server": {
                        "command": "python",
                        "args": [
                            "mcp_weather_server.py"
                        ]
                    }
                }
            }
        """
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                config = json.load(f)
            mcp_servers = config.get("mcpServers", {})
            for name, params in mcp_servers.items():
                self.add_server(
                    name,
                    StdioServerParameters(
                        command=params.get("command"),
                        args=params.get("args", []),
                        env=params.get("env", None),
                    ),
                )
        else:
            print(f"Warning: {json_path} not found. No additional servers loaded.")

    def add_server(self, name: str, server_params: MCPServerParameters):
        """Adds a new MCP server to the internal server registry.

        Parameters:
            name (str): The unique identifier for the server to be added.
            server_params (MCPServerParameters): An object containing the configuration parameters for the server. Could be a StdioServerParameters instance or a URL string.

        Features:
            - If a server with the specified name does not already exist in the registry, it is added and a confirmation message is printed.
            - If a server with the specified name already exists, the addition is skipped and a ValueError is raised.

        Raises:
            ValueError: If a server with the specified name already exists in the registry.
        """
        if name not in self.server_params:
            print(f"Adding server: {name}")
            self.server_params[name] = server_params
        else:
            raise ValueError(
                f"Server {name} already exists. Cannot override existing server configuration."
            )

    async def list_all_tools(self):
        """
        List all available resources, tools, and prompts from all added servers.
        """
        if not self.server_params:
            print("No servers added. Please add a server first.")
            return

        for name, params in self.server_params.items():
            print(f"\nListing tools for server: {name}")
            await self._list_all_server_tools(params)

    async def _list_all_server_tools(self, server_params: MCPServerParameters):
        async with mcp_session_context(server_params) as session:
            # List available resources
            print("\n🗂️  Available Resources:")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  • {resource.name}: {resource.description}")
                print(f"    URI: {resource.uri}")

            # List available tools
            print("\n🔧 Available Tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  • {tool.name}: {tool.description}")

            # List available prompts
            print("\n📝 Available Prompts:")
            prompts = await session.list_prompts()
            for prompt in prompts.prompts:
                print(f"  • {prompt.name}: {prompt.description}")

    async def get_all_tools(self) -> List[FunctionTool]:
        """
        Get all available functions from added servers as FunctionTool instances.
        """
        tools = []
        for name, params in self.server_params.items():
            print(f"\n🔧 Getting Tools from server {name}:")
            # get all tools from the server
            tools.extend(await self._get_all_server_tools(params))
        return tools

    async def _get_all_server_tools(self, server_params) -> List[FunctionTool]:
        """
        Get all available tools from all added servers.
        """
        tools = []
        async with mcp_session_context(server_params) as session:
            # List available tools
            _tools = await session.list_tools()
            for tool in _tools.tools:
                print(f"  • {tool.name}: {tool.description}")
                tools.append(tool)

        return [MCPFunctionTool(server_params, t) for t in tools]
