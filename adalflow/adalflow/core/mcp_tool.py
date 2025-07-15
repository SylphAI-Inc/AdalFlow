"""
MCP (Modular Command Protocol) Tools are function tools defined in a unified standard sharing with different models and services. They are served by an MCP server.

Features:
- The MCPFunctionTool class, which wraps MCP tools as FunctionTool instances for use in agent workflows.
- The MCPToolManager class, which manages multiple MCP server connections, loads server configurations from JSON, lists available resources/tools/prompts, and provides all tools as FunctionTool instances.

The module enables dynamic discovery and invocation of MCP tools for agent-based workflows.
"""

import os
import json
import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union
from adalflow.core.func_tool import FunctionTool
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from contextlib import asynccontextmanager
import logging
from typing import List, Any, Literal
from dataclasses import dataclass, field

from adalflow.core.component import Component
from adalflow.utils.logger import printc
from adalflow.core.types import FunctionDefinition, FunctionOutput, Function

log = logging.getLogger(__name__)


@dataclass
class MCPServerStdioParams:
    r"""Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another import."""

    command: str = field(
        metadata={
            "desc": "The executable to run to start the server. For example, `python` or `node`."
        }
    )
    args: Optional[list[str]] = field(
        default=None,
        metadata={
            "desc": "Command line args to pass to the `command` executable. For example, `['foo.py']` or `['server.js', '--port', '8080']`."
        },
    )
    env: Optional[dict[str, str]] = field(
        default=None,
        metadata={"desc": "The environment variables to set for the server."},
    )
    cwd: Optional[Union[str, Path]] = field(
        default=None,
        metadata={"desc": "The working directory to use when spawning the process."},
    )
    encoding: Optional[str] = field(
        default="utf-8",
        metadata={
            "desc": "The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`."
        },
    )
    encoding_error_handler: Optional[Literal["strict", "ignore", "replace"]] = field(
        default="strict",
        metadata={
            "desc": "The text encoding error handler. Defaults to `strict`. See https://docs.python.org/3/library/codecs.html#codec-base-classes for explanations of possible values."
        },
    )


@dataclass
class MCPServerSseParams:
    """
    Mirrors the params in `mcp.client.sse.sse_client`.
    """

    url: str = field(metadata={"desc": "The URL of the server."})
    headers: Optional[dict[str, str]] = field(
        default=None, metadata={"desc": "The headers to send to the server."}
    )
    timeout: Optional[float] = field(
        default=5,
        metadata={"desc": "The timeout for the HTTP request. Defaults to 5 seconds."},
    )
    sse_read_timeout: Optional[float] = field(
        default=60 * 5,
        metadata={
            "desc": "The timeout for the SSE connection, in seconds. Defaults to 5 minutes."
        },
    )


@dataclass
class MCPServerStreamableHttpParams:
    """
    Mirrors the params in `mcp.client.streamable_http.streamablehttp_client`.
    """

    url: str = field(metadata={"desc": "The URL of the server."})
    headers: Optional[dict[str, str]] = field(
        default=None, metadata={"desc": "The headers to send to the server."}
    )
    timeout: Optional[timedelta] = field(
        default=timedelta(seconds=30),
        metadata={"desc": "The timeout for the HTTP request. Defaults to 30 seconds."},
    )
    sse_read_timeout: Optional[timedelta] = field(
        default=timedelta(seconds=60 * 5),
        metadata={
            "desc": "The timeout for the SSE connection, in seconds. Defaults to 5 minutes."
        },
    )
    terminate_on_close: Optional[bool] = field(
        default=True, metadata={"desc": "Terminate on close"}
    )


MCPServerParameters = Union[
    MCPServerStdioParams, MCPServerSseParams, MCPServerStreamableHttpParams
]


# NOTE: mcp_session_context only works with one server at a time.
@asynccontextmanager
async def mcp_session_context(
    server_params: MCPServerParameters,
    name: Optional[str] = None,
):
    """
    Asynchronous context manager for establishing an MCP (Modular Communication Protocol) session.

    Depending on the type of `server_params`, this function initializes a connection to an MCP server
    either via standard I/O or HTTP streaming, and yields an initialized `ClientSession` object.

    Args:
        server_params (MCPServerParameters): Parameters for connecting to the MCP server.
            - If an instance of `StdioServerParameters`, connects via standard I/O.
            - If a string (interpreted as a URL), connects via HTTP streaming.

    Yields:
        ClientSession: An initialized client session for communicating with the MCP server.

    Raises:
        ValueError: If `server_params` is not a supported type.
    """
    msg = (
        f"📡 Initializing connection to {name}..."
        if name
        else "📡 Initializing connection..."
    )

    if isinstance(server_params, MCPServerStdioParams):
        async with stdio_client(
            StdioServerParameters(
                command=server_params.command,
                args=server_params.args,
                env=server_params.env,
                cwd=server_params.cwd,
                encoding=server_params.encoding,
                encoding_error_handler=server_params.encoding_error_handler,
            )
        ) as (read, write):
            async with ClientSession(read, write) as session:
                printc(msg, color="magenta")
                await session.initialize()
                printc("✅ Connection established!", color="magenta")
                yield session
    elif isinstance(server_params, MCPServerStreamableHttpParams):  # URL
        async with streamablehttp_client(
            url=server_params.url,
            headers=server_params.headers,
            timeout=server_params.timeout,
            sse_read_timeout=server_params.sse_read_timeout,
            terminate_on_close=server_params.terminate_on_close,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:

                printc(msg, color="magenta")
                await session.initialize()
                printc("✅ Connection established!", color="magenta")
                yield session
    elif isinstance(server_params, MCPServerSseParams):  # URL
        async with sse_client(
            url=server_params.url,
            headers=server_params.headers,
            timeout=server_params.timeout,
            sse_read_timeout=server_params.sse_read_timeout,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                printc(msg, color="magenta")
                await session.initialize()
                printc("✅ Connection established!", color="magenta")
                yield session
    else:
        raise ValueError(
            f"Unsupported server parameters type. Must be one of MCPServerStdioParams, MCPServerSseParams, MCPServerStreamableHttpParams. But got {type(server_params)}"
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
        if not isinstance(mcp_tool, types.Tool):
            raise ValueError("mcp_tool must be an instance of mcp.types.Tool")
        self.server_params = server_params
        self.mcp_tool = mcp_tool
        super().__init__(fn=execute_mcp_op, definition=self._create_fn_definition())

    # NOTE: dont support optional in the data class
    def _create_fn_definition(self) -> FunctionDefinition:
        """
        Create a FunctionDefinition for the MCP tool.
        This overrides the base class method to customize the function signature and description based on the MCP tool's schema.
        """
        name = self.mcp_tool.name
        description = self.mcp_tool.description
        schema = f"input schema: {self.mcp_tool.inputSchema}"

        # the outputSchema may not be available https://modelcontextprotocol.io/specification/draft/server/tools#output-schema
        if (
            hasattr(self.mcp_tool, "outputSchema")
            and self.mcp_tool.outputSchema is not None
        ):
            schema += f"\noutput schema: {self.mcp_tool.outputSchema}"

        return FunctionDefinition(
            func_name=name,
            func_desc=description,
            func_parameters=schema,
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

    def call(self, *args, **kwargs) -> FunctionOutput:
        """Execute the function synchronously by running the async call method.

        This is a convenience method that wraps the async `acall` method using `asyncio.run()`.
        It allows synchronous usage of MCP tools without requiring async/await syntax.

        Args:
            *args: Positional arguments (not supported, raises assertion error)
            **kwargs: Keyword arguments to pass to the MCP tool

        Returns:
            FunctionOutput: The result of the MCP tool execution

        Example:

        .. code-block:: python

            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"]
            )
            # Get tools from the server
            async with mcp_session_context(server_params) as session:
                tools = await session.list_tools()

            tool_1 = MCPFunctionTool(server_params, tools[0])
            output = tool_1.call(param1="value1")  # Synchronous call
        """
        return asyncio.run(self.acall(*args, **kwargs))


class MCPToolManager(Component):
    __doc__ = r"""Manage MCP client connections and resources.

    Example:

    .. code-block:: python

        from adalflow.core.mcp_tool import MCPToolManager, StdioServerParameters

        manager = MCPToolManager()
        # Add servers. Here we used a local stdio server defined in `mcp_server.py`.
        # you can add more servers by calling `add_server` for multiple times.
        manager.add_server("calculator_server", StdioServerParameters(
            command="python",  # Command to run the server
            args=["mcp_server.py"],  # Arguments (path to your server script)
        ))
        manager.add_server("weather_server", StdioServerParameters(
            command="python",  # Command to run the server
            args=["mcp_weather_server.py"],  # Arguments (path to your weather server script)
        ))
        await manager.list_all_tools()  # to see all available tools from all servers.
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

    def __init__(
        self,
        cache_tools_list: bool = True,
        client_session_timeout_seconds: Optional[float] = None,
    ):
        self.server_params = {}
        self.cache_tools_list = cache_tools_list

        self.client_session_timeout_seconds = client_session_timeout_seconds

        # The cache is always dirty at startup, so that we fetch tools at least once
        self._tools_list: Optional[list[MCPFunctionTool]] = []
        self._cached_servers: list[str] = []

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
            self._cache_dirty = True  # Mark cache as dirty since we added a new server
        else:
            raise ValueError(
                f"Server {name} already exists. Cannot override existing server configuration."
            )

    async def list_all_tools(self, server_names: List[str] = None):
        """
        Print out all available resources, tools, and prompts from all added servers.
        If `server_names` is provided, only list tools for those specific servers.

        Args:
            server_names (List[str], optional): A list of server names to filter the tools.
                If None, all servers are listed.
        """
        if not self.server_params:
            print("No servers added. Please add a server first.")
            return

        for name, params in self.server_params.items():
            if server_names and name not in server_names:
                continue
            print(f"\nListing tools for server: {name}")
            await self._list_all_server_tools(params)

    async def _list_all_server_tools(self, server_params: MCPServerParameters):
        async with mcp_session_context(server_params) as session:
            # # List available resources
            # print("\n🗂️  Available Resources:")
            # resources = await session.list_resources()
            # for resource in resources.resources:
            #     print(f"  • {resource.name}: {resource.description}")
            #     print(f"    URI: {resource.uri}")

            # List available tools
            print("\n🔧 Available Tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  • {tool.name}: {tool.description}")

            # # List available prompts
            # print("\n📝 Available Prompts:")
            # prompts = await session.list_prompts()
            # for prompt in prompts.prompts:
            #     print(f"  • {prompt.name}: {prompt.description}")

    async def get_all_tools(
        self, server_names: List[str] = None
    ) -> List[MCPFunctionTool]:
        """
        Get all available functions from added servers as FunctionTool instances.
        If `server_names` is provided, only list tools for those specific servers.

        Args:
            server_names (List[str], optional): A list of server names to filter the tools.
                If None, all servers are listed.
        """
        # TODO: this is not good implementation, it establishes two times of connections each time.
        for name, params in self.server_params.items():
            if server_names and name not in server_names:
                continue
            if name in self._cached_servers:
                print(f"🔧 Using cached tools for server {name}.")
                continue

            print(f"\n🔧 Getting Tools from server {name}:")
            # get all tools from the server
            try:
                self._tools_list.extend(await self._get_all_server_tools(params))
            except Exception as e:
                print(f"Error getting tools from server {name}: {e}")
                continue
            self._cached_servers.append(name)
        return self._tools_list

    async def _get_all_server_tools(self, server_params) -> List[MCPFunctionTool]:
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

        try:
            return [MCPFunctionTool(server_params, t) for t in tools]
        except Exception as e:
            print(f"Error getting tools from server: {e}")
            return []
