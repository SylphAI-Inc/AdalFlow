.. raw:: html

    <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
        <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/tools.py" target="_blank" style="display: flex; align-items: center;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
            <span style="vertical-align: middle;"> Open Source Code (Tool)</span>
        </a>
        <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/mcp_agent/react_agent.py" target="_blank" style="display: flex; align-items: center;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
            <span style="vertical-align: middle;"> Open Source Code (MCP Tool Agent)</span>
        </a>
    </div>

Tool Use
====================================

AdalFlow provides a unified interface for integrating external tools into LLM workflows. Tools extend the capabilities of LLM agents, enabling them to call functions, access APIs, or interact with external systems. Two main tool abstractions are provided:

- :class:`FunctionTool <adalflow.core.func_tool.FunctionTool>`: Wraps any Python function or component method as a tool for LLMs.
- :class:`MCPFunctionTool <adalflow.core.mcp_tool.MCPFunctionTool>`: Wraps tools served by an MCP (Modular Command Protocol) server, enabling dynamic tool discovery and invocation.

Introduction
------------

A "tool" is any callable function or service that an LLM agent can invoke to perform a specific task, such as retrieving information, performing calculations, or interacting with external APIs. Tools are essential for building agentic workflows, enabling LLMs to go beyond text generation.

FunctionTool: Wrapping Python Functions
--------------------------------------

The :class:`FunctionTool` class allows you to wrap any Python function or component method as a tool. This standardizes the interface and metadata, making it easy for agents to discover and invoke tools.

.. code-block:: python

    from adalflow.core.func_tool import FunctionTool

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add_tool = FunctionTool(add)
    result = add_tool.call(2, 3)
    print(result.output)  # Output: 5

You can also wrap component methods, including trainable components, as tools. This enables seamless integration with the AdalFlow pipeline and supports both synchronous and asynchronous execution.

.. note::

    FunctionTool supports both synchronous (``call``) and asynchronous (``acall``) functions.

MCPFunctionTool: Integrating MCP Tools
--------------------------------------

MCP (Modular Command Protocol) enables dynamic discovery and invocation of tools served by external servers. The :class:`MCPFunctionTool` class wraps these tools, exposing them as FunctionTool instances for agent workflows.

.. code-block:: python

    from adalflow.core.mcp_tool import MCPFunctionTool, mcp_session_context, MCPServerStdioParams

    server_params = MCPServerStdioParams(
        command="python",
        args=["mcp_server.py"],
        env=None
    )

    async with mcp_session_context(server_params) as session:
        tools = await session.list_tools()
        tool = tools.tools[0]
        mcp_tool = MCPFunctionTool(server_params, tool)
        output = await mcp_tool.acall(param1="value1")
        print(output.output)

MCPFunctionTool only supports asynchronous execution (``acall``), as all MCP tools are invoked asynchronously.

**Managing Multiple MCP Servers**

The :class:`MCPToolManager <adalflow.core.mcp_tool.MCPToolManager>` helps manage multiple MCP servers and aggregate all available tools for agent workflows.

.. code-block:: python

    from adalflow.core.mcp_tool import MCPToolManager, MCPServerStdioParams

    manager = MCPToolManager()
    manager.add_server("calculator_server", MCPServerStdioParams(
        command="python",
        args=["mcp_server.py"],
        env=None
    ))
    tools = await manager.get_all_tools()
    # Use tools in your agent pipeline

**MCPFunctionTool: Using a URL-based MCP Server**

You can also connect to a remote MCP server via SSE by passing a URL string as the server parameter. This enables integration with cloud-hosted or containerized tool servers.

.. code-block:: python

    from adalflow.core.mcp_tool import MCPFunctionTool, mcp_session_context

    # Example: connect to a remote MCP server via SSE
    smithery_api_key = os.environ.get("SMITHERY_API_KEY")
    smithery_server_id = "@nickclyde/duckduckgo-mcp-server"
    server_url = f"https://server.smithery.ai/{smithery_server_id}/mcp?api_key={smithery_api_key}"

    async with mcp_session_context(server_url) as session:
        tools = await session.list_tools()
        tool = tools.tools[0]
        mcp_tool = MCPFunctionTool(server_url, tool)
        output = await mcp_tool.acall(param1="value1")
        print(output.output)

.. note::

    The MCP protocol supports both local (stdio) and remote (HTTP) tool servers. You can mix and match them in your workflow.

References
----------

.. [1] MCP Protocol: https://github.com/SylphAI-Inc/mcp
.. [2] AdalFlow FunctionTool: https://adalflow.sylph.ai/apis/core/adalflow.core.func_tool.html
.. [3] AdalFlow MCPFunctionTool: https://adalflow.sylph.ai/apis/core/adalflow.core.mcp_tool.html

.. admonition:: API References
    :class: highlight

    - :class:`adalflow.core.func_tool.FunctionTool`
    - :class:`adalflow.core.mcp_tool.MCPFunctionTool`
    - :class:`adalflow.core.mcp_tool.MCPToolManager`
