.. raw:: html

    <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
        <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/tool_simple.py" target="_blank" style="display: flex; align-items: center;">
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

- :class:`FunctionTool <core.func_tool.FunctionTool>`: Wraps any Python function or class/component method as a tool for LLMs.
- :class:`MCPFunctionTool <core.mcp_tool.MCPFunctionTool>`: Wraps tools served by an MCP (Modular Command Protocol) server, enabling dynamic tool discovery and invocation.

Introduction
------------

A "tool" is any callable function or service that an LLM agent can invoke to perform a specific task, such as retrieving information, performing calculations, or interacting with external APIs. 
Tools are essential for building agentic workflows, enabling LLMs to go beyond text generation.

FunctionTool: Wrapping Tools
--------------------------------------

:class:`FunctionTool <core.func_tool.FunctionTool>` make it easy for developers by providing:

- **Automatic Schema Generation**: Creates function definitions from docstrings and type hints for LLM understanding
- **Unified Interface**: It always output :class:`FunctionOutput <core.types.FunctionOutput>` where the initial tool output is stored in the `output` field.
- **Type Support**: Provides consistent ``call()`` and ``acall()`` methods regardless of the underlying function type, be it sync, async, generator, or async generator.
.. - **Training Integration**: Supports gradient flow when wrapping trainable components
- **Error Handling**: If the function call fails, the error will be stored in the `error` field of the :class:`FunctionOutput <core.types.FunctionOutput>`.

**Basic Example:**

.. code-block:: python

    from adalflow.core.func_tool import FunctionTool

    def add(a: int, b: int) -> int:
        """Add two numbers."""  # This docstring becomes the tool description for LLM
        return a + b

    # Wrap the function
    add_tool = FunctionTool(add)
    
    # Execute the tool
    result = add_tool.call(2, 3)
    print(result.output)  # Output: 5
    print(result.name)    # Output: "add"

The complete `FunctionOutput` print is:

.. code-block:: 

    FunctionOutput(name='add', input=Function(id=None, thought=None, name='add', args=(2, 3), kwargs={}, _is_answer_final=None, _answer=None), parsed_input=None, output=5, error=None)



**Supported Function Types:**

FunctionTool supports four core function types, automatically detected and handled appropriately:

1. **Synchronous Functions** (Regular Python functions):

.. code-block:: python

    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle."""
        return length * width
    
    area_tool = FunctionTool(calculate_area)
    result = area_tool.call(5.0, 3.0)
    print(result.output)  # Output: 15.0

2. **Asynchronous Functions** (Async/await functions):

.. code-block:: python

    async def fetch_data(url: str) -> dict:
        """Fetch data from a URL asynchronously."""
        # Simulate async operation
        await asyncio.sleep(1)
        return {"data": f"Content from {url}"}
    
    fetch_tool = FunctionTool(fetch_data)
    # Use acall for async functions
    result = await fetch_tool.acall("https://api.example.com")
    print(result.output)  # Output: {"data": "Content from https://api.example.com"}

3. **Synchronous Generators** (Functions that yield values):

.. code-block:: python

    def count_to_n(n: int):
        """Count from 1 to n, yielding each number."""
        for i in range(1, n + 1):
            yield i
    
    counter_tool = FunctionTool(count_to_n)
    result = counter_tool.call(5)
    # For generators, output contains the generator object
    for num in result.output:
        print(num)  # Outputs: 1, 2, 3, 4, 5

4. **Asynchronous Generators** (Async functions that yield):

.. code-block:: python

    async def stream_updates(source: str):
        """Stream updates from a source."""
        for i in range(3):
            await asyncio.sleep(0.5)
            yield f"Update {i} from {source}"
    
    stream_tool = FunctionTool(stream_updates)
    result = await stream_tool.acall("sensor1")
    async for update in result.output:
        print(update)  # Outputs updates over time

**Advanced Examples:**

**Class Methods and Component Integration:**

.. code-block:: python

    from adalflow.core import Component
    
    class DataProcessor(Component):
        def __init__(self):
            super().__init__()
            self.preprocessing_steps = ["normalize", "clean"]
        
        def process_text(self, text: str) -> str:
            """Process text through predefined steps."""
            # Access instance attributes
            for step in self.preprocessing_steps:
                text = f"[{step}] {text}"
            return text
    
    processor = DataProcessor()
    # Wrap instance method - maintains access to self
    process_tool = FunctionTool(processor.process_text)
    result = process_tool.call("Hello World")
    print(result.output)  # Output: "[normalize] [clean] Hello World"

**Working with Complex Types:**

.. code-block:: python

    from dataclasses import dataclass
    from typing import List
    import numpy as np
    
    @dataclass
    class Point:
        x: float
        y: float
    
    def calculate_centroid(points: List[Point]) -> Point:
        """Calculate the centroid of a list of points."""
        if not points:
            return Point(0, 0)
        avg_x = sum(p.x for p in points) / len(points)
        avg_y = sum(p.y for p in points) / len(points)
        return Point(avg_x, avg_y)
    
    # FunctionTool handles complex parameter types
    centroid_tool = FunctionTool(calculate_centroid)
    points = [Point(0, 0), Point(2, 0), Point(1, 2)]
    result = centroid_tool.call(points)
    print(result.output)  # Output: Point(x=1.0, y=0.667)

**Using ToolOutput for Enhanced Control:**

We use :class:`ToolOutput <core.types.ToolOutput>` with four important fields:

- `output`: The actual output of the tool. Can be error message if the tool call fails.
- `observation`: The observation of the tool seen by LLM agent. Can be error message if the tool call fails.
- `display`: The display of the tool to users. Can be error message if the tool call fails.
- `metadata`: Any additional metadata you want to save
- `status`: The status of the tool call, can be "success", "cancelled", or "error". Important for the frontend to display the correct status.

.. code-block:: python

    from adalflow.core.types import ToolOutput
    
    def analyze_sentiment(text: str) -> ToolOutput:
        """Analyze sentiment with detailed feedback."""
        # Simulate analysis
        score = 0.8 if "happy" in text.lower() else 0.2
        
        return ToolOutput(
            output={"sentiment": "positive" if score > 0.5 else "negative", "score": score},
            observation=f"Sentiment analysis complete. Score: {score}",
            display=f"😊 Positive ({score:.0%})" if score > 0.5 else f"😢 Negative ({score:.0%})",
            metadata={"model": "simple-rule-based", "confidence": "low"}
        )
    
    sentiment_tool = FunctionTool(analyze_sentiment)
    result = sentiment_tool.call("I am very happy today!")
    print(result.output)       # The actual data
    print(result.observation)  # For agent reasoning
    print(result.display)      # For user display

**Error Handling:**

When error is encountered, it is tracked in `error` field. 
This makes it easy for agent to auto-recover in the later steps.

.. code-block:: python

    def divide(a: float, b: float) -> float:
        """Divide two numbers safely."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    divide_tool = FunctionTool(divide)
    
    # Successful call
    result = divide_tool.call(10, 2)
    print(result.output)  # Output: 5.0
    print(result.error)   # Output: None
    
    # Error case
    result = divide_tool.call(10, 0)
    print(result.output)  # Output: "Error: Cannot divide by zero"
    print(result.error)   # Contains the actual exception

To see how FunctionTool is used in agent workflows and integrated with the Agent and Runner components, refer to the :doc:`Agents and Runner <agents_runner>` documentation.


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

The :class:`MCPToolManager <core.mcp_tool.MCPToolManager>` helps manage multiple MCP servers and aggregate all available tools for agent workflows.

.. code-block:: python

    from core.mcp_tool import MCPToolManager, MCPServerStdioParams

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

    - :class:`core.func_tool.FunctionTool`
    - :class:`core.mcp_tool.MCPFunctionTool`
    - :class:`core.mcp_tool.MCPToolManager`
