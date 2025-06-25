from adalflow.core.mcp_tool import (
    MCPToolManager,
    MCPServerStdioParams,
    mcp_session_context,
)
from adalflow.components.agent import ReActAgent
from adalflow.utils import setup_env
import asyncio
import adalflow as adal
import os

if __name__ == "__main__":
    # Get current working directory for relative paths
    current_dir = os.getcwd()

    # Use local Python weather server instead of npx filesystem server
    weather_server_params = MCPServerStdioParams(
        command="python",
        args=["tutorials/mcp_agent/mcp_weather_server.py"],
        env=None,
    )

    # 1. Test the weather tool directly from MCP connection
    async def weather_ops_tool():
        async with mcp_session_context(
            weather_server_params, name="weather_ops"
        ) as session:
            tools = await session.list_tools()
            print("Available weather tools:", tools)
            try:
                # Test getting weather alerts for California
                output = await session.call_tool(
                    "get_alerts", arguments={"state": "CA"}
                )
                print("Type of output:", type(output))
                print("Weather alerts output:", output)
                return output
            except Exception as e:
                print(f"Error calling weather tool: {e}")
                return None

    print("=== Testing Weather MCP Server Directly ===")
    output = asyncio.run(weather_ops_tool())
    print(f"Weather output: {output}")

    # 2. Test the MCP tool manager with the weather server
    setup_env()

    gpt_model_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    async def test_weather_mcp_agent():
        tool_manager = MCPToolManager()
        tool_manager.add_server(
            name="weather_server",
            server_params=weather_server_params,
        )
        tools = await tool_manager.get_all_tools()
        print(f"Available tools from manager: {len(tools)} tools")

        react = ReActAgent(
            max_steps=6,
            add_llm_as_fallback=True,
            tools=tools,
            model_client=adal.OpenAIClient(),
            model_kwargs=gpt_model_kwargs,
            debug=True,
        )

        # Test weather-related queries
        queries = [
            "What are the current weather alerts in California?",
            "Get the weather forecast for San Francisco (latitude: 37.7749, longitude: -122.4194)",
        ]

        for query in queries:
            print(f"\n=== Testing Query: {query} ===")
            output = react.call(query)
            print(f"Agent output: {output}")
            final_answer = output.answer
            print(f"Final answer: {final_answer}")

    print("\n=== Testing Weather MCP Agent ===")
    asyncio.run(test_weather_mcp_agent())

    # 3. Test with multiple local Python servers
    async def test_multiple_servers():
        tool_manager = MCPToolManager()

        # Add weather server
        tool_manager.add_server(
            name="weather_server",
            server_params=weather_server_params,
        )

        # Add calculator server (if available)
        calculator_server_params = MCPServerStdioParams(
            command="python",
            args=["tutorials/mcp_agent/mcp_calculator_server.py"],
            env=None,
        )
        tool_manager.add_server(
            name="calculator_server",
            server_params=calculator_server_params,
        )

        tools = await tool_manager.get_all_tools()
        print(f"Total tools from multiple servers: {len(tools)}")

        # List all available tools
        for i, tool in enumerate(tools):
            print(f"Tool {i+1}: {tool.name} - {tool.definition}")

    print("\n=== Testing Multiple Local Python Servers ===")
    asyncio.run(test_multiple_servers())
