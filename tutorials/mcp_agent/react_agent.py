import os
from adalflow.components.agent import ReActAgent
from adalflow.core import ModelClientType, ModelClient
from adalflow.utils import setup_env
import asyncio
from mcp import StdioServerParameters
import logging
from adalflow.core.mcp_tool import MCPClientManager

# get_logger(level="DEBUG")
# logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)

setup_env('.env')


gpt_model_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
}


async def test_react_agent(model_client: ModelClient, model_kwargs: dict):

    print("\n=== Multiple Server Management ===")
    manager = MCPClientManager()
    # Add servers
    manager.add_server("calculator_server", StdioServerParameters(
        command="python",  # Command to run the server
        args=["mcp_server.py"],  # Arguments (path to your server script)
        env=None  # Optional environment variables
    ))
    # Find the configure at https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server
    # manager.add_server("duckduckgo-mcp-server", StdioServerParameters(
    #     command="npx",  # Command to run the server
    #     args=[
    #         "-y",
    #         "@smithery/cli@latest",
    #         "run",
    #         "@nickclyde/duckduckgo-mcp-server",
    #         "--key",
    #         "smithery-api-key"
    #     ],
    # ))
    # Load the DuckDuckGo MCP server from a JSON file.
    json_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
    manager.add_servers_from_json_file(json_path)
    
    await manager.list_all_resources()
    tools = await manager.get_all_tools()
    # print(tools)
    for tool in tools:
        sig = tool.definition.func_desc.split('\n')[0]
        print(f"- Tool: {tool.definition.func_name}, Signature: {sig}")
    print("Tools loaded successfully.")

    queries = [
        "What is the capital of France? What is the weather there? What is 465 times 321 then add 95297 and then divide by 13.2?",
        "Use DuckDuckGo to search for the winner on European Championship in 2025.",
    ]

    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=tools,
        model_client=model_client,
        model_kwargs=model_kwargs,
        debug=True,
    )
    print(react)

    for query in queries:
        print(f"Query: {query}")
        agent_response = react.call(query)
        print(f"\nAgent response: \n{agent_response.answer}")
        print("\nStep history:")
        for step in agent_response.step_history:
            print(
                f"  - {step.step} {step.action}\n\t\t{step.action.thought} observation: {step.observation}")
        ground_truth = (465 * 321 + 95297) / 13.2
        print(f"Ground truth: {ground_truth}")
        print("")


if __name__ == "__main__":
    asyncio.run(test_react_agent(ModelClientType.OPENAI(), gpt_model_kwargs))
    print("Done")
