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
        # Arguments (path to your server script)
        args=["mcp_calculator_server.py"],
        env=None  # Optional environment variables
    ))
    
    # duckduckgo MCP server: Find the configure at https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server
    # ======= Example 1: Add via npx server. =======
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
    
    # ======= Example 2: Load servers from a JSON file. =======
    # json_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
    # manager.add_servers_from_json_file(json_path)
    
    # ======= Example 3: Load server from sse URL. =======
    smithery_api_key = os.environ.get("SMITHERY_API_KEY")
    smithery_server_id = "@nickclyde/duckduckgo-mcp-server"
    mcp_server_url = f"https://server.smithery.ai/{smithery_server_id}/mcp?api_key={smithery_api_key}"
    # https://server.smithery.ai/@nickclyde/duckduckgo-mcp-server/mcp?api_key=c520e3ed-003a-4d30-98dc-74cfe50fa891
    manager.add_server("duckduckgo-mcp-server", mcp_server_url)

    await manager.list_all_tools()
    tools = await manager.get_all_tools()
    # print(tools)
    for tool in tools:
        sig = tool.definition.func_desc.split('\n')[0]
        print(f"- Tool: {tool.definition.func_name}, Signature: {sig}")
    print("Tools loaded successfully.")

    queries = [
        "What is the capital of the Texas? How is the weather there?",
        "What is 465 times 321 then add 95297 and then divide by 13.2?",
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
        print("")


if __name__ == "__main__":
    asyncio.run(test_react_agent(ModelClientType.OPENAI(), gpt_model_kwargs))
    print("Done")
