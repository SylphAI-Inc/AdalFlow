from adalflow.core.mcp_tool import (
    MCPToolManager,
    MCPServerStdioParams,
    mcp_session_context,
)
from adalflow.components.agent import ReActAgent
from adalflow.utils import setup_env
import asyncio
import adalflow as adal

if __name__ == "__main__":
    server_params = MCPServerStdioParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/liyin/Documents/test/LightRAG",
        ],
    )

    # manager = MCPToolManager()
    # manager.add_server(
    #     name="mcp",
    #     server_params=MCPServerStdioParams(
    #         command="npx",
    #         args=[
    #             "-y",
    #             "@modelcontextprotocol/server-filesystem",
    #             "/Users/liyin/Documents/test/LightRAG",
    #         ],
    #     ),
    # )

    # 1. test the tool directly from mcp connection

    async def file_ops_tool():
        async with mcp_session_context(server_params, name="file_ops") as session:
            tools = await session.list_tools()
            print(tools)
            try:
                output = await session.call_tool(
                    "read_file",
                    arguments={
                        "path": "/Users/liyin/Documents/test/LightRAG/README.md"
                    },
                )
                print("type of output: ", type(output))
                return output
            except Exception as e:
                print(e)
                return None

    output = asyncio.run(file_ops_tool())
    print(output)

    # 2. test the MCP function tool
    # 3. test the MCP tool manager with the agent
    setup_env()

    gpt_model_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    # TODO: make this one step
    # TODO: integrate this with the new agent + runner

    async def test_mcp_agent():
        tool_manager = MCPToolManager()
        tool_manager.add_server(
            name="file_ops",
            server_params=server_params,
        )
        tools = await tool_manager.get_all_tools()

        react = ReActAgent(
            max_steps=6,
            add_llm_as_fallback=True,
            tools=tools,
            model_client=adal.OpenAIClient(),
            model_kwargs=gpt_model_kwargs,
            debug=True,
        )
        query = "What files are in the directory?"
        output = react.call(query)
        print(output)

    asyncio.run(test_mcp_agent())
