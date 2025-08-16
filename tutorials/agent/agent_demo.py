import os
import sys
import asyncio
from adalflow.components.agent.agent import Agent
from adalflow.components.agent.runner import Runner
from adalflow.components.model_client import OpenAIClient
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import (
    ToolOutput, 
    RawResponsesStreamEvent, 
    RunItemStreamEvent,
    ToolCallRunItem,
    ToolOutputRunItem,
    FinalOutputItem,
    ToolCallActivityRunItem,
)
from adalflow.apps.cli_permission_handler import CLIPermissionHandler, AutoApprovalHandler
from adalflow.tracing import (
    set_tracing_disabled, 
    enable_mlflow_local, 
    trace
)
import adalflow as adal
from pathlib import Path
import os

from adalflow.utils import setup_env


def calculator(expression: str) -> str:
    """A synchronous calculator that evaluates mathematical expressions."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

# 2. Async Function
async def web_search(query: str="what is the weather in SF today?") -> str:
    """web search on query."""
    # Simulate async operation
    await asyncio.sleep(0.5)
    return "San Francisco will be mostly cloudy today with some afternoon sun, reaching about 67 °F (20 °C) and dipping to 57 °F (14 °C) tonight."
    
# streaming tool can include toocallactivity run item and the last one as the final output observable to the llm agent
# 3. Sync Generator Function
def counter(limit: int):
    """A counter that counts up to a limit."""
    final_output = []
    for i in range(1, limit + 1):
        stream_item =  f"Count: {i}/{limit}"
        final_output.append(stream_item)
        stream_event = ToolCallActivityRunItem(data=stream_item)
        yield stream_event

    yield final_output

# 4. Async Generator Function
async def data_streamer(data_type: str):
    """Streams different types of data.

    Args:
        data_types can be "numbers", "letters", "words"
    """
    import asyncio
    data_items = {
        "numbers": [1, 2, 3, 4, 5],
        "letters": ["a", "b", "c", "d", "e"],
        "words": ["hello", "world", "from", "async", "generator"]
    }
    
    items = data_items.get(data_type, ["unknown"])
    final_output = []
    for item in items:
        await asyncio.sleep(0.2)  # Simulate async data fetching
        stream_item =  f"Streaming {data_type}: {item}"
        final_output.append(stream_item)
        stream_event = ToolCallActivityRunItem(data=stream_item)
        yield stream_event

    yield final_output


def run_call_example():
    """When using call, the generator will be collected and the last one will be the observation"""
    print("\n=== Running Call Example (Synchronous) ===")
    
    agent = Agent(
        name="DemoAgent",
        tools=[FunctionTool(calculator), web_search, counter, data_streamer],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)
    query = "What is 15 * 7 + 23? What is the sf weather? count to 3, and stream some numbers"
    
    runner_result = runner.call(prompt_kwargs={"input_str": query})
    print(f"\nResult: {runner_result}")
    print(f"Answer: {runner_result.answer}")
    return runner_result


async def run_acall_example():
    """Example using runner.acall() - asynchronous execution"""
    print("\n=== Running ACall Example (Asynchronous) ===")
    
    agent = Agent(
        name="AsyncAgent",
        tools=[FunctionTool(calculator), web_search, counter, data_streamer],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=8
    )

    runner = Runner(agent=agent)
    query = "Calculate 42 * 3.14, search for weather in SF, count to 3, and stream some numbers"
    
    runner_result = await runner.acall(prompt_kwargs={"input_str": query})
    print(f"\nResult: {runner_result}")
    print(f"Answer: {runner_result.answer}")
    return runner_result


async def run_astream_example():
    """Example using runner.astream() - streaming execution"""
    print("\n=== Running AStream Example (Streaming) ===")
    
    agent = Agent(
        name="StreamingAgent",
        tools=[FunctionTool(calculator), web_search, counter, data_streamer],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=10
    )

    runner = Runner(agent=agent)
    query = "What is 25 * 4? Check the weather in SF. Count to 5 and stream some letters"
    
    # Start streaming
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": query},
        model_kwargs={"stream": True}
    )
    
    # Process streaming events
    async for event in streaming_result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"\n🔧 Calling tool: {event.item.data.name}")
                
            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool output: {event.item.data.output}")
                
            elif isinstance(event.item, ToolCallActivityRunItem):
                # This captures generator yields
                print(f"📝 Activity: {event.item.data}")
                
            elif isinstance(event.item, FinalOutputItem):
                print(f"\n🎯 Final answer: {event.item.data.answer}")
    
    return streaming_result


def main():
    """Main function to run all examples"""
    setup_env()
    
    print("=" * 60)
    print("AdalFlow Agent Demo - All Execution Methods")
    print("=" * 60)
    
    # 1. Run synchronous call example
    run_call_example()
    return
    
    # 2. Run asynchronous examples
    print("\n" + "=" * 60)
    print("Running Async Examples...")
    print("=" * 60)
    
    # Create async event loop and run async examples
    async def run_async_examples():
        # Run acall example
        await run_acall_example()
        
        # Run astream example
        await run_astream_example()

        
    
    # Run the async examples
    asyncio.run(run_async_examples())
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()