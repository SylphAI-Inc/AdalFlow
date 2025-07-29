"""
Advanced Agent Tutorial: Human-in-the-Loop, Streaming, and Tracing
This tutorial demonstrates advanced AdalFlow agent features including permission management,
real-time streaming, and comprehensive tracing capabilities.

Prerequisites:
1. Install AdalFlow: pip install adalflow
2. Set OpenAI API key: export OPENAI_API_KEY="your-api-key-here"
3. For MLflow tracing: mlflow server --host 127.0.0.1 --port 8000
4. Run the script: python tutorial_agent_advanced.py

Note: This script requires an OpenAI API key to function properly.
"""

# Start the MLflow Tracking Server (in a separate terminal):
#    mlflow server --host 0.0.0.0 --port 8080

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
    ToolCallActivityRunItem
)
from adalflow.apps.cli_permission_handler import CLIPermissionHandler, AutoApprovalHandler
from adalflow.tracing import (
    set_tracing_disabled, 
    enable_mlflow_local, 
    trace
)
import adalflow

# Setup environment
adalflow.setup_env()

def run_example_safely(example_func, example_name):
    """Safely run an example function with error handling."""
    try:
        print(f"\n📋 Running {example_name}...")
        result = example_func()
        if result is not None:
            print(f"✅ {example_name} completed successfully")
        else:
            print(f"⚠️ {example_name} completed with warnings")
        return result
    except Exception as e:
        print(f"❌ Error in {example_name}: {e}")
        print("Moving to next example...\n")
        return None


async def run_async_example_safely(example_func, example_name):
    """Safely run an async example function with error handling."""
    try:
        print(f"\n📋 Running {example_name}...")
        result = await example_func()
        if result is not None:
            print(f"✅ {example_name} completed successfully")
        else:
            print(f"⚠️ {example_name} completed with warnings")
        return result
    except Exception as e:
        print(f"❌ Error in {example_name}: {e}")
        print("Moving to next example...\n")
        return None


# Tool definitions
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    result = eval(expression)
    return result 


async def file_writer(filename: str, content: str) -> ToolOutput:
    """Write content to a file - requires permission."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return ToolOutput(
            output=f"Successfully wrote {len(content)} characters to {filename}",
            observation=f"File {filename} written successfully",
            display=f"✍️ Wrote: {filename}",
        )
    except Exception as e:
        return ToolOutput(
            output=f"Error writing to file: {e}",
            observation=f"Failed to write to {filename}",
            display=f"❌ Failed: {filename}",
        )


def text_processor(text: str) -> str:
    """Simple text processing tool."""
    return f"Processed: {text.upper().replace(' ', '_')}"


def analysis_tool(data: str) -> str:
    """Analyze data and return insights."""
    return f"Analysis of '{data}': This appears to be a numerical calculation with result involving multiplication and addition."


def report_generator(content: str) -> str:
    """Generate a formatted report."""
    return f"## Analysis Report\n\n**Content**: {content}\n\n**Generated**: Using AdalFlow tracing with MLflow integration"


async def data_processor(query: str):
    """Process data and yield intermediate results."""
    steps = [
        f"Analyzing query: '{query}'",
        f"Fetching relevant data for: {query}",
        f"Processing data patterns...",
        f"Generating insights from: {query}",
        f"Final analysis complete for: {query}"
    ]

    for i, step in enumerate(steps):
        await asyncio.sleep(0.5)  # Simulate processing time
        yield ToolCallActivityRunItem(data=f"Step {i+1}: {step}")


async def live_monitor(system: str):
    """Monitor system status and yield live updates."""
    statuses = [
        f"🟢 {system} system online",
        f"📊 {system} performance: Normal",
        f"🔍 {system} scanning for issues",
        f"✅ {system} health check complete"
    ]

    for status in statuses:
        await asyncio.sleep(0.4)
        yield ToolCallActivityRunItem(data=status)


# Human-in-the-Loop Examples
async def human_in_the_loop_basic_example():
    """Demonstrates basic human-in-the-loop with permission management."""
    print("\n=== Human-in-the-Loop Basic Example ===")
    
    # Create agent with tools that require permission
    agent = Agent(
        name="PermissionAgent",
        tools=[
            calculator,  # Safe tool - no permission needed
            FunctionTool(file_writer, require_approval=True),  # Requires permission
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=6
    )

    permission_handler = CLIPermissionHandler(approval_mode="default")
    runner = Runner(agent=agent, permission_manager=permission_handler)

    # Tools will now require approval before execution
    result = runner.astream(prompt_kwargs={"input_str": "call the calculator function and calculate 25 * 4 and create a file called 'test.txt' with some interesting content"}, model_kwargs={"stream": True})

    async for event in result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"🔧 Calling tool: {event.item.data.name}")

            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.data.output}")

            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")
    
    return result


async def human_in_the_loop_auto_approve_example():
    """Demonstrates auto-approval mode for development environments."""
    print("\n=== Human-in-the-Loop Auto-Approve Example ===")
    
    agent = Agent(
        name="AutoApproveAgent",
        tools=[
            calculator,
            FunctionTool(file_writer, require_approval=True),
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    # Auto-approval handler - automatically approves all tool requests
    auto_handler = AutoApprovalHandler()
    runner = Runner(agent=agent, permission_manager=auto_handler)

    result = runner.astream(prompt_kwargs={"input_str": "Call the calculator function and Calculate 25 * 4 and save the result to 'calculation.txt'"}, model_kwargs={"stream": True})
    
    async for event in result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"🔧 Calling tool: {event.item.data.name}")

            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.data.output}")

            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")

    return result


async def human_in_the_loop_yolo_mode_example():
    """Demonstrates YOLO mode that bypasses all permission checks."""
    print("\n=== Human-in-the-Loop YOLO Mode Example ===")
    
    agent = Agent(
        name="YOLOAgent",
        tools=[
            FunctionTool(calculator),
            FunctionTool(file_writer, require_approval=True),
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    # YOLO mode - bypasses all permission checks
    yolo_handler = CLIPermissionHandler(approval_mode="yolo")
    runner = Runner(agent=agent, permission_manager=yolo_handler)

    result = runner.astream(prompt_kwargs={"input_str": "Calculate 15 * 7 + 23 and write it to 'yolo_result.txt'"}, model_kwargs={"stream": True})
    
    async for event in result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"🔧 Calling tool: {event.item.data.name}")

            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.data.output}")

            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")

    return result


# Streaming Examples
async def streaming_basic_example():
    """Demonstrates basic streaming with agent execution."""
    print("\n=== Streaming Basic Example ===")
    
    agent = Agent(
        name="StreamingAgent",
        tools=[FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    # Start streaming execution
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "What is 15 * 7 + 23?"},
        model_kwargs={"stream": True}
    )

    # Process streaming events
    async for event in streaming_result.stream_events():
        print(f"Event: {event}")

    return streaming_result


async def streaming_raw_responses_example():
    """Demonstrates handling raw response stream events."""
    print("\n=== Streaming Raw Responses Example ===")
    
    agent = Agent(
        name="StreamingAgent",
        tools=[FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 25 * 4 and explain the result"},
        model_kwargs={"stream": True}
    )

    print("Raw streaming output:")
    async for event in streaming_result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            # Process raw model output
            if hasattr(event.data, 'choices') and event.data.choices:
                delta = event.data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end='', flush=True)

    print("\n")  # Add newline after streaming
    return streaming_result


async def streaming_agent_events_example():
    """Demonstrates handling high-level agent execution events."""
    print("\n=== Streaming Agent Events Example ===")
    
    agent = Agent(
        name="StreamingAgent",
        tools=[FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 15 * 7 + 23 and explain the steps"},
        model_kwargs={"stream": True}
    )

    async for event in streaming_result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"🔧 Calling tool: {event.item.data.name}")

            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.data.output}")

            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")

    return streaming_result


async def streaming_anthropic_example():
    """Demonstrates streaming with Anthropic client."""
    print("\n=== Streaming Anthropic Example ===")
    
    # Check if Anthropic API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ ANTHROPIC_API_KEY not set, skipping Anthropic example")
        return None
    
    agent = Agent(
        name="AnthropicAgent",
        tools=[FunctionTool(calculator)],
        model_client=AnthropicAPIClient(),
        model_kwargs={"model": "claude-3-5-haiku-20241022", "stream": True, "temperature": 0.8},
        max_steps=5
    )

    runner = Runner(agent=agent)

    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 42 * 3 and explain why this might be significant"},
        model_kwargs={"stream": True}
    )

    async for event in streaming_result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, FinalOutputItem):
                print(f"🎯 Anthropic Final answer: {event.item.data.answer}")

    return streaming_result


# Tracing Examples
def tracing_basic_example():
    """Demonstrates basic agent tracing."""
    print("\n=== Tracing Basic Example ===")
    
    # Enable tracing
    set_tracing_disabled(False)

    # Create agent
    agent = Agent(
        name="SimpleAgent",
        tools=[
            FunctionTool(calculator),
            FunctionTool(text_processor),
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.3},
        max_steps=3
    )

    # Create runner
    runner = Runner(agent=agent)

    # Execute agent - automatically traced
    with trace(workflow_name="Tracing SimpleAgent"):
        result = runner.call(
            prompt_kwargs={
                "input_str": "Calculate 12 * 8 and then process the text 'hello world'"
            }
        )

    print(f"Agent result: {result.answer}")
    return result


# check out local mlflow server at http://localhost:8080 for the tracing
def tracing_mlflow_integration_example():
    """Demonstrates MLflow integration for enterprise tracing."""
    print("\n=== Tracing MLflow Integration Example ===")
    
    # Try to enable MLflow tracing
    try:
        mlflow_enabled = enable_mlflow_local(
            tracking_uri="http://localhost:8080",
            experiment_name="AdalFlow-Tracing-Demo",
            project_name="Agent-Workflows"
        )
        print("✅ MLflow integration enabled")
    except Exception as e:
        print("Error message")
        print(f"⚠️ MLflow not available: {e}")
        print("Make sure MLflow server is running: mlflow server --host 127.0.0.1 --port 8080")

    # Ensure tracing is enabled
    set_tracing_disabled(False)

    # Create agent
    agent = Agent(
        name="AnalysisAgent",
        tools=[
            FunctionTool(analysis_tool),
            FunctionTool(report_generator),
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.2},
        max_steps=4
    )

    runner = Runner(agent=agent)

    # Execute workflow - automatically traced to MLflow if available
    with trace(workflow_name="Agent-Tutorial"):
        result = runner.call(
            prompt_kwargs={
                "input_str": "Analyze the calculation 25 * 4 + 15 and generate a report"
            }
        )

    print(f"Analysis complete: {result.answer}")
    return result


async def tracing_async_generator_tools_example():
    """Demonstrates tracing agent with async generator tools during streaming."""
    print("\n=== Tracing Async Generator Tools Example ===")
    
    # Enable tracing
    set_tracing_disabled(False)

    # Setup MLflow if available
    try:
        mlflow_enabled = enable_mlflow_local(
            tracking_uri="http://localhost:8080",
            experiment_name="AdalFlow-AsyncTools-Demo",
            project_name="AsyncGenerator-Workflows"
        )
        print("✅ MLflow integration enabled for async tools")
    except Exception as e:
        print(f"⚠️ MLflow not available: {e}")

    # Create agent with async generator tools
    agent = Agent(
        name="AsyncGeneratorAgent",
        tools=[
            FunctionTool(data_processor),
            FunctionTool(live_monitor),
        ],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.2},
        max_steps=6
    )

    runner = Runner(agent=agent)

    with trace(workflow_name="AsyncGeneratorAgent"):
        # Execute with tracing context
        streaming_result = runner.astream(
            prompt_kwargs={
                "input_str": "Analyze the system performance data and monitor the database system status"
            },
            model_kwargs={"stream": True}
        )

        async for event in streaming_result.stream_events():
            if isinstance(event, RunItemStreamEvent):
                if event.name == "agent.tool_call_activity":
                    # This captures async generator yields
                    if hasattr(event.item, 'data') and event.item.data:
                        print(f"📝 Yielded: {event.item.data}")

                elif isinstance(event.item, FinalOutputItem):
                    print(f"🎯 Final Result: {event.item.data.answer}")

    return streaming_result


def main():
    """Run all advanced tutorial examples."""
    print("🚀 Starting Advanced Agent Tutorial Examples...")
    print("=" * 60)
    
    print("\n🔧 Environment setup complete!\n")
    
    # Run synchronous examples
    sync_examples = [
        # (tracing_basic_example, "Tracing Basic"),
        (tracing_mlflow_integration_example, "Tracing MLflow Integration"),
    ]
    
    successful_sync = 0
    for example_func, example_name in sync_examples:
        result = run_example_safely(example_func, example_name)
        if result is not None:
            successful_sync += 1
    
    # Run asynchronous examples
    print("\n" + "=" * 60)
    print("🔄 Running Asynchronous Examples...")
    
    async_examples = [
        (human_in_the_loop_basic_example, "Human-in-the-Loop Basic"),
        (human_in_the_loop_auto_approve_example, "Human-in-the-Loop Auto-Approve"),
        (human_in_the_loop_yolo_mode_example, "Human-in-the-Loop YOLO Mode"),
        (streaming_basic_example, "Streaming Basic"),
        (streaming_raw_responses_example, "Streaming Raw Responses"),
        (streaming_agent_events_example, "Streaming Agent Events"),
        (streaming_anthropic_example, "Streaming Anthropic"),
        (tracing_async_generator_tools_example, "Tracing Async Generator Tools"),
    ]
    
    successful_async = 0
    for example_func, example_name in async_examples:
        result = asyncio.run(run_async_example_safely(example_func, example_name))
        if result is not None:
            successful_async += 1
    
    print("\n" + "=" * 60)
    print("🎉 Advanced Tutorial Complete!")
    print(f"Successfully ran {successful_sync}/{len(sync_examples)} synchronous examples!")
    print(f"Successfully ran {successful_async}/{len(async_examples)} asynchronous examples!")
    print("\nNext steps:")
    print("- Explore permission management for production deployments")
    print("- Set up MLflow for comprehensive tracing: mlflow server --host 127.0.0.1 --port 8080")
    print("- Experiment with streaming for real-time applications")
    print("- Check the AdalFlow documentation for more advanced features")

if __name__ == "__main__":
    main()