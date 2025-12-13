#!/usr/bin/env python3
"""
Test script to demonstrate the permission system for tool execution.
"""

import asyncio
from adalflow.components.agent import Agent, Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.apps.cli_permission_handler import CLIPermissionHandler
from adalflow.core.types import RunItemStreamEvent, ToolOutput
from adalflow.core.prompt_builder import Prompt
from adalflow.utils import setup_env, printc
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
import pytest

# Setup environment
try:
    setup_env()
except FileNotFoundError:
    # Skip setup_env if .env file doesn't exist for testing
    pass

# Model configurations
claude_model = {
    "model_client": AnthropicAPIClient(api_key="fake_anthropic_key"),
    "model_kwargs": {"model": "claude-3-opus-20240229", "max_tokens": 4096},
}

openai_model = {
    "model_client": OpenAIClient(api_key="fake_api_key"),
    "model_kwargs": {"model": "gpt-4o-mini", "max_tokens": 4096},
}


# Example tools with proper as_tool outputs
def search_web(query: str, max_results: int = 5) -> ToolOutput:
    """Search the web for information."""
    print(f"Searching web for: {query} (max {max_results} results)")
    return ToolOutput(
        output=f"Found {max_results} results for '{query}'",
        observation=f"Search completed for '{query}'",
        display=f"🔍 Searched: {query}",
    )


def read_file(filename: str) -> ToolOutput:
    """Read contents of a file."""
    print(f"Reading file: {filename}")
    return ToolOutput(
        output=f"Contents of {filename}: [file content here]",
        observation=f"Successfully read file {filename}",
        display=f"📄 Read: {filename}",
    )


def write_file(filename: str, content: str) -> ToolOutput:
    """Write content to a file."""
    print(f"Writing to file: {filename}")
    return ToolOutput(
        output=f"Successfully wrote {len(content)} characters to {filename}",
        observation=f"File {filename} written successfully",
        display=f"✍️ Wrote: {filename}",
    )


@pytest.mark.asyncio
async def test_sync_execution():
    """Test synchronous execution with permission checks."""
    print("\n=== Testing Synchronous Execution ===\n")

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request. Always ask for permission before executing tools."""
    )

    # Create agent with model client
    agent = Agent(
        name="PermissionTestAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model,  # Use OpenAI model configuration
    )

    # Create permission handler
    permission_handler = CLIPermissionHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {
        "input_str": "Search for Python tutorials and save the results to a file"
    }

    # Execute
    result = runner.call(prompt_kwargs)
    print(f"\nFinal result: {result}")


@pytest.mark.asyncio
async def test_async_execution():
    """Test asynchronous execution with permission checks."""
    print("\n=== Testing Asynchronous Execution ===\n")

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request. Always ask for permission before executing tools."""
    )

    # Create agent with model client
    agent = Agent(
        name="PermissionTestAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model,  # Use OpenAI model configuration
    )

    # Create permission handler
    permission_handler = CLIPermissionHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {"input_str": "Read config.json and update it with new settings"}

    # Execute
    result = await runner.acall(prompt_kwargs)
    print(f"\nFinal result: {result}")


@pytest.mark.asyncio
async def test_streaming_execution():
    """Test streaming execution with permission events."""
    print("\n=== Testing Streaming Execution ===\n")

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request. Always ask for permission before executing tools."""
    )

    # Create agent with model client
    agent = Agent(
        name="PermissionTestAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model,  # Use OpenAI model configuration
    )

    # Create permission handler
    permission_handler = CLIPermissionHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {"input_str": "Search for news and write a summary to output.txt"}

    # Execute with streaming
    streaming_result = runner.astream(prompt_kwargs)

    # Stream events to file and process
    event_count = 0
    async for event in streaming_result.stream_to_json("permission_test_events.json"):
        event_count += 1
        if isinstance(event, RunItemStreamEvent):
            if event.name == "agent.tool_permission_request":
                printc(
                    f"\n📋 Permission Request: {event.item.data.tool_name}",
                    color="yellow",
                )
            elif event.name == "agent.tool_call_start":
                printc(f"\n🔧 Tool Call Start: {event.item.data.name}", color="cyan")
            elif event.name == "agent.tool_call_complete":
                printc("\n✅ Tool Call Complete", color="green")
            elif event.name == "agent.step_complete":
                printc(f"\n📍 Step {event.item.data.step} Complete", color="blue")
            elif event.name == "agent.execution_complete":
                printc("\n🎯 Execution Complete", color="green")

    print(f"\nTotal events: {event_count}")
    print(f"Final answer: {streaming_result.answer}")


@pytest.mark.asyncio
async def test_auto_approval():
    """Test with auto approval mode."""
    print("\n=== Testing Auto Approval Mode ===\n")

    from adalflow.apps.cli_permission_handler import AutoApprovalHandler

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request."""
    )

    # Create agent with model client
    agent = Agent(
        name="PermissionTestAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model,  # Use OpenAI model configuration
    )

    # Create auto approval handler
    permission_handler = AutoApprovalHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {"input_str": "Read data.csv and write a summary"}

    # Execute
    result = await runner.acall(prompt_kwargs)
    print(f"\nFinal result: {result}")


@pytest.mark.asyncio
async def test_mixed_approval():
    """Test with mixed approval settings - some tools require approval, others don't."""
    print("\n=== Testing Mixed Approval Settings ===\n")

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Some tools require approval, others don't. Complete the user's request efficiently."""
    )

    # Create agent with mixed approval settings
    agent = Agent(
        name="MixedApprovalAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=False),  # No approval needed
            FunctionTool(read_file, require_approval=False),  # No approval needed
            FunctionTool(write_file, require_approval=True),  # Approval required
        ],
        answer_data_type=str,
        **claude_model,  # Use Claude model for this test
    )

    # Create permission handler
    permission_handler = CLIPermissionHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {
        "input_str": "Search for information about Python, read README.md, and write a summary to output.txt"
    }

    # Execute
    result = await runner.acall(prompt_kwargs)
    print(f"\nFinal result: {result}")


def test_sync_execution_wrapper():
    """Wrapper for synchronous execution test."""
    print("\n=== Testing Synchronous Execution (Non-Async) ===\n")

    # Create role description
    role_desc = Prompt(
        template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request. Always ask for permission before executing tools."""
    )

    # Create agent with model client
    agent = Agent(
        name="PermissionTestAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model,  # Use OpenAI model configuration
    )

    # Create permission handler
    permission_handler = CLIPermissionHandler()

    # Create runner with permission manager
    runner = Runner(agent=agent, permission_manager=permission_handler, max_steps=5)

    # Test prompt
    prompt_kwargs = {"input_str": "Tell me what tools you have available"}

    # Execute synchronously
    result = runner.call(prompt_kwargs)
    print(f"\nFinal result: {result}")


async def main():
    """Run all tests."""
    # Test selection menu
    print("\n" + "=" * 60)
    print("PERMISSION SYSTEM TEST SUITE")
    print("=" * 60)
    print("\nSelect a test to run:")
    print("1. Synchronous Execution (blocking)")
    print("2. Asynchronous Execution")
    print("3. Streaming Execution (with events)")
    print("4. Auto Approval Mode")
    print("5. Mixed Approval Settings")
    print("6. Run All Tests")
    print("=" * 60)

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == "1":
        test_sync_execution_wrapper()
    elif choice == "2":
        await test_async_execution()
    elif choice == "3":
        await test_streaming_execution()
    elif choice == "4":
        await test_auto_approval()
    elif choice == "5":
        await test_mixed_approval()
    elif choice == "6":
        # Run all tests
        test_sync_execution_wrapper()
        await test_async_execution()
        await test_streaming_execution()
        await test_auto_approval()
        await test_mixed_approval()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    asyncio.run(main())
