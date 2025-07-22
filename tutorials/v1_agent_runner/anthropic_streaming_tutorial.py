#!/usr/bin/env python3
"""
Tutorial: Anthropic Streaming with AdalFlow Runner and Agent

This tutorial demonstrates streaming functionality with Anthropic models using:
1. AdalFlow's Runner streaming with agent-based workflows
2. Agent execution with streaming model clients
3. Real-time event processing from streaming workflows
4. Integration testing for streaming agents and tools

Requirements:
- Set ANTHROPIC_API_KEY environment variable in .env file
- Install: pip install adalflow anthropic

Key Features Demonstrated:
1. Runner.astream() for streaming agent workflows
2. Real-time event processing and monitoring
3. Tool execution within streaming workflows
4. Integration between agents, runners, and streaming models
"""

import os
import asyncio
from typing import List
from dataclasses import dataclass
import logging

from adalflow.core.generator import Generator
from adalflow.core.types import ModelType
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.core.base_data_class import DataClass
from adalflow.utils import setup_env
import anthropic

logging.basicConfig(level=logging.DEBUG)

setup_env()  # Comment out to prevent FileNotFoundError when .env doesn't exist


def demonstrate_direct_anthropic_streaming():
    """
    Demonstrate direct streaming from Anthropic API using the official library.

    This function shows how to:
    - Create direct API calls to Anthropic's streaming endpoint
    - Process native event types and response formats
    - Extract and assemble content from streaming events
    """

    print("=== Direct Anthropic API Streaming ===\n")

    # Initialize direct Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = "Explain the concept of machine learning in simple terms."
    print(f"Prompt: {prompt}\n")

    try:
        # Direct streaming using official anthropic library
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            stream=True,
        )

        print("Direct API Event Types and Data:")
        print("-" * 50)

        event_counts = {}
        content_parts = []

        for event in stream:
            print(event)
            print(type(event))

            # Extract content for text events
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                content_parts.append(event.delta.text)

        print(f"\nEvent Type Summary: {event_counts}")
        print(f"Assembled Content: {''.join(content_parts)}")

    except Exception as e:
        print(f"Direct API streaming error: {e}")


async def demonstrate_anthropic_async_streaming():
    """
    Demonstrate asynchronous streaming using AdalFlow's AnthropicAPIClient.

    This function shows how to:
    - Use AdalFlow's Generator with async streaming
    - Process streaming responses asynchronously
    - Handle async generators for real-time text generation
    """

    print("\n\n=== Anthropic Async Streaming Tutorial ===\n")

    client = AnthropicAPIClient()

    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "claude-3-5-haiku-20241022",
            "stream": True,
            "temperature": 0.8,
        },
        model_type=ModelType.LLM,
    )

    prompt = "Write a short poem about artificial intelligence."

    print(f"Prompt: {prompt}\n")
    print("Async Streaming Response:")
    print("-" * 50)

    result = await generator.acall(prompt_kwargs={"input_str": prompt})

    async for event in result.raw_response:
        print(event)
        print(type(event))


def demonstrate_non_streaming():
    """
    Demonstrate non-streaming mode for comparison with streaming approaches.

    This function shows how to:
    - Configure AdalFlow Generator for non-streaming responses
    - Compare response formats between streaming and non-streaming modes
    - Understand the structure of completed responses
    """

    print("\n\n=== Anthropic Non-Streaming Comparison ===\n")

    client = AnthropicAPIClient()

    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "claude-3-5-haiku-20241022",
            "stream": False,  # Disable streaming
            "temperature": 0.5,
        },
        model_type=ModelType.LLM,
    )

    prompt = "What is the capital of France?"

    print(f"Prompt: {prompt}\n")

    result = generator.call(prompt_kwargs={"input_str": prompt})

    print(f"Result type: {type(result)}")
    print(f"API Response type: {type(result.api_response)}")
    print(f"Raw Response type: {type(result.raw_response)}")
    print(f"\nRaw Response content:\n{result.raw_response}")
    print(f"\nUsage: {result.usage}")


def main():
    """Main tutorial function demonstrating all streaming capabilities."""
    print("=== Anthropic Streaming Tutorial ===")
    print("Comprehensive guide to streaming with AdalFlow and Anthropic\n")

    try:
        # Core streaming demonstrations
        demonstrate_direct_anthropic_streaming()
        asyncio.run(demonstrate_anthropic_async_streaming())
        demonstrate_non_streaming()

        # Advanced agent-based streaming
        demonstrate_runner_streaming()

        print("\n=== Tutorial Complete ===")
        print("Successfully demonstrated:")
        print("1. Direct Anthropic API streaming with official library")
        print("2. AdalFlow async streaming with Generator component")
        print("3. Non-streaming mode for comparison")
        print("4. Agent-based streaming workflows with Runner")
        print("5. Real-time event processing and monitoring")

    except Exception as e:
        print(f"Tutorial error: {e}")
        print("\nTroubleshooting:")
        print("- Verify ANTHROPIC_API_KEY environment variable is set")
        print("- Ensure network connectivity")
        print(
            "- Check that required packages are installed: pip install adalflow anthropic"
        )


# Tool functions for agent demonstrations
def add_numbers(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


@dataclass
class MathResult(DataClass):
    """Data class for mathematical operations result."""

    operation: str
    result: float
    steps: List[str]


def demonstrate_runner_streaming():
    """Demonstrate Runner streaming with agent-based workflows."""
    print("\n=== Runner Streaming Demonstration ===\n")

    # Create mathematical tools for the agent
    tools = [
        FunctionTool(fn=add_numbers),
        FunctionTool(fn=multiply_numbers),
    ]

    # Initialize streaming client
    client = AnthropicAPIClient()

    # Create agent with streaming capabilities
    agent = Agent(
        name="MathAgent",
        tools=tools,
        model_client=client,
        model_kwargs={
            "model": "claude-3-5-haiku-20241022",
            "stream": True,
            "max_tokens": 1000,
        },
        answer_data_type=str,
        max_steps=3,
    )

    # Create runner for agent execution
    runner = Runner(agent=agent)

    print("Starting streaming workflow...")

    # Execute streaming workflow
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 8 + 7, then multiply the result by 3"}
    )

    async def process_streaming_events():
        """Process and display streaming events from the workflow."""
        event_count = 0

        async for event in streaming_result.stream_events():
            event_count += 1
            print(f"Event {event_count}: {type(event).__name__}")

            # Display limited event details for readability
            if hasattr(event, "name"):
                print(f"  Name: {event.name}")
            if hasattr(event, "data") and event_count <= 5:  # Limit detailed output
                print(f"  Data: {str(event.data)[:100]}...")  # Truncate long data
            print()

        print(f"Workflow completed! Processed {event_count} events")
        print(f"Final answer: {streaming_result.answer}")
        print(f"Completion status: {streaming_result.is_complete}")

    # Run the streaming demonstration
    asyncio.run(process_streaming_events())


if __name__ == "__main__":
    # Run the tutorial demonstrations
    main()
