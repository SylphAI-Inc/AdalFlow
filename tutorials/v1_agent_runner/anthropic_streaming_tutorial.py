#!/usr/bin/env python3
"""
Tutorial: Anthropic Client with Direct API Streaming and Compatibility Testing

This tutorial demonstrates:
1. Direct streaming from Anthropic API client using official anthropic library
2. AdalFlow's AnthropicAPIClient streaming via OpenAI SDK compatibility
3. Compatibility testing between direct API responses and Response API format
4. Testing compatibility with Generator component

Requirements:
- Set ANTHROPIC_API_KEY environment variable
- Install: pip install adalflow openai anthropic

Key Features Demonstrated:
1. Direct Anthropic API streaming with official anthropic library
2. AnthropicAPIClient streaming via OpenAI SDK compatibility
3. Response format compatibility testing
4. Generator integration testing
5. Event type comparison and analysis
"""

import os
import asyncio
from typing import List

from adalflow.core.generator import Generator
from adalflow.core.types import ModelType
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.core.base_data_class import DataClass
from dataclasses import dataclass

from adalflow.utils.lazy_import import setup_env

# Direct Anthropic API imports
import anthropic


setup_env()  # Comment out to prevent FileNotFoundError when .env doesn't exist


def demonstrate_direct_anthropic_streaming():
    """
    Demonstrate direct streaming from Anthropic API using official anthropic library.
    This shows the native event types and response format.
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


# def demonstrate_anthropic_streaming():
#     """
#     Demonstrate synchronous streaming with Anthropic client.

#     This example shows how the raw_response contains OpenAI Response API
#     compatible text events while api_response contains the original
#     ChatCompletion stream from Anthropic's OpenAI compatibility layer.
#     """

#     print("=== Anthropic Streaming Tutorial ===\n")

#     # Initialize Anthropic client using OpenAI SDK compatibility
#     client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

#     # Create Generator with streaming enabled
#     generator = Generator(
#         model_client=client,
#         model_kwargs={
#             "model": "claude-3-5-haiku-20241022",
#             "stream": True,  # Enable streaming
#             "temperature": 0.7
#         },
#         model_type=ModelType.LLM
#     )

#     # Test prompt
#     prompt = "Explain the concept of machine learning in simple terms."

#     print(f"Prompt: {prompt}\n")
#     print("Streaming Response (raw_response format):")
#     print("-" * 50)

#     # Call generator with streaming
#     result = generator.call(prompt_kwargs={"input_str": prompt})

#     print(f"\nResult type: {type(result)}")
#     print(f"API Response type: {type(result.api_response)}")
#     print(f"Raw Response type: {type(result.raw_response)}")

#     # Process the streaming raw_response
#     if hasattr(result.raw_response, '__iter__'):
#         accumulated_text = ""
#         chunk_count = 0

#         print("\nStreaming text chunks:")
#         for event in result.raw_response:
#             print(event)
#             print(type(event))
#             chunk_count += 1

#         print(f"\nTotal chunks received: {chunk_count}")
#         print(f"Final accumulated text:\n{accumulated_text}")

#     else:
#         print(f"Raw response (non-streaming): {result.raw_response}")

#     print(f"\nUsage: {result.usage}")
#     print(f"Error: {result.error}")


async def demonstrate_anthropic_async_streaming():
    """
    Demonstrate asynchronous streaming with Anthropic client.

    Shows how async streaming works with the converted Response API format
    in raw_response while maintaining the original ChatCompletion stream
    in api_response.
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

    async for event in result:
        print(event)
        print(type(event))


def demonstrate_non_streaming():
    """
    Demonstrate non-streaming mode for comparison.

    Shows how non-streaming responses are converted from ChatCompletion
    to text format in raw_response.
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


def test_response_compatibility():
    """
    Test compatibility between direct API responses and Response API format.
    Compares event structures and data formats.
    """

    print("\n\n=== Response API Compatibility Testing ===\n")

    adalflow_client = AnthropicAPIClient()

    prompt = "What is 2+2?"
    print(f"Test Prompt: {prompt}\n")

    try:
        print("1. AdalFlow Client Response Format:")
        generator = Generator(
            model_client=adalflow_client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": True,
            },
            model_type=ModelType.LLM,
        )

        result = generator.call(prompt_kwargs={"input_str": prompt})

        print(f"AdalFlow Result Type: {type(result)}")
        print(f"API Response Type: {type(result.api_response)}")
        print(f"Raw Response Type: {type(result.raw_response)}")

        # Test if raw_response is iterable
        if hasattr(result.raw_response, "__iter__"):
            print("\nRaw Response Stream (first 3 chunks):")
            chunk_count = 0
            for event in result.raw_response:
                if chunk_count >= 3:
                    break
                print(event)
                print(type(event))
                chunk_count += 1

        print("\n2. Compatibility Analysis:")
        print("   - Direct API: Returns anthropic event objects")
        print("   - AdalFlow: Converts to text chunks in raw_response")
        print("   - Both maintain original data in their respective formats")

    except Exception as e:
        print(f"Compatibility testing error: {e}")


def test_generator_compatibility():
    """
    Test Generator compatibility with both streaming modes.
    """

    print("\n\n=== Generator Compatibility Testing ===\n")

    client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test streaming
    print("1. Testing Generator with Streaming:")
    streaming_generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "claude-3-5-haiku-20241022",
            "stream": True,
        },
        model_type=ModelType.LLM,
    )

    try:
        result = streaming_generator.call(prompt_kwargs={"input_str": "Count to 5"})
        print("   ✓ Streaming call successful")
        print(f"   ✓ Result type: {type(result)}")
        print(f"   ✓ Has raw_response: {hasattr(result, 'raw_response')}")
        print(f"   ✓ Has api_response: {hasattr(result, 'api_response')}")
        print(f"   ✓ Has usage: {hasattr(result, 'usage')}")

        # Test iteration
        if hasattr(result.raw_response, "__iter__"):
            print("   ✓ Raw response is iterable")
            chunk_count = sum(1 for _ in result.raw_response)
            print(f"   ✓ Total chunks: {chunk_count}")

    except Exception as e:
        print(f"   ✗ Streaming test failed: {e}")

    # Test non-streaming
    print("\n2. Testing Generator without Streaming:")
    non_streaming_generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "claude-3-5-haiku-20241022",
            "stream": False,
        },
        model_type=ModelType.LLM,
    )

    try:
        result = non_streaming_generator.call(
            prompt_kwargs={"input_str": "What is 1+1?"}
        )
        print("   ✓ Non-streaming call successful")
        print(f"   ✓ Result type: {type(result)}")
        print(f"   ✓ Raw response type: {type(result.raw_response)}")
        print(f"   ✓ Raw response content: {result.raw_response[:50]}...")

    except Exception as e:
        print(f"   ✗ Non-streaming test failed: {e}")


def main():
    """Main tutorial function demonstrating all use cases."""

    try:
        # Run direct API streaming demo
        # demonstrate_direct_anthropic_streaming()

        # # Run non-streaming demo for comparison
        # demonstrate_non_streaming()

        # # Test response compatibility
        test_response_compatibility()

        # # Test generator compatibility
        test_generator_compatibility()

        # Run async streaming demo
        print("\nRunning async streaming demo...")
        asyncio.run(demonstrate_anthropic_async_streaming())

        print("\n=== Tutorial Complete ===")
        print("This tutorial demonstrated:")
        print("1. Direct Anthropic API streaming with official library")
        print("2. AdalFlow's AnthropicAPIClient streaming via OpenAI SDK compatibility")
        print("3. Response format compatibility between approaches")
        print("4. Generator integration and compatibility testing")
        print("5. Event type comparison and analysis")

    except Exception as e:
        print(f"Error running tutorial: {e}")
        print("Make sure you have:")
        print("1. Valid ANTHROPIC_API_KEY set")
        print("2. Network connectivity")
        print("3. Required packages installed (adalflow, openai, anthropic)")


def test_direct_anthropic_streaming():
    """Test cases for direct anthropic API streaming functionality."""

    print("\n=== Testing Direct Anthropic API Streaming ===\n")

    # Test 1: Basic streaming functionality
    print("Test 1: Basic streaming functionality")
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
        )

        event_count = 0
        content_chunks = []

        for event in stream:
            event_count += 1
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                content_chunks.append(event.delta.text)

        assert event_count > 0, "Should receive at least one event"
        assert len(content_chunks) > 0, "Should receive content chunks"

        print(f"   ✓ Received {event_count} events")
        print(f"   ✓ Received {len(content_chunks)} content chunks")
        print(f"   ✓ Content: {''.join(content_chunks)[:50]}...")

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Event type validation
    print("\nTest 2: Event type validation")
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=30,
            stream=True,
        )

        event_types = set()
        for event in stream:
            event_types.add(type(event).__name__)

        expected_types = {
            "MessageStartEvent",
            "ContentBlockStartEvent",
            "ContentBlockDeltaEvent",
            "ContentBlockStopEvent",
            "MessageStopEvent",
        }
        found_types = event_types.intersection(expected_types)

        assert len(found_types) > 0, "Should receive expected event types"

        print(f"   ✓ Found event types: {found_types}")

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")

    # Test 3: Content assembly
    print("\nTest 3: Content assembly")
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Say exactly: 'Hello World'"}],
            max_tokens=20,
            stream=True,
        )

        assembled_content = ""
        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                assembled_content += event.delta.text

        assert len(assembled_content) > 0, "Should assemble content"
        assert "Hello" in assembled_content, "Should contain expected content"

        print(f"   ✓ Assembled content: {assembled_content}")

    except Exception as e:
        print(f"   ✗ Test 3 failed: {e}")


def test_async_streaming():
    """Test cases for async streaming functionality."""

    print("\n=== Testing Async Streaming ===\n")

    async def run_async_tests():
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Test 1: Basic async streaming
        print("Test 1: Basic async streaming")
        try:
            generator = Generator(
                model_client=client,
                model_kwargs={
                    "model": "claude-3-5-haiku-20241022",
                    "stream": True,
                    "max_tokens": 50,
                },
                model_type=ModelType.LLM,
            )

            result = await generator.acall(prompt_kwargs={"input_str": "Count to 3"})

            chunk_count = 0
            async for event in result:
                chunk_count += 1
                if chunk_count >= 10:  # Limit for testing
                    break

            assert chunk_count > 0, "Should receive streaming chunks"

            print(f"   ✓ Received {chunk_count} chunks")

        except Exception as e:
            print(f"   ✗ Test 1 failed: {e}")

        # Test 2: Async generator response format
        print("\nTest 2: Async generator response format")
        try:
            generator = Generator(
                model_client=client,
                model_kwargs={
                    "model": "claude-3-5-haiku-20241022",
                    "stream": True,
                    "max_tokens": 30,
                },
                model_type=ModelType.LLM,
            )

            result = await generator.acall(prompt_kwargs={"input_str": "Hello"})

            # Check if result is async generator
            assert hasattr(result, "__aiter__"), "Result should be async iterable"

            first_chunk = await result.__anext__()
            assert first_chunk is not None, "Should receive first chunk"

            print("   ✓ Result is async iterable")
            print(f"   ✓ First chunk type: {type(first_chunk)}")

        except Exception as e:
            print(f"   ✗ Test 2 failed: {e}")

    # Run async tests
    asyncio.run(run_async_tests())


def test_non_streaming():
    """Test cases for non-streaming mode."""

    print("\n=== Testing Non-Streaming Mode ===\n")

    # Test 1: Basic non-streaming
    print("Test 1: Basic non-streaming")
    try:
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        generator = Generator(
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": False,
                "max_tokens": 50,
            },
            model_type=ModelType.LLM,
        )

        result = generator.call(prompt_kwargs={"input_str": "What is 2+2?"})

        assert hasattr(result, "raw_response"), "Should have raw_response"
        assert isinstance(result.raw_response, str), "Raw response should be string"
        assert len(result.raw_response) > 0, "Should have content"

        print(f"   ✓ Raw response type: {type(result.raw_response)}")
        print(f"   ✓ Content length: {len(result.raw_response)}")
        print(f"   ✓ Content preview: {result.raw_response[:50]}...")

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Response structure validation
    print("\nTest 2: Response structure validation")
    try:
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        generator = Generator(
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": False,
                "max_tokens": 30,
            },
            model_type=ModelType.LLM,
        )

        result = generator.call(prompt_kwargs={"input_str": "Hello"})

        # Validate response structure
        assert hasattr(result, "api_response"), "Should have api_response"
        assert hasattr(result, "usage"), "Should have usage"
        assert hasattr(result, "error"), "Should have error field"

        print(f"   ✓ Has api_response: {hasattr(result, 'api_response')}")
        print(f"   ✓ Has usage: {hasattr(result, 'usage')}")
        print(f"   ✓ Has error: {hasattr(result, 'error')}")

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")


def test_error_handling():
    """Test cases for error handling."""

    print("\n=== Testing Error Handling ===\n")

    # Test 1: Invalid API key
    print("Test 1: Invalid API key handling")
    try:
        client = AnthropicAPIClient(api_key="invalid_key")

        generator = Generator(
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": False,
                "max_tokens": 30,
            },
            model_type=ModelType.LLM,
        )

        try:
            result = generator.call(prompt_kwargs={"input_str": "Test"})
            # Should either raise an exception or return error in result
            if hasattr(result, "error") and result.error:
                print(f"   ✓ Error properly captured: {result.error}")
            else:
                print("   ⚠ Error not captured as expected")
        except Exception as e:
            print(f"   ✓ Exception properly raised: {type(e).__name__}")

    except Exception as e:
        print(f"   ✗ Test 1 setup failed: {e}")

    # Test 2: Invalid model name
    print("\nTest 2: Invalid model name handling")
    try:
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        generator = Generator(
            model_client=client,
            model_kwargs={
                "model": "invalid-model-name",
                "stream": False,
                "max_tokens": 30,
            },
            model_type=ModelType.LLM,
        )

        try:
            result = generator.call(prompt_kwargs={"input_str": "Test"})
            if hasattr(result, "error") and result.error:
                print(f"   ✓ Error properly captured: {result.error}")
            else:
                print("   ⚠ Error not captured as expected")
        except Exception as e:
            print(f"   ✓ Exception properly raised: {type(e).__name__}")

    except Exception as e:
        print(f"   ✗ Test 2 setup failed: {e}")


def test_different_model_configurations():
    """Test cases for different model configurations."""

    print("\n=== Testing Different Model Configurations ===\n")

    # Test 1: Different temperature values
    print("Test 1: Different temperature values")
    try:
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        temperatures = [0.0, 0.5, 1.0]

        for temp in temperatures:
            generator = Generator(
                model_client=client,
                model_kwargs={
                    "model": "claude-3-5-haiku-20241022",
                    "stream": False,
                    "max_tokens": 30,
                    "temperature": temp,
                },
                model_type=ModelType.LLM,
            )

            result = generator.call(prompt_kwargs={"input_str": "Hello"})

            assert hasattr(result, "raw_response"), f"Temperature {temp} should work"
            assert (
                len(result.raw_response) > 0
            ), f"Temperature {temp} should return content"

            print(f"   ✓ Temperature {temp}: {len(result.raw_response)} chars")

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Different max_tokens values
    print("\nTest 2: Different max_tokens values")
    try:
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        max_tokens_values = [10, 50, 100]

        for max_tokens in max_tokens_values:
            generator = Generator(
                model_client=client,
                model_kwargs={
                    "model": "claude-3-5-haiku-20241022",
                    "stream": False,
                    "max_tokens": max_tokens,
                },
                model_type=ModelType.LLM,
            )

            result = generator.call(prompt_kwargs={"input_str": "Write a short story"})

            assert hasattr(
                result, "raw_response"
            ), f"Max tokens {max_tokens} should work"

            print(f"   ✓ Max tokens {max_tokens}: {len(result.raw_response)} chars")

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")


# Runner and Agent Tools for Testing
def add_numbers(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


async def async_divide_numbers(x: int, y: int) -> float:
    """Divide two numbers asynchronously."""
    if y == 0:
        return 0
    return x / y


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


# Data classes for testing
@dataclass
class MathResult(DataClass):
    """Simple math result data class."""

    operation: str
    result: float
    steps: List[str]


@dataclass
class SearchResult(DataClass):
    """Search result data class."""

    query: str
    results_count: int
    summary: str


def test_runner_with_streaming():
    """Test cases for Runner with streaming using AnthropicAPIClient."""

    print("\n=== Testing Runner with Streaming ===\n")

    # Test 1: Basic Runner streaming with simple tools
    print("Test 1: Basic Runner streaming with simple tools")
    try:
        # Create tools
        tools = [
            FunctionTool(fn=add_numbers),
            FunctionTool(fn=multiply_numbers),
            FunctionTool(fn=search_web),
        ]

        # Create client with streaming enabled
        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Create agent
        agent = Agent(
            name="StreamingTestAgent",
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

        # Create runner
        runner = Runner(agent=agent)

        # Test streaming
        streaming_result = runner.astream(
            prompt_kwargs={"input_str": "Add 5 and 3, then multiply the result by 2"}
        )

        # Verify streaming result structure
        assert hasattr(
            streaming_result, "stream_events"
        ), "Should have stream_events method"
        assert hasattr(streaming_result, "final_result"), "Should have final_result"
        assert hasattr(streaming_result, "is_complete"), "Should have is_complete"

        print("   ✓ Streaming result created")
        print(f"   ✓ Initial completion status: {streaming_result.is_complete}")

        # Test streaming events (limited for testing)
        async def test_streaming():
            async for event in streaming_result.stream_events():
                print(f"   Event: {type(event).__name__}")
                print(f"   Event content: {event}")

            await streaming_result.wait_for_completion()
            print(f"   ✓ Final completion status: {streaming_result.is_complete}")
            print(f"   ✓ Final result type: {type(streaming_result.final_result)}")

        asyncio.run(test_streaming())

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Runner streaming with data class output
    print("\nTest 2: Runner streaming with data class output")
    try:
        tools = [
            FunctionTool(fn=add_numbers),
            FunctionTool(fn=multiply_numbers),
        ]

        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        agent = Agent(
            name="DataClassStreamingAgent",
            tools=tools,
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": True,
                "max_tokens": 1000,
            },
            answer_data_type=MathResult,
            max_steps=3,
        )

        runner = Runner(agent=agent)

        streaming_result = runner.astream(
            prompt_kwargs={"input_str": "Calculate 7 * 6 and show the steps"}
        )

        async def test_data_class_streaming():
            # Wait for completion
            await streaming_result.wait_for_completion()

            assert streaming_result.is_complete, "Should be complete"
            assert streaming_result.final_result is not None, "Should have final result"

            print("   ✓ Streaming completed")
            print("   ✓ Final result available")

        asyncio.run(test_data_class_streaming())

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")


def test_agent_with_streaming():
    """Test cases for Agent with streaming functionality."""

    print("\n=== Testing Agent with Streaming ===\n")

    # Test 1: Agent with streaming model client
    print("Test 1: Agent with streaming model client")
    try:
        tools = [
            FunctionTool(fn=add_numbers),
            FunctionTool(fn=search_web),
        ]

        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        agent = Agent(
            name="StreamingAgent",
            tools=tools,
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": True,
                "max_tokens": 1000,
            },
            answer_data_type=str,
            max_steps=2,
        )

        # Test agent planner with streaming
        prompt_kwargs = {"input_str": "Add 10 and 20", "step_history": []}

        # Test async call with streaming
        async def test_agent_streaming():
            result = await agent.planner.acall(prompt_kwargs=prompt_kwargs)

            # Should be async generator for streaming
            assert hasattr(result, "__aiter__"), "Result should be async iterable"

            chunk_count = 0
            async for event in result:
                chunk_count += 1
                print(f"   Event {chunk_count}: {type(event).__name__}")
                print(f"   Event content: {event}")
                if chunk_count >= 3:  # Limit for testing
                    break

            assert chunk_count > 0, "Should receive chunks"
            print(f"   ✓ Received {chunk_count} streaming chunks")

        asyncio.run(test_agent_streaming())

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Agent properties and methods
    print("\nTest 2: Agent properties and methods")
    try:
        tools = [FunctionTool(fn=add_numbers)]

        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        agent = Agent(
            name="TestAgent",
            tools=tools,
            model_client=client,
            model_kwargs={"model": "claude-3-5-haiku-20241022", "stream": True},
            answer_data_type=str,
            max_steps=3,
        )

        # Test agent properties
        assert agent.name == "TestAgent", "Agent should have correct name"
        assert agent.max_steps == 3, "Agent should have correct max_steps"
        assert (
            agent.answer_data_type == str
        ), "Agent should have correct answer_data_type"
        assert hasattr(agent, "planner"), "Agent should have planner"
        assert hasattr(agent, "tool_manager"), "Agent should have tool_manager"

        # Test prompt generation
        prompt = agent.get_prompt(input_str="Test prompt", step_history=[])
        assert isinstance(prompt, str), "Should return string prompt"
        assert len(prompt) > 0, "Prompt should not be empty"

        print(f"   ✓ Agent name: {agent.name}")
        print(f"   ✓ Max steps: {agent.max_steps}")
        print(f"   ✓ Answer type: {agent.answer_data_type}")
        print(f"   ✓ Prompt generated: {len(prompt)} chars")

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")


def test_runner_agent_integration_streaming():
    """Test integration of Runner and Agent with streaming."""

    print("\n=== Testing Runner + Agent Integration with Streaming ===\n")

    # Test 1: Complete workflow with streaming
    print("Test 1: Complete workflow with streaming")
    try:
        tools = [
            FunctionTool(fn=add_numbers),
            FunctionTool(fn=multiply_numbers),
            FunctionTool(fn=async_divide_numbers),
        ]

        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        agent = Agent(
            name="IntegrationAgent",
            tools=tools,
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": True,
                "max_tokens": 1000,
            },
            answer_data_type=str,
            max_steps=4,
        )

        runner = Runner(agent=agent)

        # Test the complete workflow
        streaming_result = runner.astream(
            prompt_kwargs={"input_str": "Calculate (5 + 3) * 2 / 4"}
        )

        async def test_complete_workflow():
            events_received = []

            async for event in streaming_result.stream_events():
                events_received.append(event)
                print(f"   Event: {type(event).__name__}")
                print(f"   Event content: {event}")

            # Verify completion
            assert streaming_result.is_complete, "Workflow should be complete"
            assert streaming_result.final_result is not None, "Should have final result"
            assert len(events_received) > 0, "Should receive events"

            print(f"   ✓ Received {len(events_received)} events")
            print("   ✓ Workflow completed successfully")
            print(f"   ✓ Final answer: {streaming_result.final_result.answer}")

        asyncio.run(test_complete_workflow())

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Error handling in streaming
    print("\nTest 2: Error handling in streaming")
    try:
        tools = [FunctionTool(fn=add_numbers)]

        # Use invalid API key to test error handling
        client = AnthropicAPIClient(api_key="invalid_key")

        agent = Agent(
            name="ErrorTestAgent",
            tools=tools,
            model_client=client,
            model_kwargs={"model": "claude-3-5-haiku-20241022", "stream": True},
            answer_data_type=str,
            max_steps=2,
        )

        runner = Runner(agent=agent)

        streaming_result = runner.astream(prompt_kwargs={"input_str": "Add 1 and 1"})

        async def test_error_handling():
            try:
                async for event in streaming_result.stream_events():
                    pass
                # If we get here, check if error was captured
                if streaming_result.final_result and hasattr(
                    streaming_result.final_result, "error"
                ):
                    print("   ✓ Error properly captured in final result")
                else:
                    print("   ⚠ No error found, might have succeeded unexpectedly")
            except Exception as e:
                print(f"   ✓ Exception properly raised: {type(e).__name__}")

        asyncio.run(test_error_handling())

    except Exception as e:
        print(f"   ✗ Test 2 setup failed: {e}")


def test_runner_async_streaming():
    """Test cases for Runner async streaming methods."""

    print("\n=== Testing Runner Async Streaming ===\n")

    print("Test 1: astream vs acall comparison")
    try:
        tools = [FunctionTool(fn=add_numbers)]

        client = AnthropicAPIClient()

        agent = Agent(
            name="AsyncCompareAgent",
            tools=tools,
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": False,  # Non-streaming for acall test
                "max_tokens": 1000,
            },
            answer_data_type=str,
            max_steps=2,
        )

        runner = Runner(agent=agent)

        async def compare_async_methods():
            # Test acall (non-streaming)
            acall_result = await runner.acall(
                prompt_kwargs={"input_str": "Add 2 and 3"}
            )

            # Test astream
            # Update agent for streaming
            agent.planner.model_kwargs.update({"stream": True})

            astream_result = runner.astream(prompt_kwargs={"input_str": "Add 2 and 3"})

            await astream_result.wait_for_completion()

            print("\nAsync Compare Agent Final Results:\n")
            print(f"acall result: {acall_result}")
            print(f"astream result: {astream_result.final_result}")

            # Both should complete successfully
            assert hasattr(acall_result, "answer"), "acall should have answer"
            assert (
                astream_result.final_result is not None
            ), "astream should have final result"

            print(f"   ✓ acall result type: {type(acall_result)}")
            print(f"   ✓ astream result type: {type(astream_result.final_result)}")

        asyncio.run(compare_async_methods())

    except Exception as e:
        print(f"   ✗ Test 1 failed: {e}")

    # Test 2: Streaming event types
    print("\nTest 2: Streaming event types")
    try:
        tools = [FunctionTool(fn=multiply_numbers)]

        client = AnthropicAPIClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

        agent = Agent(
            name="EventTypeAgent",
            tools=tools,
            model_client=client,
            model_kwargs={
                "model": "claude-3-5-haiku-20241022",
                "stream": True,
                "max_tokens": 1000,
            },
            answer_data_type=str,
            max_steps=2,
        )

        runner = Runner(agent=agent)

        streaming_result = runner.astream(
            prompt_kwargs={"input_str": "Multiply 6 by 7"}
        )

        async def test_event_types():
            event_types = set()

            async for event in streaming_result.stream_events():
                event_types.add(type(event).__name__)

            # Should see various event types
            expected_types = {"RawResponsesStreamEvent", "RunItemStreamEvent"}
            found_types = event_types.intersection(expected_types)

            assert len(found_types) > 0, "Should receive expected event types"

            print(f"   ✓ Event types found: {event_types}")

        asyncio.run(test_event_types())

    except Exception as e:
        print(f"   ✗ Test 2 failed: {e}")


async def run_all_tests():
    """Run all test cases."""

    print("=" * 60)
    print("RUNNING ALL ANTHROPIC STREAMING TUTORIAL TESTS")
    print("=" * 60)

    try:
        # test_direct_anthropic_streaming()
        # test_async_streaming()
        # test_non_streaming()
        # test_response_compatibility()
        # test_generator_compatibility()
        # test_error_handling()
        # test_different_model_configurations()

        # New Runner and Agent tests
        # test_runner_with_streaming()
        # test_agent_with_streaming()
        # test_runner_agent_integration_streaming()
        test_runner_async_streaming()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"Error running tests: {e}")


if __name__ == "__main__":
    # Run the original tutorial
    # main()

    # Run all test cases
    asyncio.run(run_all_tests())
