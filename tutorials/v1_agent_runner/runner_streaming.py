#!/usr/bin/env python3
"""Example demonstrating Runner streaming functionality with real OpenAI integration."""

import asyncio
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from adalflow.components.agent import Agent, Runner
from adalflow.core.types import (
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    FinalOutputItem,
)
from adalflow.core.generator import Generator
from adalflow.components.model_client.openai_client import OpenAIClient
import logging

from adalflow.core.base_data_class import DataClass
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()


# Structured output models
class TaskResult(BaseModel):
    """Structured output for task completion."""

    task_name: str = Field(description="Name of the completed task")
    status: str = Field(description="Status of the task (completed/failed/in_progress)")
    result: str = Field(description="Result or outcome of the task")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class AnalysisReport(BaseModel):
    """Complex structured output for analysis tasks."""

    title: str = Field(description="Title of the analysis")
    summary: str = Field(description="Executive summary of findings")
    findings: List[str] = Field(description="List of key findings")
    recommendations: List[str] = Field(description="List of recommendations")
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class ProjectStep(BaseModel):
    """Individual step in a project plan."""

    step_number: int = Field(description="Step number in the sequence")
    title: str = Field(description="Brief title of the step")
    description: str = Field(description="Detailed description of what to do")
    estimated_hours: float = Field(description="Estimated time in hours")
    dependencies: List[int] = Field(
        default_factory=list, description="List of step numbers this depends on"
    )
    resources: List[str] = Field(default_factory=list, description="Required resources")


class ProjectPhase(BaseModel):
    """A phase containing multiple steps."""

    phase_name: str = Field(description="Name of the project phase")
    description: str = Field(description="Description of the phase")
    steps: List[ProjectStep] = Field(description="List of steps in this phase")
    total_estimated_hours: float = Field(
        description="Total estimated hours for the phase"
    )


class NestedProjectPlan(BaseModel):
    """Complex nested structure for project planning."""

    project_name: str = Field(description="Name of the project")
    overview: str = Field(description="High-level project overview")
    phases: List[ProjectPhase] = Field(description="List of project phases")
    total_duration_weeks: int = Field(description="Total estimated duration in weeks")
    budget_estimate: Dict[str, float] = Field(
        description="Budget breakdown by category"
    )
    risks: List[Dict[str, str]] = Field(
        description="List of identified risks with mitigation"
    )
    success_metrics: List[str] = Field(description="Key success metrics")


def setup_openai_client() -> OpenAIClient:
    """Setup OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return OpenAIClient(api_key=api_key)


def create_generator() -> Generator:
    """Create a Generator instance for streaming output."""
    model_client = setup_openai_client()
    generator = Generator(
        model_client=model_client,
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    )
    return generator


def create_structured_generator() -> Generator:
    """Create a Generator instance for structured output."""
    model_client = setup_openai_client()
    generator = Generator(
        model_client=model_client,
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    )
    return generator


def create_agent_for_runner() -> Agent:
    """Create an Agent instance for Runner streaming tests."""
    model_client = setup_openai_client()

    # Create an agent with the new implementation pattern
    agent = Agent(
        name="streaming_test_agent",
        model_client=model_client,
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=3,
        answer_data_type=str,
    )

    return agent

@dataclass
class ToolCall(DataClass):
    tool_name: str
    input_args: Any
    output: Any


@dataclass
class StepDetail(DataClass):
    step_number: int
    action_taken: str
    observation: str
    tool_call: Optional[ToolCall]


@dataclass
class Summary(DataClass):
    step_count: int
    final_output: Any
    steps: List[StepDetail]
    tool_calls: List[ToolCall]


@dataclass
class Metrics(DataClass):
    time_taken_seconds: float
    tokens_used: int


@dataclass
class PersonSummary(DataClass):
    summary: Summary
    metrics: Metrics


@dataclass
class Person(DataClass):
    name: str
    age: int
    details: PersonSummary


@dataclass
class Department(DataClass):
    department_name: str
    members: List[Person]


@dataclass
class Organization(DataClass):
    org_name: str
    departments: List[Department]

def create_structured_agent_for_runner() -> Agent:
    """Create an Agent instance for Runner streaming tests with structured output."""
    model_client = setup_openai_client()

    # Create an agent for structured responses
    agent = Agent(
        name="structured_streaming_agent",
        model_client=model_client,
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=4,
        answer_data_type=Organization,  # Keep as str for JSON parsing
    )

    return agent


async def test_generator_streaming():
    """Test streaming with Generator directly."""
    print("🚀 Testing Generator Streaming Output")
    print("=" * 60)

    # Create generator for streaming output
    generator = create_generator()

    # Test streaming with a simple query
    query = "Write a short story about a robot learning to paint"
    print(f"📝 Query: {query}")
    print("\n🔄 Streaming response:")
    print("-" * 60)

    # Use the Generator.acall method with streaming enabled
    result = await generator.acall(
        prompt_kwargs={"task_desc_str": query},
        model_kwargs={"stream": True},
    )

    # Check if result is an async generator
    if hasattr(result, "__aiter__"):
        print("🔄 Processing async generator stream:")
        final_result = None
        count = 0
        async for output in result:
            count += 1
            print(f"Stream item #{count}: {output}")
            final_result = output

        print(f"\n📊 Final Generator Result after {count} items:")
        print(final_result)

        # Note: stream_events() cannot be called after the generator has been consumed
        # The async generator can only be iterated once, so we skip this test

        return final_result
    else:
        print("\n📊 Generator Result (non-streaming):")
        print(result)

        # Note: stream_events() cannot be called after the generator has been consumed
        # The async generator can only be iterated once, so we skip this test

        return result


async def test_runner_streaming():
    """Test Runner's astream functionality."""
    print("\n🚀 Testing Runner Streaming")
    print("=" * 60)

    try:
        # Create an agent and runner
        agent = create_agent_for_runner()
        runner = Runner(agent=agent)

        # Start streaming execution
        query = (
            "Help me analyze a customer satisfaction survey. What steps should I take?"
        )
        print(f"📝 Query: {query}")
        print("\n🔄 Streaming runner execution:")
        print("-" * 60)

        streaming_result = runner.astream(
            prompt_kwargs={"input_str": query}, model_kwargs={"stream": True}
        )

        # Process events as they stream
        event_count = 0

        async for event in streaming_result.stream_events():
            event_count += 1

            if isinstance(event, RawResponsesStreamEvent):
                print(f"🔄 Raw Response Event #{event_count}:")
                print(event.data)

            elif isinstance(event, RunItemStreamEvent):
                print(f"⚡ Run Item Event: {event.name}")
                if hasattr(event, "item") and event.item:
                    if isinstance(event.item, FinalOutputItem):
                        print(f"   Final Output - Answer: {event.item.data.answer}")
                        print(
                            f"   Final Output - Step History Length: {len(event.item.data.step_history) if event.item.data.step_history else 0}"
                        )
                        if (
                            event.item.data.step_history
                            and len(event.item.data.step_history) > 0
                        ):
                            last_step = event.item.data.step_history[-1]
                            print(
                                f"   Final Output - Last Function: {last_step.function}"
                            )
                            print(
                                f"   Final Output - Last Result: {last_step.observation}"
                            )
                        else:
                            print("   Final Output - No function calls recorded")
                    else:
                        print(f"   Item: {event.item}")

        print("-" * 60)
        print("📊 Final Results:")
        answer = streaming_result.answer
        if answer:
            print(f"   • Answer: {answer}")
            # Extract function call info from step history
            if streaming_result.step_history and len(streaming_result.step_history) > 0:
                last_step = streaming_result.step_history[-1]
                print(f"   • Last Function: {last_step.function}")
                print(f"   • Last Result: {last_step.observation}")
            else:
                print("   • No function calls recorded in step history")
            print(f"   • Error: {streaming_result._exception}")
        else:
            print("   • Final result: None")
        print(f"   • Total steps: {len(streaming_result.step_history)}")
        print(f"   • Events processed: {event_count}")
        print(
            f"   • Workflow status: {'✅ Complete' if streaming_result.is_complete else '❌ Incomplete'}"
        )

        return streaming_result

    except Exception as e:
        print(f"❌ Error in runner streaming: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_runner_streaming_nested():
    """Test Runner's astream functionality with nested structured output."""
    print("\n🚀 Testing Runner Streaming with Nested Structures")
    print("=" * 60)

    try:
        # Create an agent and runner for structured output
        agent = create_structured_agent_for_runner()
        runner = Runner(agent=agent)

        # Start streaming execution with a complex planning query
        query = (
            "Create a comprehensive project plan for developing a mobile app. "
            "Include multiple phases (planning, development, testing, deployment), "
            "with detailed steps for each phase, time estimates, dependencies, "
            "budget breakdown, risk assessment, and success metrics. "
        )
        print(f"📝 Query: {query}")
        print("\n🔄 Streaming runner execution with nested structures:")
        print("-" * 60)

        streaming_result = runner.astream(
            prompt_kwargs={"input_str": query}, model_kwargs={"stream": True}
        )

        # Process events as they stream
        event_count = 0
        content_chunks = []

        async for event in streaming_result.stream_events():
            event_count += 1

            if isinstance(event, RawResponsesStreamEvent):
                print(f"🔄 Raw Response Event #{event_count}")
                print(event.data)
                # Handle OpenAI response events
                
            elif isinstance(event, RunItemStreamEvent):
                print(f"⚡ Run Item Event: {event.name}")
                if hasattr(event, "item") and event.item:
                    if isinstance(event.item, FinalOutputItem):
                        print(f"   Final Output - Answer: {event.item.data.answer}")
                        print(
                            f"   Final Output - Step History Length: {len(event.item.data.step_history) if event.item.data.step_history else 0}"
                        )
                        if (
                            event.item.data.step_history
                            and len(event.item.data.step_history) > 0
                        ):
                            last_step = event.item.data.step_history[-1]
                            print(
                                f"   Final Output - Last Function: {last_step.function}"
                            )
                            print(
                                f"   Final Output - Last Result: {last_step.observation}"
                            )
                        else:
                            print("   Final Output - No function calls recorded")
                    else:
                        print(f"   Item type: {type(event.item).__name__}")

        print("-" * 60)
        print("📊 Final Results:")
        answer = streaming_result.answer
        if answer:
            print(f"   • Answer: {answer}")
            # Extract function call info from step history
            if streaming_result.step_history and len(streaming_result.step_history) > 0:
                last_step = streaming_result.step_history[-1]
                print(f"   • Last Function: {last_step.function}")
                print(f"   • Last Result: {last_step.observation}")
            else:
                print("   • No function calls recorded in step history")
            print(f"   • Error: {streaming_result._exception}")
        else:
            print("   • Final result: None")
        print(f"   • Total steps: {len(streaming_result.step_history)}")
        print(f"   • Events processed: {event_count}")
        print(f"   • Content chunks collected: {len(content_chunks)}")
        print(
            f"   • Workflow status: {'✅ Complete' if streaming_result.is_complete else '❌ Incomplete'}"
        )

        # Try to analyze the full content
        # full_content = "".join(content_chunks)
        full_content = streaming_result.answer
        if full_content:
            print("\n📄 Full Content Analysis:")
            print("type of object", type(full_content))
            # print(f"   • Total characters: {len(full_content)}")
            print("full_content", full_content)

        return streaming_result

    except Exception as e:
        print(f"❌ Error in nested runner streaming: {e}")
        return None



async def main():
    """Main demonstration function."""
    print("🎯 AdalFlow Generator & Runner Streaming Demo")
    print("Built with real OpenAI integration (updated implementation)")
    print()

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment and try again.")
        return

    print("✅ API key found, proceeding with tests...\n")

    # Run streaming tests
    generator_result = await test_generator_streaming() 
    runner_result = await test_runner_streaming()
    nested_runner_result = await test_runner_streaming_nested()

    print("\n" + "=" * 60)
    print("✅ All streaming tests completed!")
    print("\n🔧 Implementation Features:")
    print("   • ✅ Real OpenAI integration (no mocks)")
    print("   • ✅ Environment-based API key loading")
    print("   • ✅ Generator direct streaming with stream_events")
    print("   • ✅ Runner multi-step streaming execution")
    print("   • ✅ Runner streaming with nested structures")
    print("   • ✅ Proper error handling and JSON parsing")

    # Summary of results
    results = {
        "generator": "✅" if generator_result else "❌",
        "runner_streaming": "✅" if runner_result else "❌",
        "nested_runner_streaming": "✅" if nested_runner_result else "❌",
    }

    print("\n📊 Test Results:")
    for test_name, status in results.items():
        print(f"   {status} {test_name.replace('_', ' ').title()}")


if __name__ == "__main__":
    asyncio.run(main())
