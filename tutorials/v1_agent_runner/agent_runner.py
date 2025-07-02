"""
ReAct (Reasoning and Acting) Agent Runner Tutorial

This tutorial demonstrates how to use the Runner class with a ReAct (Reasoning and Acting) agent
for multi-step reasoning and tool usage. The example shows how to:
1. Define custom tools for the agent
2. Set up a ReAct agent with the tools
3. Use the Runner for multi-step execution
4. Handle function calls and observations
5. Process and display results
"""

from typing import TypeVar, List, Any, Optional

# Import AdalFlow components
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.components.model_client.openai_client import OpenAIClient

from adalflow.core.base_data_class import DataClass
from dataclasses import dataclass

from adalflow.utils import get_logger

import asyncio


from load_dotenv import load_dotenv

load_dotenv()

logger = get_logger(level="DEBUG", enable_file=False)

# Type variable for generic return type
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------


def search_tool(query: str) -> str:
    """Search for information on the web.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """
    # In a real implementation, this would call a search API
    logger.info(f"Searching for: {query}")
    return "Search has been completed"


def add_tool(x: int, y: int) -> int:
    return x + y


def sub_tool(x: int, y: int) -> int:
    return x - y


async def async_sub_tool(x: int, y: int) -> int:
    return x - y


async def async_multiply_tool(x: int, y: int) -> int:
    return x * y


async def async_divide_tool(x: int, y: int) -> int:
    return x / y


def square_root_tool(x: int) -> int:
    return x**0.5


# ---------------------------------------------------------------------------
# ReAct Agent Setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------


def run_react_agent_example():
    """Run an example of the ReAct agent with multiple tools."""
    try:
        # Create tool instances
        tools = [
            FunctionTool(
                fn=search_tool,
            ),
            FunctionTool(
                fn=add_tool,
            ),
            FunctionTool(
                fn=sub_tool,
            ),
            FunctionTool(
                fn=async_sub_tool,
            ),
            FunctionTool(
                fn=async_multiply_tool,
            ),
            FunctionTool(
                fn=async_divide_tool,
            ),
            FunctionTool(
                fn=square_root_tool,
            ),
        ]

        @dataclass
        class Summary(DataClass):
            """Stores the results of an agent's execution."""

            step_count: int
            final_output: Any
            search_tool_calls: List[str]

        @dataclass
        class Person(DataClass):
            name: str
            age: int
            summary: Summary

        # Create the ReAct agent
        agent = Agent(
            name="ReActAgent",
            tools=tools,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
            answer_data_type=Person,
        )

        # Create the runner
        # use default executor
        runner = Runner(agent=agent, max_steps=5)

        # Example query
        query = (
            "What is 5 * 5000 + 10000? Then search for information about the result."
        )
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")

        # Run the agent
        history, result = runner.call(
            prompt_kwargs={
                "input_str": query,
            },
        )

        # Print results
        print("\nSTEP HISTORY:")
        for i, step in enumerate(history):
            print(f"\nStep {i}:")
            print(step)
        print("\nFINAL RESULT (RunnerResponse):")
        print(f"Answer: {result.answer}")
        print(f"Function Call: {result.function_call}")
        print(f"Function Call Result: {result.function_call_result}")
        print(f"Full Result: {result}")

        return history, result

    except Exception as e:
        logger.error(f"Error running ReAct agent example: {str(e)}")
        raise


def run_react_agent_primitive_type():
    """Run an example of the ReAct agent with multiple tools."""
    try:
        # Create tool instances
        tools = [
            FunctionTool(
                fn=search_tool,
            ),
            FunctionTool(
                fn=add_tool,
            ),
            FunctionTool(
                fn=sub_tool,
            ),
            FunctionTool(
                fn=async_sub_tool,
            ),
            FunctionTool(
                fn=async_multiply_tool,
            ),
            FunctionTool(
                fn=async_divide_tool,
            ),
            FunctionTool(
                fn=square_root_tool,
            ),
        ]

        # Create the ReAct agent
        agent = Agent(
            name="ReActAgent",
            tools=tools,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3},
            # answer_data_type=list,
            answer_data_type=dict,
        )

        # Create the runner
        # use default executor
        runner = Runner(agent=agent, max_steps=5)

        # Example query
        query = (
            "What is 5 * 5000 + 10000? Then search for information about the result."
        )
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")

        # Run the agent
        history, result = runner.call(
            prompt_kwargs={
                "input_str": query,
            },
        )

        # Print results
        print("\nSTEP HISTORY:")
        for i, step in enumerate(history):
            print(f"\nStep {i}:")
            print(step)
        print("\nFINAL RESULT (RunnerResponse):")
        print(f"Answer: {result.answer}")
        print(f"Function Call: {result.function_call}")
        print(f"Function Call Result: {result.function_call_result}")
        print(f"Full Result: {result}")

        return history, result

    except Exception as e:
        logger.error(f"Error running ReAct agent example: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Async Example
# ---------------------------------------------------------------------------


async def arun_react_agent_example():
    """Run an async example of the ReAct agent."""
    try:
        # Create tool instances
        tools = [
            FunctionTool(
                fn=search_tool,
            ),
            FunctionTool(
                fn=add_tool,
            ),
            FunctionTool(
                fn=async_sub_tool,
            ),
            FunctionTool(
                fn=async_multiply_tool,
            ),
            FunctionTool(
                fn=async_divide_tool,
            ),
            FunctionTool(
                fn=square_root_tool,
            ),
        ]

        @dataclass
        class Summary(DataClass):
            """Stores the results of an agent's execution."""

            step_count: int
            final_output: Any
            search_tool_calls: List[str]

        # Create the ReAct agent
        agent = Agent(
            name="AsyncReActAgent",
            tools=tools,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
            answer_data_type=Summary,
        )

        # Create the runner
        runner = Runner(agent=agent, max_steps=5)

        # Example query
        query = "What is 5 * 5? Then search for information about the result."
        print(f"\n{'='*80}")
        print(f"ASYNC QUERY: {query}")
        print(f"{'='*80}")

        # Run the agent asynchronously
        history, result = await runner.acall(
            prompt_kwargs={
                "input_str": query,
            },
        )

        # Print results
        print("\nASYNC STEP HISTORY:")
        for i, step in enumerate(history):
            print(f"\nStep {i}:")
            print(step)

        print("\nASYNC FINAL RESULT (RunnerResponse):")
        print(f"Answer: {result.answer}")
        print(f"Function Call: {result.function_call}")
        print(f"Function Call Result: {result.function_call_result}")
        print(f"Full Result: {result}")

        return history, result

    except Exception as e:
        logger.error(f"Error running async ReAct agent example: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Nested DataClass Structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ReAct Agent Runner Example
# ---------------------------------------------------------------------------


def run_advanced_react_agent():
    # Create tool instances
    tools = [
        FunctionTool(fn=search_tool),
        FunctionTool(fn=add_tool),
        FunctionTool(fn=async_sub_tool),
        FunctionTool(fn=async_multiply_tool),
        FunctionTool(fn=async_divide_tool),
        FunctionTool(fn=square_root_tool),
    ]

    # Instantiate the agent with the complex answer data type
    # agent = Agent(
    #     name="AdvancedReActAgent",
    #     tools=tools,
    #     model_client=OpenAIClient(),
    #     model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.5},
    #     answer_data_type=Organization,
    # )

    agent = Agent(
        name="AdvancedReActAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.5},
        answer_data_type=PersonSummary,
    )

    # Allow up to 10 reasoning/tool steps
    runner = Runner(agent=agent, max_steps=10)

    # Complex query that triggers multiple tool invocations
    query = (
        "Compute (12 * 12) and then search for information about '144 puzzle'. "
        "Finally, calculate the square root of your first result."
    )

    print("RUNNING ADVANCED SYNCHRONOUS REACT AGENT EXAMPLE")
    history, result = runner.call(prompt_kwargs={"input_str": query})

    # Display results
    print("\nSTEP HISTORY SUMMARY:")
    for step in history:
        print(step)

    print("\nFINAL RESULT (RunnerResponse):")
    print(f"Answer: {result.answer}")
    print(f"Function Call: {result.function_call}")
    print(f"Function Call Result: {result.function_call_result}")
    print(f"Full Result: {result}")

    return history, result


# ---------------------------------------------------------------------------
# Async Example
# ---------------------------------------------------------------------------


async def arun_advanced_react_agent():
    tools = [
        FunctionTool(fn=search_tool),
        FunctionTool(fn=add_tool),
        FunctionTool(fn=async_sub_tool),
        FunctionTool(fn=async_multiply_tool),
        FunctionTool(fn=async_divide_tool),
        FunctionTool(fn=square_root_tool),
    ]

    agent = Agent(
        name="AsyncAdvancedReActAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.5},
        # answer_data_type=Organization,
        answer_data_type=PersonSummary,
    )

    # Increase max_steps to allow deeper reasoning
    runner = Runner(agent=agent, max_steps=12)

    query = "Find 7 * 6, search details for '42 meaning', then summarize how the number appears in popular culture."

    print("RUNNING ADVANCED ASYNC REACT AGENT EXAMPLE")
    history, result = await runner.acall(prompt_kwargs={"input_str": query})

    print("\nASYNC STEP HISTORY SUMMARY:")
    for step in history:
        print(step)

    print(result)

    print("\nFINAL RESULT (RunnerResponse):")
    print(f"Answer: {result.answer}")
    print(f"Function Call: {result.function_call}")
    print(f"Function Call Result: {result.function_call_result}")
    print(f"Full Result: {result}")

    return history, result


def no_structured_output_run_agent():
    """Run an agent with no structured output (returns raw string)."""
    try:
        # Create tool instances
        tools = [
            FunctionTool(fn=add_tool),
            FunctionTool(fn=sub_tool),
            FunctionTool(fn=square_root_tool),
        ]

        # Create the agent with no answer_data_type (returns raw string)
        agent = Agent(
            name="NoStructuredOutputAgent",
            tools=tools,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
            answer_data_type=float,
        )

        # Create the runner
        runner = Runner(agent=agent, max_steps=5)

        # Example query
        query = "Calculate the square root of (25 + 10 - 5)"
        print(f"\n{'='*80}")
        print(f"NO STRUCTURED OUTPUT QUERY: {query}")
        print(f"{'='*80}")

        # Run the agent
        history, result = runner.call(prompt_kwargs={"input_str": query})

        # Print results
        print("\nFINAL RESULT (RunnerResponse):")
        print(f"Answer: {result.answer}")
        print(f"Function Call: {result.function_call}")
        print(f"Function Call Result: {result.function_call_result}")
        print(f"Result type: {type(result).__name__}")
        print(f"Full Result: {result}")

        return history, result

    except Exception as e:
        logger.error(f"Error in no_structured_output_run_agent: {str(e)}")
        raise


def pydantic_dataclass_run_agent():
    """Run an agent with Pydantic model as output type."""
    from pydantic import BaseModel, Field
    from typing import List

    # Define Pydantic models for structured output
    class CalculationStep(BaseModel):
        step_number: int
        operation: str
        result: float

    class CalculationResult(BaseModel):
        final_answer: float
        steps: List[CalculationStep]
        is_correct: bool = Field(
            ..., description="Whether the calculation appears correct"
        )

    try:
        # Create tool instances
        tools = [
            FunctionTool(fn=add_tool),
            FunctionTool(fn=sub_tool),
            FunctionTool(fn=async_multiply_tool),
            FunctionTool(fn=square_root_tool),
        ]

        # Create the agent with Pydantic model as output type
        agent = Agent(
            name="PydanticOutputAgent",
            tools=tools,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3},
            answer_data_type=CalculationResult,
        )

        # Create the runner
        runner = Runner(agent=agent, max_steps=5)

        # Example query
        query = "Calculate ((5 + 3) * 2) - 4 and show your work step by step"
        print(f"\n{'='*80}")
        print(f"PYDANTIC OUTPUT QUERY: {query}")
        print(f"{'='*80}")

        # Run the agent
        history, result = runner.call(prompt_kwargs={"input_str": query})

        # Print results
        print("\nFINAL RESULT (RunnerResponse):")
        print(f"Answer: {result.answer}")
        print(f"Function Call: {result.function_call}")
        print(f"Function Call Result: {result.function_call_result}")
        print(f"Full Result: {result}")

        return history, result

    except Exception as e:
        logger.error(f"Error in pydantic_dataclass_run_agent: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # Run synchronous example
#     print("Running synchronous ReAct agent example...")
#     run_react_agent_example()

#     # print("\n" + "=" * 80)
#     # print("Running primitive type ReAct agent example...")
#     # run_react_agent_primitive_type()

#     # # Run async example
#     # print("\n" + "=" * 80)
#     # print("Running async ReAct agent example...")
#     # asyncio.run(arun_react_agent_example())

#     # print("\n" + "=" * 80)
#     # print("Running advanced ReAct agent example...")
#     # run_advanced_react_agent()

#     # print("\n" + "=" * 80)
#     # print("Running advanced async ReAct agent example...")
#     # asyncio.run(arun_advanced_react_agent())

#     # print("\n" + "=" * 80)
#     # print("Running no structured output agent example...")
#     # no_structured_output_run_agent()

#     # print("\n" + "=" * 80)
#     # print("Running pydantic dataclass agent example...")
#     # pydantic_dataclass_run_agent()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

outputs = {}


def run_example(example_name: str, example_func, *args, **kwargs):
    """Helper function to run an example and track its success/failure."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {example_name}")
    print(f"{'='*80}")

    try:
        start_time = time.time()
        result = example_func(*args, **kwargs)
        elapsed = time.time() - start_time

        print(f"\nSUCCESS: {example_name} completed in {elapsed:.2f} seconds")
        if result:
            history, output = result
            print("\nOUTPUT (RunnerResponse):")
            print(f"Answer: {output.answer}")
            print(f"Function Call: {output.function_call}")
            print(f"Function Call Result: {output.function_call_result}")
            outputs[example_name] = output
            print(f"\nOutput type: {type(output).__name__}")
        return True
    except Exception as e:
        print(f"\n FAILED: {example_name}")
        print(f"Error: {str(e)}")
        logger.exception(f"Error in {example_name}:")
        return False


if __name__ == "__main__":
    import time
    from typing import Dict, List, Any

    # Define all examples to run
    examples = [
        # ("Synchronous ReAct Agent", run_react_agent_example),
        # ("Primitive Type ReAct Agent", run_react_agent_primitive_type),
        # ("Async ReAct Agent", lambda: asyncio.run(arun_react_agent_example())),
        # ("Advanced ReAct Agent", run_advanced_react_agent),
        (
            "Advanced Async ReAct Agent",
            lambda: asyncio.run(arun_advanced_react_agent()),
        ),
        # ("No Structured Output Agent", no_structured_output_run_agent),
        # ("Pydantic Dataclass Agent", pydantic_dataclass_run_agent),
    ]

    # Run all examples and track results
    results: Dict[str, bool] = {}
    for name, func in examples:
        results[name] = run_example(name, func)

    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    for name, success in results.items():

        if success:
            print(f"\n✅ SUCCESS: {name}")
            output = outputs[name]
            print(f"Answer: {output.answer}")
            print(f"Function Call: {output.function_call}")
            print(f"Function Call Result: {output.function_call_result}")
        else:
            print(f"\n❌ FAILED: {name}")

    # Exit with non-zero code if any example failed
    if not all(results.values()):
        print("\nSome examples failed. Check the logs for details.")
        exit(1)
