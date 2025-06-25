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

from adalflow.utils import setup_env, get_logger

setup_env()

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
    return f"Search results for '{query}': [Result 1, Result 2, Result 3]"


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
            model_kwargs={"model": "gpt-4o-mini", "temperature": 0.7},
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
        print("\nFINAL RESULT:")
        print(result)

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
            model_kwargs={"model": "gpt-4o-mini", "temperature": 0.7},
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

        print("\nASYNC FINAL RESULT:")
        print(result)

        return history, result

    except Exception as e:
        logger.error(f"Error running async ReAct agent example: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Nested Dataclass Structures
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
    agent = Agent(
        name="AdvancedReActAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.5},
        answer_data_type=Organization,
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

    print("\nFINAL RESULT:")
    print(result)

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
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.5},
        answer_data_type=Organization,
    )

    # Increase max_steps to allow deeper reasoning
    runner = Runner(agent=agent, max_steps=12)

    query = "Find 7 * 6, search details for '42 meaning', then summarize how the number appears in popular culture."

    print("RUNNING ADVANCED ASYNC REACT AGENT EXAMPLE")
    history, result = await runner.acall(prompt_kwargs={"input_str": query})

    print("\nASYNC STEP HISTORY SUMMARY:")
    for step in history:
        print(step)

    print("\nFINAL RESULT")
    print(result)

    return history, result


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run synchronous example
    print("Running synchronous ReAct agent example...")
    run_react_agent_example()

    # Run async example
    # print("\n" + "=" * 80)
    # print("Running async ReAct agent example...")
    # asyncio.run(arun_react_agent_example())

    # print("\n" + "=" * 80)
    # print("Running advanced ReAct agent example...")
    # run_advanced_react_agent()

    # print("\n" + "=" * 80)
    # print("Running advanced async ReAct agent example...")
    # asyncio.run(arun_advanced_react_agent())
