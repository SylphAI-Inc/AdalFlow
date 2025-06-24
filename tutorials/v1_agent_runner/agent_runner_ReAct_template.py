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

import logging
import asyncio
from typing import Dict, Any, List, Optional, Type, TypeVar
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Import AdalFlow components
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.tool_manager import ToolManager
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import Function, StepOutput, GeneratorOutput
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.core.component import DataComponent
import os

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')

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

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")
        
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

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
                fn=calculate,
            )
        ]

        api_key = os.getenv("OPENAI_API_KEY")

        model_client = OpenAIClient(api_key=api_key)

        output_parser = JsonOutputParser(
            data_class=Function,
            return_data_class=True,
            include_fields=[
                "action",
                "function",
                "observation",
            ],
        )

        
        # Create the ReAct agent
        agent = Agent(
            name="ReActAgent",
            tools=tools,
            model_client=model_client,
            # use default template and use default task description 
            prompt_kwargs={
                "tool_definitions": ToolManager(tools).yaml_definitions,
            },
            output_processors=output_parser, 
        )
        
        # Create the runner
        # use default executor 
        runner = Runner(planner=agent, max_steps=5)
        
        # Example query
        query = "What is 5 * 5000 + 10000? Then search for information about the result."
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        # Run the agent
        history, result = runner.call(
            prompt_kwargs={
                "input_str": query, 
                "output_format_str": output_parser.format_instructions(),
            },
            model_kwargs={"model": "gpt-4o-mini", "temperature": 0.7}
        )
        
        # Print results
        print("\nSTEP HISTORY:")
        for i, step in enumerate(history, 1):
            print(f"\nStep {i}:")
            print(f"- Action: {getattr(step.data, 'action', 'N/A')}")
            if hasattr(step.data, 'function'):
                print(f"- Function: {step.data.function.name if step.data.function else 'N/A'}")
            print(f"- Observation: {getattr(step.data, 'observation', 'N/A')}")
        
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
                name="search",
                description="Search the web for information. Input should be a search query."
            ),
            FunctionTool(
                fn=calculate,
                name="calculate",
                description="Evaluate a mathematical expression. Input should be a valid mathematical expression."
            )
        ]
        
        # Create the ReAct agent
        agent = create_react_agent(
            tools=tools,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Create the runner
        runner = Runner(planner=agent, max_steps=5)
        
        # Example query
        query = "What is 5 * 5? Then search for information about the result."
        print(f"\n{'='*80}")
        print(f"ASYNC QUERY: {query}")
        print(f"{'='*80}")
        
        # Run the agent asynchronously
        history, result = await runner.acall(
            prompt_kwargs={"input": query},
            model_kwargs={"temperature": 0.7, "max_tokens": 500}
        )
        
        # Print results
        print("\nASYNC STEP HISTORY:")
        for i, step in enumerate(history, 1):
            print(f"\nStep {i}:")
            print(f"- Action: {getattr(step.data, 'action', 'N/A')}")
            if hasattr(step.data, 'function'):
                print(f"- Function: {step.data.function.name if step.data.function else 'N/A'}")
            print(f"- Observation: {getattr(step.data, 'observation', 'N/A')}")
        
        print("\nASYNC FINAL RESULT:")
        print(result)
        
        return history, result
        
    except Exception as e:
        logger.error(f"Error running async ReAct agent example: {str(e)}")
        raise

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run synchronous example
    print("Running synchronous ReAct agent example...")
    run_react_agent_example()
    
    # Run async example
    print("\n" + "="*80)
    print("Running async ReAct agent example...")
    asyncio.run(arun_react_agent_example())
