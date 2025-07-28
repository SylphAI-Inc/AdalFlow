"""
Agent Tutorial: Breaking down the agents_runner tutorial into separate functions
This tutorial demonstrates the core concepts of AdalFlow agents with clear, modular examples.

Prerequisites:
1. Install AdalFlow: pip install adalflow
2. Set OpenAI API key: export OPENAI_API_KEY="your-api-key-here"
3. Run the script: python tutorial_agent.py

Note: This script requires an OpenAI API key to function properly.
"""

import os
import sys
from adalflow.components.agent.agent import Agent
from adalflow.components.agent.runner import Runner
from adalflow.components.model_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import ToolOutput
from pydantic import BaseModel
from typing import List
import adalflow

# store your openai key as OPENAI_API_KEY under .env file
adalflow.setup_env()

def check_environment():
    """Check if the required environment variables and dependencies are set up."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  # Or on Windows: set OPENAI_API_KEY=your-api-key-here")
        return False
    
    # Test OpenAI client creation
    try:
        client = OpenAIClient()
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
        print("Please check your API key and internet connection.")
        return False


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


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


def advanced_search(query: str, max_results: int = 5) -> ToolOutput:
    """Advanced search with enhanced agent feedback."""
    results = [f"Result {i}: Info about {query}" for i in range(max_results)]
    return ToolOutput(
        output=results,
        observation=f"Found {len(results)} results for '{query}'",
        display=f"=Searched: {query}",
        metadata={"query": query, "count": len(results)}
    )


class AnalysisResult(BaseModel):
    """Structure for analysis results."""
    summary: str
    confidence: float
    recommendations: List[str]


def basic_agent_example():
    """Demonstrates basic agent creation and usage with a calculator tool."""
    print("\n=== Basic Agent Example ===")
    
    try:
        # Create agent with tools that require permission
        agent = Agent(
            name="PermissionAgent",
            tools=[
                FunctionTool(calculator), 
            ],
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-4o", "temperature": 0.3},
            max_steps=6
        )

        runner = Runner(agent=agent)

        result = runner.call(prompt_kwargs={"input_str": "Invoke the calculator tool and calculate 15 * 7 + 23"})
        
        print(f"Result: {result.answer}")
        print(f"Steps taken: {len(result.step_history)}")
        return result
    except Exception as e:
        print(f"❌ Error in basic_agent_example: {e}")
        print("This example requires a valid OpenAI API key and network connectivity.")
        return None


def multi_tool_agent_example():
    """Demonstrates an agent with multiple tools including basic and advanced search."""
    print("\n=== Multi-Tool Agent Example ===")
    
    tools = [
        FunctionTool(search_web),
        FunctionTool(send_email),
        FunctionTool(advanced_search)
    ]

    agent = Agent(
        name="MultiToolAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    result = runner.call(prompt_kwargs={"input_str": "Search for information about Python programming and then send an email summary"})
    
    print(f"Result: {result.answer}")
    print(f"Steps taken: {len(result.step_history)}")
    return result


def callable_tools_example():
    """Demonstrates using callables directly as tools (without FunctionTool wrapper)."""
    print("\n=== Callable Tools Example ===")
    
    # Using callables directly without FunctionTool wrapper
    tools = [
        search_web,
        send_email,
        advanced_search
    ]

    agent = Agent(
        name="CallableAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    result = runner.call(prompt_kwargs={"input_str": "Search for machine learning tutorials"})
    
    print(f"Result: {result.answer}")
    print(f"Steps taken: {len(result.step_history)}")
    return result


def structured_output_example():
    """Demonstrates agent with structured output using Pydantic models."""
    print("\n=== Structured Output Example ===")
    
    agent = Agent(
        name="AnalysisAgent",
        tools=[FunctionTool(search_web), FunctionTool(advanced_search)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        answer_data_type=AnalysisResult,
        max_steps=5
    )

    runner = Runner(agent=agent)

    result = runner.call(prompt_kwargs={"input_str": "Analyze the current state of artificial intelligence research"})
    
    print(f"Analysis Result: {result.answer}")
    if isinstance(result.answer, AnalysisResult):
        print(f"Summary: {result.answer.summary}")
        print(f"Confidence: {result.answer.confidence}")
        print(f"Recommendations: {result.answer.recommendations}")
    
    return result


def custom_role_example():
    """Demonstrates agent with custom role description."""
    print("\n=== Custom Role Example ===")
    
    custom_role_desc = """
You are a helpful assistant specialized in data analysis.
Always provide step-by-step reasoning and cite your sources.
When using tools, explain why you chose each tool.
"""

    agent = Agent(
        name="DataAnalyst",
        role_desc=custom_role_desc,
        tools=[FunctionTool(calculator), FunctionTool(search_web)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )

    runner = Runner(agent=agent)

    result = runner.call(prompt_kwargs={"input_str": "Calculate the average of 10, 20, 30, 40, 50 and explain your process"})
    
    print(f"Result: {result.answer}")
    print(f"Steps taken: {len(result.step_history)}")
    return result


def runner_configuration_example():
    """Demonstrates different Runner configurations."""
    print("\n=== Runner Configuration Example ===")
    
    agent = Agent(
        name="ConfigAgent",
        tools=[FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=10
    )

    # Override agent's max_steps with runner configuration
    runner = Runner(
        agent=agent,
        max_steps=3  # Override agent's max_steps
    )

    result = runner.call(prompt_kwargs={"input_str": "Calculate 5 + 3 * 2 and then multiply by 4"})
    
    print(f"Result: {result.answer}")
    print(f"Steps taken: {len(result.step_history)}")
    print(f"Error: {result.error}")
    print(f"Context: {result.ctx}")
    return result


def main():
    """Run all tutorial examples."""
    print("🚀 Starting Agent Tutorial Examples...")
    print("=" * 50)
    
    # Check environment setup first
    if not check_environment():
        print("\n❌ Environment setup failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    print("\n🔧 Environment setup complete!\n")
    
    try:
        # Run all examples safely
        examples = [
            (basic_agent_example, "Basic Agent Example"),
            (multi_tool_agent_example, "Multi-Tool Agent Example"),
            (callable_tools_example, "Callable Tools Example"),
            (structured_output_example, "Structured Output Example"),
            (custom_role_example, "Custom Role Example"),
            (runner_configuration_example, "Runner Configuration Example")
        ]
        
        successful_examples = 0
        for example_func, example_name in examples:
            result = run_example_safely(example_func, example_name)
            if result is not None:
                successful_examples += 1
        
        print("\n" + "=" * 50)
        print("🎉 Tutorial Complete!")
        print(f"Successfully ran {successful_examples}/{len(examples)} examples!")
        print("\nNext steps:")
        print("- Explore the source code to understand each example")
        print("- Modify the tools and prompts to experiment")
        print("- Check the AdalFlow documentation for more features")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tutorial interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nThis might be due to:")
        print("- Invalid API key")
        print("- Network connectivity issues")
        print("- API rate limits")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()