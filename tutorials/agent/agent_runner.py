"""
Agent and Runner Tutorial

This tutorial demonstrates all the features of AdalFlow's Agent and Runner components
including:
1. Basic agent setup with calculator tool
2. Web search and email tools
3. Streaming execution
4. Structured output with Pydantic and AdalFlow dataclasses
5. Custom role descriptions
6. Permission management
7. Async operations

Each feature is encapsulated in its own function for easy testing and understanding.
"""

from typing import List, Any, Optional
from dataclasses import dataclass
import asyncio

# AdalFlow imports
from adalflow import Agent, Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import ToolOutput, RunnerResult
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
from adalflow.core.base_data_class import DataClass
from adalflow.utils import get_logger, setup_env

# Pydantic for structured output examples
from pydantic import BaseModel, Field


setup_env()

logger = get_logger(level="INFO", enable_file=False)


# ---------------------------------------------------------------------------
# Basic Tool Definitions
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 1. Basic Agent with Calculator Tool
# ---------------------------------------------------------------------------

def basic_calculator_agent():
    """Demonstrates basic agent setup with a calculator tool."""
    
    # Create agent with calculator tool
    agent = Agent(
        name="CalculatorAgent",
        tools=[calculator],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=6
    )
    
    # Create runner
    runner = Runner(agent=agent)
    
    # Execute task
    result = runner.call(
        prompt_kwargs={"input_str": "Calculate 15 * 7 + 23"}
    )
    
    print(f"Result: {result.answer}")
    print(f"Steps taken: {len(result.step_history)}")
    
    return result


# ---------------------------------------------------------------------------
# 2. Web Search and Email Tools Example
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Simulated search results
    """
    logger.info(f"Searching for: {query}")
    return f"Search results for '{query}': Found relevant information about {query}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.
    
    Args:
        to: Recipient email
        subject: Email subject
        body: Email body
        
    Returns:
        Confirmation message
    """
    logger.info(f"Sending email to {to} with subject: {subject}")
    return f"Email sent successfully to {to}"


def advanced_search(query: str, max_results: int = 5) -> ToolOutput:
    """Advanced search with enhanced agent feedback using ToolOutput.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        ToolOutput with structured information
    """
    # Simulate search results
    results = [f"Result {i+1} for '{query}'" for i in range(max_results)]
    
    return ToolOutput(
        output=results,  # The actual data
        observation=f"Found {len(results)} results for '{query}'",  # For agent reasoning
        display=f"🔍 Searched: {query}",  # For user display
        metadata={"query": query, "count": len(results)}
    )


def multi_tool_agent():
    """Demonstrates agent with multiple tools including search and email."""
    
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
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Search for information about AI agents and then send an email to user@example.com summarizing what you found"
        }
    )
    
    print(f"Result: {result.answer}")
    for i, step in enumerate(result.step_history):
        print(f"Step {i}: {step.action.name if step.action else 'No action'}")
    
    return result


# ---------------------------------------------------------------------------
# 3. Streaming Execution Example
# ---------------------------------------------------------------------------

async def streaming_agent_example():
    """Demonstrates streaming execution for real-time updates."""
    
    def process_data(data: str, operation: str = "analyze") -> str:
        """Process data with specified operation."""
        return f"Processed '{data}' with operation '{operation}'"
    
    agent = Agent(
        name="StreamingAgent",
        tools=[FunctionTool(process_data), FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=3
    )
    
    runner = Runner(agent=agent)
    
    # Note: For actual streaming, you would use runner.stream() method
    # This is a simplified async example
    result = await runner.acall(
        prompt_kwargs={
            "input_str": "Process the data 'customer feedback' and then calculate 100 * 25"
        }
    )
    
    print(f"Streaming result: {result.answer}")
    return result


# ---------------------------------------------------------------------------
# 4. Structured Output Examples
# ---------------------------------------------------------------------------

# Using Pydantic BaseModel
class AnalysisResult(BaseModel):
    """Structured analysis result using Pydantic."""
    summary: str = Field(description="Brief summary of the analysis")
    confidence: float = Field(description="Confidence score between 0 and 1")
    recommendations: List[str] = Field(description="List of recommendations")


def pydantic_output_agent():
    """Demonstrates structured output with Pydantic model."""
    
    def analyze_text(text: str) -> str:
        """Analyze text and return insights."""
        return f"Analysis complete for: {text}. The text appears to be about technology."
    
    agent = Agent(
        name="AnalysisAgent",
        tools=[FunctionTool(analyze_text)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        answer_data_type=AnalysisResult,
        max_steps=3
    )
    
    runner = Runner(agent=agent)
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Analyze the following text and provide recommendations: 'AI is transforming industries'"
        }
    )
    
    print(f"Analysis Result: {result.answer}")
    if isinstance(result.answer, AnalysisResult):
        print(f"Confidence: {result.answer.confidence}")
        print(f"Recommendations: {result.answer.recommendations}")
    
    return result


# Using AdalFlow DataClass
@dataclass
class TaskReport(DataClass):
    """Task execution report using AdalFlow DataClass."""
    task_name: str
    status: str
    execution_time: float
    results: List[str]
    errors: Optional[List[str]] = None


def adalflow_dataclass_output_agent():
    """Demonstrates structured output with AdalFlow DataClass."""
    
    def execute_task(task_name: str) -> str:
        """Execute a named task."""
        import time
        start = time.time()
        # Simulate task execution
        time.sleep(0.1)
        elapsed = time.time() - start
        return f"Task '{task_name}' completed in {elapsed:.2f} seconds"
    
    agent = Agent(
        name="TaskExecutor",
        tools=[FunctionTool(execute_task)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        answer_data_type=TaskReport,
        max_steps=3
    )
    
    runner = Runner(agent=agent)
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Execute the data_processing task and provide a detailed report"
        }
    )
    
    print(f"Task Report: {result.answer}")
    return result


# ---------------------------------------------------------------------------
# 5. Custom Role Description Example
# ---------------------------------------------------------------------------

def custom_role_agent():
    """Demonstrates agent with custom role description."""
    
    custom_role_desc = """
You are a helpful assistant specialized in data analysis.
Always provide step-by-step reasoning and cite your sources.
When using tools, explain why you chose each tool.
Be concise but thorough in your explanations.
"""
    
    def analyze_numbers(numbers: List[float]) -> str:
        """Analyze a list of numbers and return statistics."""
        if not numbers:
            return "No numbers provided"
        
        avg = sum(numbers) / len(numbers)
        return f"Count: {len(numbers)}, Average: {avg:.2f}, Min: {min(numbers)}, Max: {max(numbers)}"
    
    agent = Agent(
        name="DataAnalyst",
        role_desc=custom_role_desc,
        tools=[FunctionTool(analyze_numbers), FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )
    
    runner = Runner(agent=agent)
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Analyze these numbers [10, 25, 30, 15, 40] and then calculate the sum of the min and max values"
        }
    )
    
    print(f"Analysis: {result.answer}")
    return result


# ---------------------------------------------------------------------------
# 6. Different Model Providers Example
# ---------------------------------------------------------------------------

def anthropic_model_agent():
    """Demonstrates using Anthropic Claude model."""
    
    agent = Agent(
        name="ClaudeAgent",
        tools=[FunctionTool(calculator)],
        model_client=AnthropicAPIClient(),
        model_kwargs={"model": "claude-3-sonnet-20240229", "temperature": 0.3},
        max_steps=3
    )
    
    runner = Runner(agent=agent)
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Calculate the factorial of 5 (5!)"
        }
    )
    
    print(f"Claude Result: {result.answer}")
    return result


# ---------------------------------------------------------------------------
# 7. Complex Multi-Step Agent Example
# ---------------------------------------------------------------------------

def file_operations(operation: str, filename: str, content: str = "") -> str:
    """Simulate file operations."""
    if operation == "read":
        return f"Content of {filename}: [simulated file content]"
    elif operation == "write":
        return f"Written to {filename}: {content}"
    else:
        return f"Unknown operation: {operation}"


def database_query(query: str) -> str:
    """Simulate database query."""
    return f"Query result for '{query}': [10 rows returned]"


def complex_multi_step_agent():
    """Demonstrates complex multi-step reasoning with multiple tools."""
    
    tools = [
        FunctionTool(calculator),
        FunctionTool(file_operations),
        FunctionTool(database_query),
        FunctionTool(send_email)
    ]
    
    agent = Agent(
        name="ComplexAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=10  # Allow more steps for complex reasoning
    )
    
    runner = Runner(agent=agent)
    
    complex_task = """
    1. Read the sales data from 'sales_2024.csv'
    2. Query the database for customer count
    3. Calculate the average sales per customer (assume sales total is 50000)
    4. Write the results to 'analysis_report.txt'
    5. Send an email to manager@company.com with the findings
    """
    
    result = runner.call(
        prompt_kwargs={"input_str": complex_task}
    )
    
    print(f"Complex Task Result: {result.answer}")
    print(f"Total steps taken: {len(result.step_history)}")
    
    return result


# ---------------------------------------------------------------------------
# 8. Error Handling Example
# ---------------------------------------------------------------------------

def risky_operation(value: int) -> str:
    """Operation that might fail."""
    if value < 0:
        raise ValueError("Value must be non-negative")
    return f"Successfully processed value: {value}"


def error_handling_agent():
    """Demonstrates agent behavior with tool errors."""
    
    agent = Agent(
        name="ErrorHandlingAgent",
        tools=[FunctionTool(risky_operation), FunctionTool(calculator)],
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5
    )
    
    runner = Runner(agent=agent)
    
    result = runner.call(
        prompt_kwargs={
            "input_str": "Try risky_operation with -5, then if it fails, calculate 10 + 20 instead"
        }
    )
    
    print(f"Result with error handling: {result.answer}")
    return result


# ---------------------------------------------------------------------------
# Main Execution Functions
# ---------------------------------------------------------------------------

def run_all_examples():
    """Run all example functions and display results."""
    
    examples = [
        ("Basic Calculator Agent", basic_calculator_agent),
        ("Multi-Tool Agent", multi_tool_agent),
        ("Pydantic Output Agent", pydantic_output_agent),
        ("AdalFlow DataClass Output Agent", adalflow_dataclass_output_agent),
        ("Custom Role Agent", custom_role_agent),
        ("Complex Multi-Step Agent", complex_multi_step_agent),
        ("Error Handling Agent", error_handling_agent),
    ]
    
    # Optional examples that require API keys
    optional_examples = [
        ("Anthropic Model Agent", anthropic_model_agent),
        ("Streaming Agent", lambda: asyncio.run(streaming_agent_example())),
    ]
    
    results = {}
    
    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        try:
            result = func()
            results[name] = ("SUCCESS", result)
            print(f"✅ {name} completed successfully")
        except Exception as e:
            results[name] = ("FAILED", str(e))
            print(f"❌ {name} failed: {e}")
    
    # Run optional examples with error handling
    print("\n\nRunning optional examples (may fail if API keys not configured)...")
    
    for name, func in optional_examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        try:
            result = func()
            results[name] = ("SUCCESS", result)
            print(f"✅ {name} completed successfully")
        except Exception as e:
            results[name] = ("SKIPPED", str(e))
            print(f"⚠️  {name} skipped: {e}")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for name, (status, _) in results.items():
        emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
        print(f"{emoji} {name}: {status}")
    
    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Exit with appropriate code
    failed_count = sum(1 for _, (status, _) in results.items() if status == "FAILED")
    if failed_count > 0:
        print(f"\n{failed_count} examples failed.")
        exit(1)
    else:
        print("\nAll required examples completed successfully!")