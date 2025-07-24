# Agents and Runner

Agents are the core building block for creating autonomous AI systems in AdalFlow. An agent combines reasoning capabilities with tool usage, allowing it to break down complex tasks into steps, use available tools, and iteratively work toward solutions.

## Overview

An AdalFlow agent consists of two main components:
- **Agent**: Handles planning and decision-making using a Generator-based planner
- **Runner**: Manages execution, tool calling, and conversation flow

This separation allows for flexible customization of both planning and execution logic.

## Quick Start

Here's a minimal example to get you started:

```python
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool

# Define a simple tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create the agent
agent = Agent(
    name="MathAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

# Create the runner
runner = Runner(agent=agent)

# Execute a query
result = runner.call(
    prompt_kwargs={"input_str": "What is 15 * 7 + 23?"}
)

print(result.answer)
```

## Core Components

### Agent

The Agent class orchestrates AI planning and tool execution using a ReAct (Reasoning and Acting) architecture. By default, the Agent uses a Generator-based planner for decision-making and a ToolManager for managing and executing tools when not explicitly provided.

#### Basic Configuration

```python
from adalflow.components.agent import Agent
from adalflow.components.model_client.openai_client import OpenAIClient

agent = Agent(
    name="MyAgent",                    # Agent identifier
    tools=[],                          # List of available tools
    model_client=OpenAIClient(),       # Language model client
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},   # Model configuration
    max_steps=10                       # Maximum reasoning steps
)
```

#### Key Parameters

- **name**: A descriptive name for your agent
- **tools**: List of FunctionTool objects the agent can use
- **model_client**: The language model client (OpenAI, Anthropic, etc.)
- **model_kwargs**: Configuration for the language model
- **max_steps**: Maximum number of reasoning steps before termination
- **answer_data_type**: Expected type for the final answer (str, int, etc.)

### Runner

The Runner executes Agent instances with support for multi-step reasoning, tool execution, and conversation management.

```python
from adalflow.components.agent import Runner

runner = Runner(
    agent=agent,
    max_steps=5,           # Override agent's max_steps if needed
    memory=None,           # Optional conversation memory
    permissions=None       # Optional permission manager
)
```

### Tools

Tools extend your agent's capabilities. AdalFlow supports several tool types:

#### Function Tools

Convert regular Python functions into agent tools. Tools can return various types including basic Python types, custom objects, or `ToolOutput` for enhanced control:

```python
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import ToolOutput

# Basic return types
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return f"Search results for: {query}"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    # Implementation here
    return f"Email sent to {to}"

# Enhanced control with ToolOutput
def advanced_search(query: str, max_results: int = 5) -> ToolOutput:
    """Advanced search with enhanced agent feedback."""
    results = perform_search(query, max_results)
    return ToolOutput(
        output=results,                           # The actual data
        observation=f"Found {len(results)} results for '{query}'",  # For agent reasoning
        display=f"🔍 Searched: {query}",         # For user display
        metadata={"query": query, "count": len(results)}
    )

tools = [
    FunctionTool(search_web),
    FunctionTool(send_email),
    FunctionTool(advanced_search)
]
```

#### Component Tools

Use AdalFlow components as tools:

```python
from adalflow.core.retriever import Retriever
from adalflow.core.func_tool import FunctionTool

# Assuming you have a retriever implementation
retriever = MyRetriever(top_k=3)
tools = [FunctionTool(retriever.call)]
```

#### Tool Return Types

AdalFlow tools are flexible and can return various types:

```python
from adalflow.core.types import ToolOutput
import asyncio

# Basic Python types
def calculate(expression: str) -> float:
    """Return a number."""
    return eval(expression)

def get_data() -> dict:
    """Return structured data."""
    return {"key": "value", "count": 42}

# Async tools
async def fetch_api_data(url: str) -> dict:
    """Async tool returning data."""
    # Simulate API call
    await asyncio.sleep(0.1)
    return {"status": "success", "data": "api_result"}

# Generator tools for streaming
def stream_results(count: int):
    """Generator tool for streaming output."""
    for i in range(count):
        yield f"Result {i+1}"

# ToolOutput for enhanced control
def enhanced_tool(query: str) -> ToolOutput:
    """Tool with rich output information."""
    result = f"Processed: {query}"
    return ToolOutput(
        output=result,                    # Main output for the agent
        observation="Processing completed successfully",  # Agent reasoning context
        display=f"✅ {query} processed",  # User-friendly display
        metadata={"processing_time": 0.5, "status": "success"}
    )

# All tool types can be used with FunctionTool
tools = [
    FunctionTool(calculate),
    FunctionTool(get_data),
    FunctionTool(fetch_api_data),
    FunctionTool(stream_results),
    FunctionTool(enhanced_tool)
]
```

## Advanced Features

### Streaming Execution

Execute agents with streaming support for real-time updates:

```python
import asyncio

async def stream_example():
    runner = Runner(agent=agent)
    
    async for event in runner.astream(
        prompt_kwargs={"input_str": "Analyze this data..."}
    ):
        print(f"Event: {event}")

# Run the streaming example
asyncio.run(stream_example())
```

### Memory Integration

Add persistent conversation memory:

```python
from adalflow.components.memory import Memory

memory = Memory()
runner = Runner(agent=agent, memory=memory)

# Conversations will now maintain context across calls
result1 = runner.call(prompt_kwargs={"input_str": "Remember my name is John"})
result2 = runner.call(prompt_kwargs={"input_str": "What's my name?"})
```

### Permission Management

Add approval workflows for sensitive operations:

```python
from adalflow.components.agent.permissions import PermissionManager

permissions = PermissionManager()
runner = Runner(agent=agent, permissions=permissions)

# Tools will now require approval before execution
result = runner.call(prompt_kwargs={"input_str": "Delete all files"})
```

## Examples

### RAG Agent

Create a Retrieval-Augmented Generation agent. We will provide a more detailed tutorial for the RAG Agent and also evaluate its performance against benchmarks. 

```python
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.retriever import Retriever

setup_env()

class DocumentRetriever(Retriever):
    def __init__(self, documents: list):
        super().__init__()
        self.documents = documents
    
    def call(self, input: str, top_k: int = 3):
        # Simple similarity search implementation
        # In practice, use vector embeddings
        relevant_docs = [doc for doc in self.documents if input.lower() in doc.lower()]
        return relevant_docs[:top_k]

# Setup
documents = [
    "Python is a programming language.",
    "Machine learning uses algorithms to learn patterns.",
    "AdalFlow is a framework for building AI applications."
]

retriever = DocumentRetriever(documents)

agent = Agent(
    name="RAGAgent", 
    tools=[FunctionTool(retriever.call)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o"},
    max_steps=5
)

runner = Runner(agent=agent)

result = runner.call(
    prompt_kwargs={"input_str": "What is AdalFlow?"}
)
```

### Multi-Tool Calculator Agent

An agent that can perform various mathematical operations:

```python
import math
from adalflow.components.agent import Agent, Runner
from adalflow.core.func_tool import FunctionTool

def basic_calculator(expression: str) -> str:
    """Evaluate basic mathematical expressions."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def advanced_math(operation: str, value: float) -> str:
    """Perform advanced mathematical operations."""
    ops = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log,
        "factorial": math.factorial
    }
    
    if operation in ops:
        try:
            result = ops[operation](value)
            return f"{operation}({value}) = {result}"
        except Exception as e:
            return f"Error: {e}"
    else:
        return f"Unknown operation: {operation}"

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between different units."""
    # Length conversions (to meters)
    length_units = {
        "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34
    }
    
    if from_unit in length_units and to_unit in length_units:
        meters = value * length_units[from_unit]
        result = meters / length_units[to_unit]
        return f"{value} {from_unit} = {result} {to_unit}"
    else:
        return "Unsupported unit conversion"

# Create agent with multiple tools
tools = [
    FunctionTool(basic_calculator),
    FunctionTool(advanced_math),
    FunctionTool(unit_converter)
]

agent = Agent(
    name="MathAgent",
    tools=tools,
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=10
)

runner = Runner(agent=agent)

# Example usage
result = runner.call(
    prompt_kwargs={
        "input_str": "First calculate the square root of 144, then convert 5 feet to meters"
    }
)
```

### Research Agent with Web Search

An agent that can search for information and synthesize results:

```python
from adalflow.utils import setup_env
from adalflow.core.func_tool import FunctionTool
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient

setup_env()

def web_search(query: str) -> str:
    """Search for information on the web."""
    # This is a simplified example - in practice, use proper search APIs
    try:
        # Placeholder for actual web search implementation
        return f"Search results for '{query}': [Relevant information found]"
    except Exception as e:
        return f"Search failed: {e}"

def summarize_text(text: str, max_length: int = 200) -> str:
    """Summarize a piece of text."""
    if len(text) <= max_length:
        return text
    
    # Simple truncation - in practice, use proper summarization
    return text[:max_length] + "..."

def fact_check(claim: str) -> str:
    """Fact-check a claim by searching for supporting evidence."""
    # In practice, implement proper fact-checking logic
    return f"Fact-checking result for '{claim}': [Analysis needed]"

tools = [
    FunctionTool(web_search),
    FunctionTool(summarize_text),
    FunctionTool(fact_check)
]

agent = Agent(
    name="ResearchAgent",
    tools=tools,
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=8
)

runner = Runner(agent=agent)

result = runner.call(
    prompt_kwargs={
        "input_str": "Research the latest developments in quantum computing and provide a summary"
    }
)
```

## Execution Flow

AdalFlow agents follow a structured execution pattern:

1. **Planning**: The agent analyzes the input and creates a plan
2. **Tool Selection**: Based on the plan, selects appropriate tools
3. **Tool Execution**: Executes the selected tools with parameters
4. **Observation**: Processes tool outputs and updates context
5. **Iteration**: Repeats steps 1-4 until the task is complete or max_steps is reached
6. **Final Answer**: Synthesizes all information into a final response

## Configuration Options

### Model Configuration

Configure different language models:

```python
from adalflow.utils import setup_env
setup_env()

# OpenAI
from adalflow.components.model_client.openai_client import OpenAIClient
model_client = OpenAIClient()
model_kwargs = {"model": "gpt-4o", "temperature": 0.7}

# Anthropic
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
model_client = AnthropicAPIClient()
model_kwargs = {"model": "claude-3-sonnet-20240229"}
```

### Output Types

Specify expected output types:

```python
# String output
agent = Agent(
    name="TextAgent",
    answer_data_type=str,
    # ... other config
)

# Structured output with Pydantic
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    recommendations: list[str]

agent = Agent(
    name="AnalysisAgent",
    answer_data_type=AnalysisResult,
    # ... other config
)
```

### Custom System Prompts

Customize the agent's behavior with custom prompts:

```python
custom_prompt = """
You are a helpful assistant specialized in data analysis. 
Always provide step-by-step reasoning and cite your sources.
When using tools, explain why you chose each tool.
"""

agent = Agent(
    name="DataAnalyst",
    system_prompt=custom_prompt,
    # ... other config
)
```

## Error Handling

Agents include robust error handling:

```python
try:
    result = runner.call(
        prompt_kwargs={"input_str": "Complex query here"}
    )
    print(f"Success: {result.answer}")
except Exception as e:
    print(f"Agent execution failed: {e}")
    
    # Access step history for debugging
    if hasattr(result, 'step_history'):
        for i, step in enumerate(result.step_history):
            print(f"Step {i+1}: {step.function} -> {step.observation}")
```

## Best Practices

### Tool Design

1. **Clear Descriptions**: Provide detailed docstrings for tools
2. **Type Hints**: Use proper type annotations. The return type can be `ToolOutput` or other types. 
3. **Error Handling**: Handle exceptions gracefully within tools
4. **Focused Functionality**: Keep tools focused on single responsibilities

```python
def good_tool_example(query: str, limit: int = 10) -> str:
    """
    Search for information with a specific query.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        
    Returns:
        Formatted search results as a string
        
    Raises:
        ValueError: If query is empty
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
        # Tool implementation
        results = perform_search(query, limit)
        return format_results(results)
    except Exception as e:
        return f"Search failed: {str(e)}"
```

### Agent Configuration

1. **Reasonable Step Limits**: Set appropriate max_steps to prevent infinite loops
2. **Model Selection**: Choose models appropriate for your task complexity
3. **Tool Combinations**: Ensure tools work well together
4. **Testing**: Test agents thoroughly with various inputs

### Performance Optimization

1. **Caching**: Enable caching for expensive operations
2. **Async Tools**: Use async tools for I/O operations
3. **Batch Operations**: Group related operations when possible
4. **Monitoring**: Use AdalFlow's tracing capabilities
