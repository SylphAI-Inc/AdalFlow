# Agents and Runner

Agents are the core building block for creating autonomous AI systems in AdalFlow. An agent combines reasoning capabilities with tool usage, allowing it to break down complex tasks into steps, use available tools, and iteratively work toward solutions. This approach is motivated by the ReAcT (Reasoning and Acting) framework [[Yao et al., 2022]](https://arxiv.org/abs/2210.03629), which combines reasoning traces and task-specific actions in language models.

## Overview

An AdalFlow agent consists of two main components:
- **Agent**: Handles planning and decision-making using a Generator-based planner
- **Runner**: Manages execution, tool calling, and conversation flow

This separation allows for flexible customization of both planning and execution logic.

## Quick Start

Here's a minimal example to get you started:

```python
from adalflow.utils import setup_env
from adalflow.core.types import ToolOutput
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.apps.cli_permission_handler import CLIPermissionHandler

setup_env()

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def file_writer(filename: str, content: str) -> ToolOutput:
    """Write content to a file - requires permission."""
    print(f"[Tool Execution] Writing to file: {filename}")
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return ToolOutput(
            output=f"Successfully wrote {len(content)} characters to {filename}",
            observation=f"File {filename} written successfully",
            display=f"✍️ Wrote: {filename}",
        )
    except Exception as e:
        return ToolOutput(
            output=f"Error writing to file: {e}",
            observation=f"Failed to write to {filename}",
            display=f"❌ Failed: {filename}",
        )

# Create agent with tools that require permission
agent = Agent(
    name="PermissionAgent",
    tools=[
        FunctionTool(calculator),  # Safe tool - no permission needed
        FunctionTool(file_writer, require_approval=True),  # Requires permission
    ],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=6
)

permission_handler = CLIPermissionHandler(approval_mode="default")
runner = Runner(agent=agent, permission_manager=permission_handler)

# Tools will now require approval before execution
result = runner.call(prompt_kwargs={"input_str": "Invoke the file_writer tool and create a temporary file"})
```

The result of the above code is as follows:
```
RunnerResult(
    step_history=[StepOutput(step=0, action=Function(...), observation='The result is: 128', ctx=None)],
    answer='The result of 15 * 7 + 23 is 128.',
    error=None,
    ctx=None
)
```


## Core Components

### Agent
The Agent uses a Generator-based planner for decision-making and a ToolManager which we can configure.

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
- **tools**: List of FunctionTool or callable objects the agent can use (see [Tool Helper](tool_helper.rst) for detailed information)
- **model_client**: The language model client used by the generator (OpenAI, Anthropic, etc.) (see [Generator](generator.rst) and [Model Client](model_client.rst) for detailed information)
- **model_kwargs**: Configuration for the language model used by the generator
- **max_steps**: Maximum number of reasoning steps before termination
- **answer_data_type**: Expected type for the final answer. The data type can be a Pydantic dataclass, Adalflow dataclass (see [Base Data Class](base_data_class.rst)), or a built-in Python type.

### Runner

The Runner executes Agent instances with support for multi-step reasoning, tool execution, and conversation management.

```python
from adalflow.components.agent import Agent, Runner

runner = Runner(
    agent=agent,
    max_steps=5,           # Override agent's max_steps if needed
)
```

#### RunnerResult

The `Runner.call()` method returns a `RunnerResult` object that contains comprehensive information about the execution:

```python
@dataclass
class RunnerResult:
    step_history: List[StepOutput] = field(
        metadata={"desc": "The step history of the execution"},
        default_factory=list,
    )
    answer: Optional[str] = field(
        metadata={"desc": "The answer to the user's query"}, default=None
    )
    error: Optional[str] = field(
        metadata={"desc": "The error message if the code execution failed"},
        default=None,
    )
    ctx: Optional[Dict] = field(
        metadata={"desc": "The context of the execution"},
        default=None,
    )
```

**Field Descriptions:**

- **step_history**: A chronological list of `StepOutput` objects representing each reasoning step the agent took, including the action performed, tool calls, and observations received.
- **answer**: The final answer or result produced by the agent after completing all reasoning steps.
- **error**: Contains error information if the agent execution failed due to an exception, timeout, or other issues. `None` if execution was successful.
- **ctx**: Optional context dictionary that can store additional metadata or state information from the execution process.

### Tools

Tools extend your agent's capabilities. AdalFlow supports several tool types: 

#### Function Tools

Convert regular Python functions into agent tools. Tools can return various types including basic Python types, custom objects, or `ToolOutput` for enhanced control:

```python
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import ToolOutput
from adalflow.components.agent import Agent
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.utils import setup_env

setup_env()


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

# Create the agent
agent = Agent(
    name="MathAgent",
    tools=tools,
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)
```

You can also use synchronous and asynchronous callables (such as a custom function, class method) as tools that are not necessarily FunctionTools. Further information is provided in the [Tool Helper](tool_helper.rst) documentation. For instance, you could have created tools as below without using FunctionTool

```python
tools = [
    search_web,
    send_email,
    advanced_search
]

# Create the agent
agent = Agent(
    name="MathAgent",
    tools=tools,
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)
```

## Advanced Features

### Streaming Execution

You can execute agents with streaming support for real-time updates. See the [Streaming](streaming.md) tutorial for detailed information and examples.

### Permission Management

AdalFlow provides a permission management system that allows you to control and approve tool executions before they run. This is particularly useful for tools that perform sensitive operations like file system access, API calls, or external communications.

For comprehensive coverage of permission management features including CLI handlers, FastAPI integration, custom permission managers, and security best practices, see the dedicated [Permission Management](permission_management.md) tutorial.

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
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool

setup_env()

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

Specify expected output types. The output types can be a built-in Python type, a Pydantic dataclass, or an Adalflow dataclass (see [Base Data Class](base_data_class.rst)).

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

### Custom System Templates

Customize the agent's behavior with custom prompt templates which are further detailed in the [Prompt](prompt.rst) documentation.

```python
custom_template = """
<system>
{{ system_prompt }}
You are a helpful assistant specialized in data analysis.
Always provide step-by-step reasoning and cite your sources.
When using tools, explain why you chose each tool.
</system>
"""

agent = Agent(
    name="DataAnalyst",
    template=custom_template,
    # ... other config
)
```
