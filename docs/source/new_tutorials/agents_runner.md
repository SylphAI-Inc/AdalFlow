.. raw:: html

    <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
        <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/agent/tutorial_agent_basic.py" target="_blank" style="display: flex; align-items: center;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
            <span style="vertical-align: middle;"> Open Source Code (Agent Basic)</span>
        </a>
        <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/agent/tutorial_agent_advanced_features.py" target="_blank" style="display: flex; align-items: center;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
            <span style="vertical-align: middle;"> Open Source Code (Agent Advanced)</span>
        </a>

# Agents and Runner

Agents are the core building block for creating autonomous AI systems in AdalFlow. An agent combines reasoning capabilities with tool usage, allowing it to break down complex tasks into steps, use available tools, and iteratively work toward solutions. This approach is motivated by the ReAcT (Reasoning and Acting) framework [[Yao et al., 2022]](https://arxiv.org/abs/2210.03629), which combines reasoning traces and task-specific actions in language models.

## Overview

An AdalFlow agent consists of two main components:
- {doc}`Agent <../apis/components/components.agent.agent>`: Handles planning and decision-making using a Generator-based planner
- {doc}`Runner <../apis/components/components.agent.runner>`: Manages execution, tool calling, and conversation flow

This separation allows for flexible customization of both planning and execution logic.

## Minimal Example

We have a simple calculator tool that can evaluate mathematical expressions.

```python
from adalflow import Agent, Runnerfrom adalflow.components.model_client.openai_client import OpenAIClient


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create agent with tools that require permission
agent = Agent(
    name="CalculatorAgent",
    tools=[
        calculator,
    ],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=6
)

runner = Runner(agent=agent)

result = runner.call(prompt_kwargs={"input_str": "Invoke the calculator tool and calculate 15 * 7 + 23"})
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
- **tools**: List of FunctionTool or callable objects the agent can use (see [Tool Helper](../tutorials/tool_helper) for detailed information)
- **model_client**: The language model client used by the generator (OpenAI, Anthropic, etc.) (see [Generator](../tutorials/generator) and [Model Client](../tutorials/model_client) for detailed information)
- **model_kwargs**: Configuration for the language model used by the generator
- **max_steps**: Maximum number of reasoning steps before termination
- **answer_data_type**: Expected type for the final answer. The data type can be a Pydantic dataclass, Adalflow dataclass (see [Base Data Class](../tutorials/base_data_class)), or a built-in Python type.

### Runner

The Runner executes Agent instances with support for multi-step reasoning, tool execution, and conversation management.

```python
runner = Runner(
    agent=agent,
    max_steps=5,           # Override agent's max_steps if needed
)
```

#### RunnerResult

The `Runner.call()` method returns a `RunnerResult` object that contains comprehensive information about the execution:

**Field Descriptions:**

- **step_history**: A chronological list of `StepOutput` objects representing each reasoning step the agent took, including the action performed, tool calls, and observations received.
- **answer**: The final answer or result produced by the agent after completing all reasoning steps.
- **error**: Contains error information if the agent execution failed due to an exception, timeout, or other issues. `None` if execution was successful.
- **ctx**: Optional context dictionary that can store additional metadata or state information from the execution process.

### Tools

Tools extend your agent's capabilities by allowing them to call functions, access APIs, or interact with external systems. AdalFlow supports multiple tool types including FunctionTool (wrapping Python functions) and MCPFunctionTool (for MCP protocol integration). Tools can return various types including basic Python types, custom objects, or `ToolOutput` for enhanced control.

For comprehensive information about tool types, implementation patterns, error handling, and advanced features, see the {doc}`Tool Use <tool>` documentation.

## Advanced Features

### Streaming Execution

You can execute agents with {doc}`RunnerStreamingResult <../apis/core/core.types>` support for real-time updates. See the [Streaming](streaming) tutorial for detailed information and examples.

### Human in the Loop

AdalFlow provides a **PermissionManager** system that allows you to control and approve tool executions before they run. This is particularly useful for tools that perform sensitive operations like file system access, API calls, or external communications.

For comprehensive coverage of permission management features including CLI handlers, FastAPI integration, custom permission managers, and security best practices, see the dedicated [Human in the Loop](permission_management.md) tutorial.

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
model_client = OpenAIClient()
model_kwargs = {"model": "gpt-4o", "temperature": 0.7}

# Anthropic
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
model_client = AnthropicAPIClient()
model_kwargs = {"model": "claude-3-sonnet-20240229"}
```

### Output Types

Specify expected output types. The output types can be a built-in Python type, a Pydantic dataclass, or an Adalflow dataclass (see [Base Data Class](../tutorials/base_data_class)).

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

Customize the agent's role description as below by creating a custom role description string.

```python
custom_role_desc = """
You are a helpful assistant specialized in data analysis.
Always provide step-by-step reasoning and cite your sources.
When using tools, explain why you chose each tool.
</system>
"""

agent = Agent(
    name="DataAnalyst",
    role_desc=custom_role_desc,
    # ... other config
)
```

In practice to customize the whole template and planner configurations, pass in a new planner with custom configurations. Refer to `Generator` for more details.

## Tracing

AdalFlow provides comprehensive tracing capabilities to monitor and debug agent execution. You can trace agent steps, tool calls, and model interactions. See the [Tracing](tracing) tutorial for detailed information on setting up and using tracing features.

## API Reference

:::{admonition} API reference
:class: highlight

- {doc}`components.agent.agent.Agent <../apis/components/components.agent.agent>`
- {doc}`components.agent.runner.Runner <../apis/components/components.agent.runner>`
- {doc}`core.types.RunnerResult <../apis/core/core.types>`
- {doc}`core.types.RunnerStreamingResult <../apis/core/core.types>`
- {doc}`core.types.StepOutput <../apis/core/core.types>`
- {doc}`core.types.FunctionOutput <../apis/core/core.types>`
- {doc}`core.types.ToolOutput <../apis/core/core.types>`
- {doc}`func_tool.FunctionTool <../apis/core/core.func_tool>`
- {doc}`generator.Generator <../apis/core/core.generator>`
- {doc}`tool_manager.ToolManager <../apis/core/core.tool_manager>`
- {doc}`tracing.runner_span <../apis/tracing/tracing.create>`
- {doc}`tracing.tool_span <../apis/tracing/tracing.create>`
- {doc}`tracing.GeneratorStateLogger <../apis/tracing/tracing.generator_state_logger>`
- {doc}`tracing.GeneratorCallLogger <../apis/tracing/tracing.generator_call_logger>`
:::


