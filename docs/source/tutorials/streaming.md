# Streaming
Streaming allows you to receive real-time updates as your agent executes steps, tools, and generates responses. This enables you to build responsive user interfaces and monitor agent progress in real-time.

## Overview

AdalFlow's streaming architecture provides two types of real-time events:

1. **Raw Response Events**: Token-level updates from the language model
2. **Run Item Events**: High-level agent execution progress (tool calls, step completion, etc.)

Both event types can be consumed simultaneously, giving you fine-grained control over how you handle streaming data.

## Basic Streaming

The simplest way to stream agent execution is using the `Runner.astream()` method:

```python
import asyncio
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool

# Setup environment
setup_env()

# Define a simple tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create agent and runner
agent = Agent(
    name="StreamingAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)

async def stream_example():
    # Start streaming execution
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "What is 15 * 7 + 23?"},
        model_kwargs={"stream": True}
    )
    
    # Process streaming events
    async for event in streaming_result.stream_events():
        print(f"Event: {event}")

# Run the example
asyncio.run(stream_example())
```

## Raw Response Events

Raw response events provide token-level updates directly from the language model. These events contain the streaming chunks as they're generated:

```python
import asyncio
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import RawResponsesStreamEvent

# Setup environment
setup_env()

# Define a simple tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create agent and runner
agent = Agent(
    name="StreamingAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)

async def handle_raw_responses():
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 25 * 4 and explain the result"},
        model_kwargs={"stream": True}
    )
    
    async for event in streaming_result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            # Process raw model output
            if hasattr(event.data, 'choices') and event.data.choices:
                delta = event.data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end='', flush=True)

asyncio.run(handle_raw_responses())
```

## Streaming Event Reference

### Event Types

| Event Type | Description | When Emitted |
|------------|-------------|--------------|
| `RawResponsesStreamEvent` | Raw model output chunks | During model generation |
| `ToolCallRunItem` | Tool about to be executed | Before tool execution |
| `ToolOutputRunItem` | Tool execution result | After tool execution |
| `StepRunItem` | Agent step completed | After each reasoning step |
| `FinalOutputItem` | Final execution result | At completion |

### Event Names

| Event Name | Description |
|------------|-------------|
| `agent.tool_call_start` | Tool execution starting |
| `agent.tool_call_activity` | Tool intermediate activity and progress updates |
| `agent.tool_call_complete` | Tool execution completed |
| `agent.step_complete` | Reasoning step completed |
| `agent.final_output` | Final processed output available |
| `agent.execution_complete` | Entire execution finished |
| `agent.tool_permission_request` | Tool permission request before execution |
| `message_output_created` | New message output created |
| `handoff_requested` | Agent handoff requested |
| `handoff_occured` | Agent handoff occurred |
| `tool_called` | Tool function called |
| `tool_output` | Tool execution output |
| `reasoning_item_created` | Reasoning item created |
| `mcp_approval_requested` | MCP approval requested |
| `mcp_list_tools` | MCP tools listed |

This comprehensive streaming system enables you to build responsive, real-time applications with AdalFlow agents while maintaining full control over the execution flow and user experience.

### Raw Response Event Structure

Raw response events contain the streaming data under its `data` field,directly from the model client:

```python
# Example raw response event data
{
    "type": "raw_response_event",
    "data": {
        "choices": [{
            "delta": {
                "content": "Quantum computing is a revolutionary technology..."
            },
            "index": 0
        }]
    }
}
```

## Run Item Events

Run item events provide high-level updates about agent execution progress. These events tell you when tools are called, when steps complete, and when the final answer is ready:

```python
import asyncio
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import (
    RunItemStreamEvent, 
    ToolCallRunItem, 
    ToolOutputRunItem, 
    StepRunItem, 
    FinalOutputItem
)

# Setup environment
setup_env()

# Define a simple tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create agent and runner
agent = Agent(
    name="StreamingAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)

async def handle_agent_events():
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 15 * 7 + 23 and explain the steps"},
        model_kwargs={"stream": True}
    )
    
    async for event in streaming_result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"🔧 Calling tool: {event.item.function.name}")
                print(f"   Arguments: {event.item.function.kwargs}")
            
            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.function.name}")
                print(f"   Result: {event.item.output}")
            
            elif isinstance(event.item, StepRunItem):
                print(f"📋 Step {event.item.step} completed")
                print(f"   Observation: {event.item.observation}")
            
            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")

asyncio.run(handle_agent_events())
```

## Advanced Streaming Examples

### Multi-Tool Agent Streaming

Here's a more complex example with multiple tools:

```python
import asyncio
import math
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import RunItemStreamEvent, ToolCallRunItem, ToolOutputRunItem, StepRunItem, FinalOutputItem

# Setup environment
setup_env()

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def advanced_calculator(operation: str, value: float) -> str:
    """Perform advanced mathematical operations."""
    ops = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log
    }
    
    if operation in ops:
        try:
            result = ops[operation](value)
            return f"{operation}({value}) = {result}"
        except Exception as e:
            return f"Error: {e}"
    else:
        return f"Unknown operation: {operation}"

def web_search(query: str) -> str:
    """Search for information (simulated)."""
    return f"Search results for '{query}': [Relevant information found]"

# Create agent with multiple tools
tools = [
    FunctionTool(calculator),
    FunctionTool(advanced_calculator),
    FunctionTool(web_search)
]

agent = Agent(
    name="MultiToolAgent",
    tools=tools,
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=8
)

runner = Runner(agent=agent)

async def multi_tool_streaming():
    streaming_result = runner.astream(
        prompt_kwargs={
            "input_str": "Calculate the square root of 144, then search for information about that number"
        },
        model_kwargs={"stream": True}
    )
    
    step_count = 0
    async for event in streaming_result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallRunItem):
                print(f"\n🔧 Step {step_count + 1}: Calling {event.item.function.name}")
                print(f"   Arguments: {event.item.function.kwargs}")
            
            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Result: {event.item.output}")
            
            elif isinstance(event.item, StepRunItem):
                step_count += 1
                print(f"📋 Step {step_count} completed")
            
            elif isinstance(event.item, FinalOutputItem):
                print(f"\n🎯 Final Answer:")
                print(event.item.data.answer)

asyncio.run(multi_tool_streaming())
```

## Streaming with Permissions

AdalFlow supports permission management during streaming, allowing you to approve or deny tool calls in real-time. The tutorial should create a calculation_result.txt. 

```python
import asyncio
import os
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import RunItemStreamEvent, FinalOutputItem, ToolOutput
from adalflow.apps.cli_permission_handler import CLIPermissionHandler
from adalflow.core.types import FunctionRequest

# Setup environment
setup_env()

# Create directory for logs if it doesn't exist
os.makedirs("logs", exist_ok=True)

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

def sensitive_system_command(command: str) -> ToolOutput:
    """Execute a system command - requires permission."""
    print(f"[Tool Execution] Executing system command: {command}")
    return ToolOutput(
        output=f"Simulated execution of: {command}",
        observation=f"System command '{command}' executed (simulated)",
        display=f"🔧 Executed: {command}",
    )

# Create permission handler with auto-approval for demo
permission_handler = CLIPermissionHandler(approval_mode="default")

# Create agent with tools that require permission
agent = Agent(
    name="PermissionAgent",
    tools=[
        FunctionTool(calculator),  # Safe tool - no permission needed
        FunctionTool(file_writer, require_approval=True),  # Requires permission
        FunctionTool(sensitive_system_command, require_approval=True),  # Requires permission
    ],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=6
)

# Create runner with permission manager
runner = Runner(agent=agent, permission_manager=permission_handler)

async def stream_with_permissions():
    """Demonstrate streaming with permission management."""
    print("🚀 Starting streaming execution with permission management")
    print("Note: Some tools will require approval during execution\n")
    
    streaming_result = runner.astream(
        prompt_kwargs={
            "input_str": "Calculate 25 * 4, then write the result to 'calculation_result.txt', and finally run a system check command 'ls -la'"
        },
        model_kwargs={"stream": True}
    )
    
    # Stream events to JSON file and handle permission requests
    async for event in streaming_result.stream_to_json("logs/permission_execution_log.json"):
        if isinstance(event, RunItemStreamEvent):
            if event.name == "agent.tool_permission_request":
                # event item data should be FunctionRequest
                assert isinstance(event.item.data, FunctionRequest)
                print(f"\n⏸️  PERMISSION REQUEST: {event.item.data.tool_name}")
                print("   Waiting for user approval...")
            elif event.name == "agent.tool_call_start":
                print(f"🔧 Executing: {event.item.data.name}")
            elif event.name == "agent.tool_call_complete":
                print(f"✅ Completed: Tool execution finished")
            elif isinstance(event.item, FinalOutputItem):
                print(f"\n🎯 Final result: {event.item.data.answer}")
                print("Execution completed. Check logs/permission_execution_log.json for full event history.")

async def demo_permission_system():
    """Run the permission system demonstration."""
    print("=" * 80)
    print("🔐 ADALFLOW PERMISSION MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    print("This demo shows how to:")
    print("• Set up tools that require approval")
    print("• Create a permission manager")
    print("• Handle permission requests during streaming")
    print("• Stream events with permission controls")
    print("=" * 80)
    
    await stream_with_permissions()

# Run the demonstration
asyncio.run(demo_permission_system())
```

## Streaming to File

You can save streaming events to a file for later analysis:

```python
import asyncio
import os 
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import RunItemStreamEvent, FinalOutputItem

# Create directory for logs if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup environment
setup_env()

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

agent = Agent(
    name="StreamingAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)

async def stream_to_file():
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Calculate 50 * 3 + 25 and explain the calculation"},
        model_kwargs={"stream": True}
    )
    
    # Stream events to JSON file
    async for event in streaming_result.stream_to_json("logs/execution_log.json"):
        # Events are automatically saved to file
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, FinalOutputItem):
                print("Execution completed. Check logs/execution_log.json for full event history.")

asyncio.run(stream_to_file())
```

## Streaming with Different Model Clients

### OpenAI Streaming

```python
from adalflow.utils import setup_env
from adalflow.components.agent import Agent
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool

setup_env()

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

openai_agent = Agent(
    name="OpenAIAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "stream": True},
    max_steps=5
)
```

### Anthropic Streaming

```python
from adalflow.utils import setup_env
from adalflow.components.agent import Agent
from adalflow.components.model_client.anthropic_client import AnthropicAPIClient
from adalflow.core.func_tool import FunctionTool

setup_env()

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

anthropic_agent = Agent(
    name="AnthropicAgent", 
    tools=[FunctionTool(calculator)],
    model_client=AnthropicAPIClient(),
    model_kwargs={"model": "claude-3-sonnet-20240229", "stream": True},
    max_steps=5
)
```

### Groq Streaming

```python
from adalflow.utils import setup_env
from adalflow.components.agent import Agent
from adalflow.components.model_client.groq_client import GroqAPIClient
from adalflow.core.func_tool import FunctionTool

setup_env()

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

groq_agent = Agent(
    name="GroqAgent",
    tools=[FunctionTool(calculator)],
    model_client=GroqAPIClient(),
    model_kwargs={"model": "llama3-70b-8192", "stream": True},
    max_steps=5
)
```

## Error Handling in Streaming

Handle errors gracefully during streaming execution:

```python
import asyncio
from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import RunItemStreamEvent, ToolOutputRunItem, FinalOutputItem

# Setup environment
setup_env()

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

agent = Agent(
    name="StreamingAgent",
    tools=[FunctionTool(calculator)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)

async def robust_streaming():
    try:
        streaming_result = runner.astream(
            prompt_kwargs={"input_str": "Calculate 100 / 0 and then 50 * 2"},
            model_kwargs={"stream": True}
        )
        
        async for event in streaming_result.stream_events():
            try:
                if isinstance(event, RunItemStreamEvent):
                    if isinstance(event.item, ToolOutputRunItem):
                        if "Error:" in str(event.item.output):
                            print(f"⚠️  Tool error detected: {event.item.output}")
                        else:
                            print(f"✅ Tool success: {event.item.output}")
                    
                    elif isinstance(event.item, FinalOutputItem):
                        print(f"🎯 Execution completed: {event.item.data.answer}")
            
            except Exception as e:
                print(f"Error processing event: {e}")
                continue
    
    except Exception as e:
        print(f"Streaming failed: {e}")

asyncio.run(robust_streaming())
```

### Consuming from Stream 
Consume the events in the stream as they come and only process the events you need. 

```python
async def filtered_streaming():
    streaming_result = runner.astream(
        prompt_kwargs={"input_str": "Process this data"},
        model_kwargs={"stream": True}
    )
    
    async for event in streaming_result.stream_events():
        # Only process tool completion and final output events
        if isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, (ToolOutputRunItem, FinalOutputItem)):
                print(f"Important event: {event}")

asyncio.run(filtered_streaming())
```
