<div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
    <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/agent/tutorial_agent_advanced_features.py" target="_blank" style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
        <span style="vertical-align: middle;"> Open Source Code (Agent Advanced)</span>
    </a>
</div>

# Streaming
Streaming allows you to receive real-time updates as your agent executes steps, tools, and generates responses. This enables you to build responsive user interfaces and monitor agent progress in real-time.

## Overview

AdalFlow's streaming architecture provides two types of real-time events:

1. {doc}`RawResponsesStreamEvent <../apis/core/core.types>`: Token-level updates from the language model
2. {doc}`RunItemStreamEvent <../apis/core/core.types>`: High-level agent execution progress (tool calls, step completion, etc.)

Both event types can be consumed simultaneously, giving you fine-grained control over how you handle streaming data.

## Basic Streaming

The simplest way to stream agent execution is using the `Runner.astream()` method which returns a `RunnerStreamingResult` object. You can consume the events by calling the `stream_events()` method on the `RunnerStreamingResult` object which internally holds an asyncio queue.

```python
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

## Raw Response Stream Events

Raw response stream events are raw events from the language model. These events contain the streaming chunks as they're generated:

```python
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

This comprehensive streaming system enables you to build responsive, real-time applications with AdalFlow agents while maintaining full control over the execution flow and user experience.

### Raw Response Event Structure

Raw response stream events contain the streaming data under its `data` field,directly from the model client:

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
                print(f"🔧 Calling tool: {event.item.data.name}")

            elif isinstance(event.item, ToolOutputRunItem):
                print(f"✅ Tool completed: {event.item.data.output}")

            elif isinstance(event.item, FinalOutputItem):
                print(f"🎯 Final answer: {event.item.data.answer}")

asyncio.run(handle_agent_events())
```

## Streaming with Different Model Clients

### OpenAI Streaming

```python
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
    model_kwargs={"model": "claude-3-5-haiku-20241022", "stream": True, "temperature": 0.8},
    max_steps=5
)
```

## API Reference

:::{admonition} API reference
:class: highlight

- {doc}`adalflow.components.agent.agent.Agent <../apis/components/components.agent.agent>`
- {doc}`adalflow.components.agent.runner.Runner <../apis/components/components.agent.runner>`
- {doc}`adalflow.core.types.RunnerStreamingResult <../apis/core/core.types>`
- {doc}`adalflow.core.types.RawResponsesStreamEvent <../apis/core/core.types>`
- {doc}`adalflow.core.types.RunItemStreamEvent <../apis/core/core.types>`
- {doc}`adalflow.core.types.ToolCallRunItem <../apis/core/core.types>`
- {doc}`adalflow.core.types.ToolOutputRunItem <../apis/core/core.types>`
- {doc}`adalflow.core.types.StepRunItem <../apis/core/core.types>`
- {doc}`adalflow.core.types.FinalOutputItem <../apis/core/core.types>`
- {doc}`adalflow.core.types.ToolCallActivityRunItem <../apis/core/core.types>`
- {doc}`adalflow.core.func_tool.FunctionTool <../apis/core/core.func_tool>`
- {doc}`adalflow.components.model_client.openai_client.OpenAIClient <../apis/components/components.model_client.openai_client>`
- {doc}`adalflow.components.model_client.anthropic_client.AnthropicAPIClient <../apis/components/components.model_client.anthropic_client>`
:::
