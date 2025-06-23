"""
Quick start tutorial showing a minimal end-to-end flow with

1. A custom `greet` tool.
2. An `Agent` that can call the tool and a simple LLM.
3. A `Runner` that executes multi-step reasoning and post-processes the
   model response into a `StepOutput` so that downstream components can
   easily consume the **observation** field.

Key concepts covered:
1. Creating an Agent with tools and context
2. Using the Runner for multi-step execution
3. Handling function calls
4. Configuring output types
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

from adalflow.core.embedder import Embedder
from adalflow.core.types import ModelClientType

from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.types import StepOutput
from adalflow.core.tool_manager import ToolManager
from adalflow.core.func_tool import FunctionTool
from adalflow.core.component import DataComponent


from dotenv import load_dotenv
import os
from adalflow.components.model_client import OpenAIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# Example tool for demonstration
def greet(name: str) -> str:
    """Simple greeting function."""
    return f"Hello, {name}!"

# ---------------------------------------------------------------------------
# Simple output processor that turns the raw LLM response into a StepOutput.
# ---------------------------------------------------------------------------
def step_output_processor(raw_output) -> StepOutput:
    """Process the raw output into a structured StepOutput with AgentResponse.
    
    Args:
        raw_output: The raw output from the generator, which should be an AgentResponse
        step: The current step number in the execution sequence
        **_: Additional keyword arguments (ignored)
        
    Returns:
        StepOutput: A structured output containing the agent's response
    """
    return StepOutput(
        action=raw_output.action,
        function=raw_output.function,
        observation=raw_output.observation,
        step=raw_output.step,
        id=raw_output.id,
    )

class StepOutputProcessor(DataComponent):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def call(self, *args, **kwargs):
        print("This is the args", args)
        return self.func(*args, **kwargs)

    def bicall(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    


# ---------------------------------------------------------------------------
# Pydantic Models for Output Type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Demo pipeline
# ---------------------------------------------------------------------------

def main():
    api_key = load_dotenv()
    # 1. Create tools
    tools = [
        FunctionTool(
            fn=greet,
        )
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)

    # 2. Create model client
    model_client = OpenAIClient(api_key=api_key) 

    basic_template = basic_template = """You are a helpful assistant. Use the tools provided to answer questions.
    
    Available tools:
    {{ tool_definitions }}

    Respond with a JSON object containing your thoughts and the action to take.
    The response must follow these rules: 
    - The output must follow the exact format specified below that can then be parsed into a python object from the returned string
    - action: The action to take is always ("observation" or "finish"). It should always be set to "observation" by default 
    - function: If you would like to invoke a function, follow the output structure guideline of Function and fill out its fields 
    - observation: The final text response. There should always be an observation field 
    - Once you believe you have finished your task, respond with the action field as "finish" and include the 
    final response in "observation" field.
    - If you don't need to invoke a function, don't include the field function 

    The output response format is 
    {% raw %}{{ 
        "action": str, 
        "function": Optional[Function], 
        "observation": str, 
    }}{% endraw %}

    Function has the output structure of 
    {% raw %}{{
        "name": str, # parameter used to identify which function from tool_definitions 
        "args": Optional[List[object]], # parameters will be passed to the actual function 
        "kwargs": Optional[Dict[str, object]], # parameters will be passed to the actual function 
        "thought": Optional[str]
    }}{% endraw %}
        
    Example 2 (Text response):
    {% raw %}{{
        "action": "result", 
        "observation": "Function greet returned: Hello, John!"
    }}{% endraw %}
    """
    output_processors = StepOutputProcessor(step_output_processor)

    print(ToolManager(tools).yaml_definitions)

    # 3. Create Agent with output processors
    agent = Agent(
        name="example_agent",
        tools=tools,
        model_client=model_client,
        model_kwargs={"model": "gpt-3.5-turbo"},
        template=basic_template,
        prompt_kwargs={
            "tool_definitions": ToolManager(tools).yaml_definitions,
        },
        output_processors=output_processors,
    )

    # 4. Create Runner with output type
    runner = Runner(
        agent=agent,
        max_steps=3,
        # output_type=AgentResponse  # Use our Pydantic model as output type
    )

    # -------------------------------------------------------------------
    # 5. Example usage
    prompt_kwargs = {
        "question": "Say hello to John"
    }

    # Synchronous execution
    try:
        result = runner.call(
            prompt_kwargs=prompt_kwargs,
            model_kwargs={"temperature": 0.7},
            use_cache=False
        )
        print(f"Final result: {result.observation if hasattr(result, 'observation') else result}")
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()
