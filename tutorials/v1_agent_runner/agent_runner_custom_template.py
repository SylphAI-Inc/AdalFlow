# """
# Quick start tutorial showing a minimal end-to-end flow with

# 1. A custom `greet` tool.
# 2. An `Agent` that can call the tool and a simple LLM.
# 3. A `Runner` that executes multi-step reasoning and post-processes the
#    model response into a `StepOutput` so that downstream components can
#    easily consume the **observation** field.

# Key concepts covered:
# 1. Creating an Agent with tools and context
# 2. Using the Runner for multi-step execution
# 3. Handling function calls
# 4. Configuring output types
# """

# import logging
# from typing import Any, List


# from adalflow.core.agent import Agent
# from adalflow.core.runner import Runner
# from adalflow.core.types import StepOutput
# from adalflow.core.types import Function
# from adalflow.core.tool_manager import ToolManager
# from adalflow.core.func_tool import FunctionTool
# from adalflow.core.component import DataComponent

# import numpy as np
# import asyncio


# from dotenv import load_dotenv
# import os
# from adalflow.components.model_client import OpenAIClient

# import time

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)


# # ---------------------------------------------------------------------------
# # Tools
# # ---------------------------------------------------------------------------
# # Example tool for demonstration
# def greet(name: str) -> str:
#     """Simple greeting function."""
#     print("=======" * 100)
#     print("Function was called")
#     print("=======" * 100)
#     return f"Hello, {name}!"


# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     time.sleep(1)
#     return a * b


# def add(a: int, b: int) -> int:
#     """Add two numbers."""
#     time.sleep(1)
#     return a + b


# async def divide(a: float, b: float) -> float:
#     """Divide two numbers."""
#     await asyncio.sleep(1)
#     return float(a) / b


# async def search(query: str) -> List[str]:
#     """Search for query and return a list of results."""
#     await asyncio.sleep(1)
#     return ["result1" + query, "result2" + query]


# def numpy_sum(arr: np.ndarray) -> float:
#     """Sum the elements of an array."""
#     return np.sum(arr)


# import json
# import re

# # ---------------------------------------------------------------------------
# # Output Parsing Helper
# # ---------------------------------------------------------------------------


# def parse_step_output_from_string(log_string: str) -> StepOutput:
#     """Extract and parse JSON string from log output into a StepOutput object."""
#     # Extract the JSON part using a more robust pattern
#     match = re.search(r"(\{[\s\S]*\})", log_string)
#     if not match:
#         raise ValueError("No JSON found in the input string")

#     json_str = match.group(1)

#     try:
#         # Clean up the string before JSON parsing
#         # Replace escaped single quotes with double quotes
#         json_str = json_str.replace("\\'", "'")
#         # Remove any trailing commas that might be after the JSON object
#         json_str = re.sub(r",\s*\}", "}", json_str)
#         json_str = re.sub(r",\s*\]", "]", json_str)

#         # Parse the JSON
#         data = json.loads(json_str)

#         # Create and return StepOutput
#         if data.get("function"):
#             function = Function(**data.get("function"))
#         else:
#             function = None
#         return StepOutput(
#             action=data.get("action"),
#             function=function,
#             observation=data.get("observation"),
#             step=data.get("step", 0),
#         )

#     except json.JSONDecodeError as e:
#         # For debugging, print the problematic string
#         print(f"Failed to parse JSON. String was: {json_str}")
#         raise ValueError(f"Failed to parse JSON: {e}")


# # Example usage:
# log_string = """'{\n    "action": "observation",\n    "observation": "The available tool is a function called greet, which is a simple greeting function. It belongs to the class greet and takes a \\'name\\' parameter of type string."\n}',)"""

# # print(parse_step_output_from_string(log_string))

# # ---------------------------------------------------------------------------
# # Output Processor
# # ---------------------------------------------------------------------------


# class StepOutputProcessor(DataComponent):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func

#     def call(self, response: str) -> Any:
#         return self.func(response)

#     def bicall(self, response: str) -> Any:
#         return self.func(response)


# CUSTOM_TEMPLATE = """You are a helpful assistant. Use the tools provided to answer questions.

#     Available tools:
#     {{ tool_definitions }}

#     Respond with a JSON object containing your thoughts and the action to take.
#     The response must follow these rules:
#     - The output must follow the exact format specified below that can then be parsed into a python object from the returned string
#     - action: The action to take is always ("observation" or "finish")
#     - function: to answer or carry out tasks, invoke necessary functions in the above available tools. To invoke a function include the function field in the output response
#     - observation: The final text response. There should always be an observation field
#     - you cannot invoke a function with an action set to "finish", the action must be set as "observation"
#     - Only when you believe the user's query or demand has been exactly satisfied (and only when this is the case), respond with the action field as "finish" and include the
#     final response to the user's query under the "observation" field. Provide reasoning for why you chose finish
#     - For any of the optional fields if they are not be used rather than filling them with a None or placeholder value, leave them out
#     - args and kwargs should be passed in as if exactly passing into the function
#     - ENSURE THAT THE RESPONSE IS ALWAYS IN THE FORMAT SPECIFIED BELOW

#     These are outputs from previous steps (use them to evaluate whether you have already satisfied the user query):
#         Maximum Number of Steps:
#         {{ max_steps }}

#         Previous Output:
#         {{ previous_output }}

#         Function Results:
#         {{ function_results }}

#     USER QUERY:
#     {{ input_str }}

#     The output response format is as follows which must be strictly kept
#     {% raw %}{{
#         "action": str,
#         "function": Optional[Function],
#         "observation": str,
#     }}{% endraw %}

#     Function has the output structure of
#     {% raw %}{{
#         "name": str, # parameter used to identify which function from tool_definitions
#         "args": Optional[List[object]], # parameters will be passed to the actual function
#         "kwargs": Optional[Dict[str, object]], # parameters will be passed to the actual function
#         "thought": Optional[str]
#     }}{% endraw %}

#     Example of the output response:
#     {% raw %}{{
#         "action": "result",
#         "observation": "Hi!"
#     }}{% endraw %}
# """


# # ---------------------------------------------------------------------------
# # Pydantic Models for Output Type
# # ---------------------------------------------------------------------------


# # ---------------------------------------------------------------------------
# # Demo pipeline
# # ---------------------------------------------------------------------------


# def synchronous_calls_to_multiple_function_tools_use_custom_template():
#     api_key = load_dotenv()
#     # 1. Create tools
#     tools = [
#         FunctionTool(
#             fn=greet,
#         ),
#         FunctionTool(
#             fn=multiply,
#         ),
#         FunctionTool(
#             fn=add,
#         ),
#         FunctionTool(
#             fn=divide,
#         ),
#         FunctionTool(
#             fn=search,
#         ),
#         FunctionTool(
#             fn=numpy_sum,
#         ),
#     ]

#     # tools[0].bicall("John")
#     print(tools[2].bicall(a=5.0, b=9999000.0))

#     api_key = os.getenv("OPENAI_API_KEY")

#     # 2. Create model client
#     model_client = OpenAIClient(api_key=api_key)

#     basic_template = CUSTOM_TEMPLATE

#     output_processors = StepOutputProcessor(parse_step_output_from_string)

#     print(ToolManager(tools).yaml_definitions)

#     # 3. Create Agent with output processors
#     agent = Agent(
#         name="example_agent",
#         tools=tools,
#         model_client=model_client,
#         model_kwargs={"model": "gpt-4o-mini"},
#         # model_kwargs={"model": "gpt-3.5-turbo"},
#         template=basic_template,
#         prompt_kwargs={
#             "tool_definitions": ToolManager(tools).yaml_definitions,
#         },
#         output_processors=output_processors,
#     )

#     # 4. Create Runner with output type
#     runner = Runner(
#         agent=agent,
#         max_steps=6,
#         # output_type=AgentResponse  # Use our Pydantic model as output type
#     )

#     # -------------------------------------------------------------------
#     # 5. Example usage
#     prompt_kwargs = {
#         "input_str": "Use the given tools in the most optimal way to evaluate 5 / (9999 * 1000) + 1",
#         "max_steps": 6,
#         # "input_str": "Who are you?"
#     }

#     # Synchronous execution
#     try:
#         result = runner.call(
#             prompt_kwargs=prompt_kwargs,
#             model_kwargs={"temperature": 0.3},
#             use_cache=False,
#         )

#         print("=====" * 100)

#         for i in result[0]:
#             print("Result of Iteration: ", i)
#             print("\n")

#         print("Final Response: ", result[1])

#     except Exception as e:
#         print(f"Error during execution: {str(e)}")


# def synchronous_calls_to_single_function_tool_use_custom_template():
#     api_key = load_dotenv()
#     # 1. Create tools
#     tools = [
#         FunctionTool(
#             fn=greet,
#         )
#     ]

#     # tools[0].bicall("John")

#     api_key = os.getenv("OPENAI_API_KEY")

#     # 2. Create model client
#     model_client = OpenAIClient(api_key=api_key)

#     # custom_template = CUSTOM_TEMPLATE

#     output_processors = StepOutputProcessor(parse_step_output_from_string)

#     print(ToolManager(tools).yaml_definitions)

#     # 3. Create Agent with output processors
#     agent = Agent(
#         name="example_agent",
#         tools=tools,
#         model_client=model_client,
#         model_kwargs={"model": "gpt-4o-mini"},
#         # model_kwargs={"model": "gpt-3.5-turbo"},
#         # template=basic_template,
#         prompt_kwargs={
#             "tool_definitions": ToolManager(tools).yaml_definitions,
#         },
#         output_processors=output_processors,
#     )

#     # 4. Create Runner with output type
#     runner = Runner(
#         agent=agent,
#         max_steps=3,
#         # output_type=AgentResponse  # Use our Pydantic model as output type
#     )

#     # -------------------------------------------------------------------
#     # 5. Example usage
#     prompt_kwargs = {
#         "input_str": "Say hello exactly once to John for me",
#         "max_steps": 3,
#         # "input_str": "Who are you?"
#     }

#     # Synchronous execution
#     try:
#         result = runner.call(
#             prompt_kwargs=prompt_kwargs,
#             model_kwargs={"temperature": 0.3},
#             use_cache=False,
#         )

#         print("=====" * 100)

#         for i in result[0]:
#             print("Result of Iteration: ", i)
#             print("\n")

#         print("Final Response: ", result[1])

#     except Exception as e:
#         print(f"Error during execution: {str(e)}")


# def use_react_template():
#     api_key = load_dotenv()
#     # 1. Create tools
#     tools = [
#         FunctionTool(
#             fn=greet,
#         )
#     ]

#     api_key = os.getenv("OPENAI_API_KEY")
#     print(api_key)

#     # 2. Create model client
#     model_client = OpenAIClient(api_key=api_key)

#     basic_template = react_agent_task_desc

#     output_processors = StepOutputProcessor(parse_step_output_from_string)

#     print(ToolManager(tools).yaml_definitions)

#     # 3. Create Agent with output processors
#     agent = Agent(
#         name="example_agent",
#         tools=tools,
#         model_client=model_client,
#         model_kwargs={"model": "gpt-4o-mini"},
#         # model_kwargs={"model": "gpt-3.5-turbo"},
#         template=basic_template,
#         prompt_kwargs={
#             "tool_definitions": ToolManager(tools).yaml_definitions,
#         },
#         output_processors=output_processors,
#     )

#     # 4. Create Runner with output type
#     runner = Runner(
#         agent=agent,
#         max_steps=3,
#         # output_type=AgentResponse  # Use our Pydantic model as output type
#     )

#     # -------------------------------------------------------------------
#     # 5. Example usage
#     prompt_kwargs = {
#         "input_str": "Say hello to John for me"
#         # "input_str": "Who are you?"
#     }

#     # Synchronous execution
#     try:
#         result = runner.call(
#             prompt_kwargs=prompt_kwargs,
#             model_kwargs={"temperature": 0.3},
#             use_cache=False,
#         )

#         print("=====" * 100)

#         for i in result[0]:
#             print("Result of Iteration: ", i)
#             print("\n")

#         print("Final Response: ", result[1])

#     except Exception as e:
#         print(f"Error during execution: {str(e)}")


# if __name__ == "__main__":
#     # synchronous_calls_to_single_function_tool_use_custom_template()
#     synchronous_calls_to_multiple_function_tools_use_custom_template()
