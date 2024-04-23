"""
Agent is not a model or LLM model.
Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.

The future: the agent can write your prompts too. Check out dspy: https://github.com/stanfordnlp/dspy
"""

"""
The initial ReAct paper does not support different types of tools. REact agent can be useful for
- Multi-hop reasoning [Q&A]
- Plan the usage of given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools. 
- Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
- The planner itself can answer the question directly or use the tools to answer the question. We might need to highlight your specifications.
"""

from jinja2 import Template
from lightrag.tool import FunctionTool, AsyncCallable, ToolMetadata
from lightrag.string_parser import BaseTextParser, JsonParser
from typing import List, Union, Callable, Optional, Any
from dataclasses import dataclass
import re
from lightrag.light_rag import Generator


DEFAULT_REACT_AGENT_PROMPT = r"""
<START_OF_SYSTEAM_PROMPT>
{# role/task description #}
You will solve a question answering task.
{# REACT instructions #}
To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:
{# tools #}
{% for tool in tools %}
{{ loop.index }}. ToolName: {{ tool.metadata.name }}
    Tool Description: {{ tool.metadata.description }}
    Tool Args: {{ tool.metadata.fn_schema_str }}
{% endfor %}
{# output is always more robust to use json than string #}
---
Your output should be in json format with two keys:
{
"thought_<step>": "<why you are taking this action>",
"action_<step>": "<ToolName>(arg1, arg2, ...)",
}
---
{# Specifications #}
Remember:
- <step> starts at 1 and increments by 1 for each step.
- Action must call one of the above tools with Took Name. It can not be empty. 
- Use Finish() to finish the task.
- arg1, arg2, ... are the arguments to the tool.
- Only use positional arguments, no keyword arguments. e.g. Finish(...), not Finish(answer=...).
- Finish(answer) to finish the task. Answer can be either the final answer or any message even if its an error message.
{#Examples can be here#}
<END_OF_SYSTEAM_PROMPT>
-----------------
User: {{user_query}}
{# History #}
Your previous Thought, Action, and Observation steps if exists:
{% for history in histories %}
Thought {{history.step}}: {{history.thought}}
Action {{history.step}}: {{history.action}}
Observation {{history.step}}: {{history.observation}}
{% endfor %}
You:
"""


@dataclass
class StepOutput:
    step: int
    thought: str
    action: str
    fun_name: Optional[str] = None  # parsed from action
    fun_args: Optional[List[Any]] = None  # parsed from action
    observation: Optional[str] = (
        None  # when step is created, observation is not available, the funtion result
    )

    def __str__(self):
        return f"Thought {self.step}: {self.thought}\nAction {self.step}: {self.action}\nObservation {self.step}: {self.observation}"


class ReActAgent:
    def __init__(
        self,
        generator: Generator = None,
        generator_prompt: str = DEFAULT_REACT_AGENT_PROMPT,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
    ):
        self.prompt = generator_prompt
        # convert all functions to FunctionTool, and track how to call each function, either call or acall
        self.tools = [
            (
                tool
                if isinstance(tool, FunctionTool)
                else FunctionTool.from_defaults(fn=tool)
            )
            for tool in tools
        ]

        finish_tool_metadata = ToolMetadata(
            name="Finish",
            description="Finish(answer)\nFinish the task",
            parameters={"type": "object", "properties": {"answer": {"type": "str"}}},
        )
        finish_tool = FunctionTool(metadata=finish_tool_metadata, fn=None)
        self.tools.append(finish_tool)

        self.tools_map = {tool.metadata.name: tool for tool in self.tools}

        self.generator = generator
        self.max_steps = max_steps
        self.history: List[StepOutput] = []
        self.text_output_parser = JsonParser()

    def reset(self):
        self.history = []

    def _parse_text_response(self, response: str, step: int) -> Optional[StepOutput]:
        """
        Parse the json output"""
        try:
            json_obj_response = self.text_output_parser(response)
            thought_key = f"thought_{step}"
            action_key = f"action_{step}"
            thought = json_obj_response.get(thought_key, "")
            action = json_obj_response.get(action_key, "")
            return StepOutput(step=step, thought=thought, action=action)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """
        Parse the action string to a function call and execute it.

        Args:
            tools_map (Dict[str, Callable]): Dictionary mapping tool names to their callable functions.

        Returns:
            Optional[Any]: The result of the function call, or None if something goes wrong.
        """
        # Use regex to parse the function name and arguments from the action string
        pattern = r"(\w+)\((.*?)\)"
        action = action_step.action
        match = re.match(pattern, action)
        if not match:
            print(f"No match found for action: {action}")
            return None

        func_name, arg_str = match.groups()
        func_name = func_name.strip()
        print(f"func_name: {func_name}, arg_str: {arg_str}")
        if func_name not in self.tools_map:
            print(f"Function {func_name} not found in tools map.")
            return None
        # update the action_step with the parsed function name and arguments
        action_step.fun_name = func_name

        # Prepare the arguments for the function
        try:
            # This assumes that all arguments are integers. You might need to handle other types.
            # TODO: handle more complicated arguments
            args = list(map(int, arg_str.split(",")))
            action_step.fun_args = args
        except ValueError as e:
            print(f"Error converting arguments: {e}")
            return None

        # Get the function from the tools map and call it
        if func_name == "Finish":
            action_step.observation = arg_str
            return action_step
        func = self.tools_map[func_name]
        try:
            result = func(*args)
            # TODO: why isnt the result of ToolOutput type?
            action_step.observation = result
            return action_step
        except Exception as e:
            print(f"Error executing {func_name} with args {args}: {e}")
            return None

    def _run_one_step(self, input: str, step: int) -> str:
        """
        Run one step of the agent.
        """
        # generate the prompt
        template = Template(self.prompt)
        prompt = template.render(
            user_query=input, tools=self.tools, histories=self.history
        )
        # get the response from the model
        messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        print(f"step {step}: {prompt}")
        response = self.generator(messages)
        print(f"raw generator output: {response}")
        parsed_response = self._parse_text_response(response, step)
        print(f"parsed_response: {parsed_response}")
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
        else:
            print(f"Failed to parse response for step {step}")
        self.history.append(parsed_response)

        print(f"parsed_response: {parsed_response}")

        return response

    def run(self, input: str) -> str:
        """
        Run the agent on the given input.
        """
        for i in range(self.max_steps):
            step = i + 1
            self._run_one_step(input, step)
            print("history:", self.history)
            # check if the task is finished
            if self.history[-1].fun_name == "Finish":
                break
        return self.history[-1].observation


if __name__ == "__main__":
    from lightrag.light_rag import OpenAIGenerator
    from extend.generator.groq_generator import GroqGenerator

    def multiply(a: int, b: int) -> int:
        """
        Multiply two numbers.
        """
        return a * b

    def add(a: int, b: int) -> int:
        """
        Add two numbers.
        """
        return a + b

    tools = [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
    ]
    settings = {
        "provider": "groq",
        "model": "llama3-8b-8192",  # llama3 is not good with string formatting
    }

    planner = GroqGenerator(**settings)
    # settings = {
    #     "provider": "openai",
    #     "model": "gpt-3.5-turbo",
    # }
    # planner = OpenAIGenerator(**settings)
    agent = ReActAgent(generator=planner, tools=tools, max_steps=10)
    answer = agent.run("What is 2 times 3?")
    print(f"Answer: {answer}")
