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
from lightrag.tool_helper import FunctionTool, AsyncCallable
from lightrag.string_parser import JsonParser, parse_function_call
from typing import List, Union, Callable, Optional, Any, Dict
from dataclasses import dataclass
from lightrag.light_rag import Generator


DEFAULT_REACT_AGENT_PROMPT = r"""
<START_OF_SYSTEAM_PROMPT>
{# role/task description #}
You task is to answer user's query with minimum steps and maximum accuracy using the tools provided.
{# REACT instructions #}
To do this, you will interleave Thought, Action, and Observation(execution result of the action) at each step.
Each step you will read the previous Thought, Action, and Observation, and then provide the next Thought and Action.

You have access to the following tools:
{# tools #}
{% for tool in tools %}
{{ loop.index }}. ToolName: {{ tool.metadata.name }}
    Tool Description: {{ tool.metadata.description }}
    Tool Args: {{ tool.metadata.fn_schema_str }} {#tool args can be misleading, especially if we already have type hints and docstring in the function#}
{% endfor %}
{# output is always more robust to use json than string #}
---
Your output must be in JSON format with two keys:
{
"thought": "<summarize previous actions if needed and provide the next thought>",
"action": "ToolName(<args>, <kwargs>)",
}
---
{# Specifications #}
Remember:
- Action must call one of the above tools with Took Name. It can not be empty. 
- Read the Tool Description and ensure your args and kwarg follow what each tool expects. e.g. (a=1, b=2) if it is keyword argument or (1, 2) if it is positional.
- You will always end with 'finish' action to finish the task. Call 'finish' as soon as you can answer the initial user query.
- If the argument is a string, it must be enclosed in single quotes.
- Do not do expressional evaluation in the action. For example, 2+3 should be avoided to be passed as an argument.
{#Examples can be here#}
{# Check if there are any examples #}
{% if examples %}
Examples:
{% for example in examples %}
{{ example }}
{% endfor %}
{% endif %}
<END_OF_SYSTEAM_PROMPT>
-----------------
User query: {{user_query}}
{# History #}
{% for history in step_history %}
Step {{history.step}}:
{
 "thought": "{{history.thought}}",
 "action": "{{history.action}}",
}
"observation": "{{history.observation}}"
{% endfor %}
You:
"""

# NOTE: if the positional and keyword arguments are not working well,
# you can let it be a json string and use only keyword arguments and use json parser to parse the arguments instead of parse_function_call


@dataclass
class StepOutput:
    step: int
    thought: str
    action: str
    fun_name: Optional[str] = None  # parsed from action
    fun_args: Optional[List[Any]] = None  # parsed from action
    fun_kwargs: Optional[Dict[str, Any]] = None  # parsed from action
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
        examples: List[str] = [],
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
    ):
        self.prompt = generator_prompt
        self.examples = examples
        # convert all functions to FunctionTool, and track how to call each function, either call or acall
        self.tools = [
            (
                tool
                if isinstance(tool, FunctionTool)
                else FunctionTool.from_defaults(fn=tool)
            )
            for tool in tools
        ]

        def finish(answer: str) -> str:
            """
            Finish the task by returning the answer.
            """
            return answer

        finish_tool = FunctionTool.from_defaults(fn=finish)
        self.tools.append(finish_tool)

        self.tools_map = {tool.metadata.name: tool for tool in self.tools}
        print(f"tools_map: {self.tools_map}")

        self.generator = generator
        self.max_steps = max_steps
        self.step_history: List[StepOutput] = []
        self.text_output_parser = JsonParser()

    def reset(self):
        self.step_history = []

    def _parse_text_response(self, response: str, step: int) -> Optional[StepOutput]:
        """
        Parse the json output"""
        try:
            json_obj_response = self.text_output_parser(response)
            thought_key = "thought"
            action_key = "action"
            thought = json_obj_response.get(thought_key, "")
            action = json_obj_response.get(action_key, "")
            return StepOutput(step=step, thought=thought, action=action)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """
        Parse the action string to a function call and execute it. Update the action_step with the result.
        """
        action = action_step.action
        try:
            fun_name, args, kwargs = parse_function_call(action, self.tools_map)
            fun: Union[Callable, AsyncCallable] = self.tools_map[fun_name].fn
            result = fun(*args, **kwargs)
            action_step.fun_name = fun_name
            action_step.fun_args = args
            action_step.fun_kwargs = kwargs

            action_step.observation = result
            return action_step
        except Exception as e:
            print(f"Error executing {action}: {e}")
            return None

    def _run_one_step(self, input: str, step: int) -> str:
        """
        Run one step of the agent.
        """
        template = Template(self.prompt)
        prompt = template.render(
            user_query=input,
            tools=self.tools,
            step_history=self.step_history,
            examples=self.examples,
        )
        messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        response = self.generator(messages)
        print(f"raw generator output: {response}")
        parsed_response = self._parse_text_response(response, step)
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
        else:
            print(f"Failed to parse response for step {step}")
        self.step_history.append(parsed_response)

        return response

    def run(self, input: str) -> str:
        """
        Run the agent on the given input.
        """
        for i in range(self.max_steps):
            step = i + 1
            self._run_one_step(input, step)
            if self.step_history[-1].fun_name == "finish":
                break
        answer = self.step_history[-1].observation
        self.reset()
        return answer


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

    def search(query: str) -> str:
        """
        Search the web for the given query.
        """
        return f"python programming is a great way to learn programming"

    tools = [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=search),
    ]
    settings = {
        "provider": "groq",
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        # mistral also not good: mixtral-8x7b-32768, but with better prompt, it can still work
    }

    planner = GroqGenerator(**settings)
    settings = {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
    }
    planner = OpenAIGenerator(**settings)
    agent = ReActAgent(generator=planner, tools=tools, max_steps=10)
    queries = [
        "What is 2 times 3?",
        "What is 3 plus 4?",
        "Search for 'python programming'",
    ]
    """
    Results: mixtral-8x7b-32768, 0.9s per query
    llama3-70b-8192, 1.8s per query
    gpt-3.5-turbo, 2.2s per query
    """
    import time

    average_time = 0
    for query in queries:
        t0 = time.time()
        answer = agent.run(query)
        average_time += time.time() - t0
        print(f"Answer: {answer}")
    print(f"Average time: {average_time / len(queries)}")
