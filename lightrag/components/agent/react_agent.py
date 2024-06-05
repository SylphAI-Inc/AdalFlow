"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Agent is not a model or LLM model.
Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.

The future: the agent can write your prompts too. Check out dspy: https://github.com/stanfordnlp/dspy

REact agent can be useful for
- Multi-hop reasoning [Q&A], including dividing the query into subqueries and answering them one by one.
- Plan the usage of the given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools.

The initial ReAct paper does not support different types of tools. We have greatly extended the flexibility of tool adaption, even including an llm tool
to answer questions that cant be answered or better be answered by llm using its world knowledge.
- Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
- 'finish' tool is defined to finish the task by joining all subqueries answers.
"""

from typing import List, Union, Callable, Optional, Any, Dict
from dataclasses import dataclass
from copy import deepcopy


from core.generator import Generator
from core.component import Component
from core.tool_helper import FunctionTool, AsyncCallable
from core.string_parser import JsonParser, parse_function_call

from core.api_client import APIClient

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""
{# role/task description #}
You task is to answer user's query with minimum steps and maximum accuracy using the tools provided.
{# REACT instructions #}
Each step you will read the previous Thought, Action, and Observation(execution result of the action)steps and then provide the next Thought and Action.

You only have access to the following tools:
{# tools #}
{% for tool in tools %}
{{ loop.index }}. ToolName: {{ tool.metadata.name }}
    Tool Description: {{ tool.metadata.description }}
    Tool Parameters: {{ tool.metadata.fn_schema_str }} {#tool args can be misleading, especially if we already have type hints and docstring in the function#}
{% endfor %}
{# output is always more robust to use json than string #}
---
Your output must be in valid JSON format(raw Python string format) with two keys:
{
    "thought": "<Why you are taking this action>",
    "action": "ToolName(<args>, <kwargs>)"
}
- Must double quote the JSON str.
- Inside of the JSON str, Must use escape double quote and escape backslash for string.
For example:
"action": "finish(\"John's.\")"
---
{# Specifications TODO: preference between the usage of llm tool vs the other tool #}
Process:
- Step 1: Read the user query and potentially divide it into subqueries. And get started with the first subquery.
- Call one available tool at a time to solve each subquery/subquestion. \
- At step 'finish', join all subqueries answers and finish the task.
Remember:
- Action must call one of the above tools with Took Name. It can not be empty.
- Read the Tool Description and ensure your args and kwarg follow what each tool expects in types. e.g. (a=1, b=2) if it is keyword argument or (1, 2) if it is positional.
- You will always end with 'finish' action to finish the task. The answer can be the final answer or failure message.
- When the initial query is simple, use minimum steps to answer the query.
{#Examples can be here#}
{# Check if there are any examples #}
{% if examples %}
<EXAMPLES>
{% for example in examples %}
{{ example }}
{% endfor %}
</EXAMPLES>
{% endif %}
<</SYS>>
-----------------
{# History #}
{% for history in step_history %}
Step {{history.step}}:
{
 "thought": "{{history.thought}}",
 "action": "{{history.action}}",
}
"observation": "{{history.observation}}"
{% endfor %}
"""


# TODO: add better logging @xiaoyi
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


class ReActAgent(Generator):
    r"""
    ReActAgent is a type of Generator that runs multiple and sequential steps to generate the final response, with DEFAULT_REACT_AGENT_SYSTEM_PROMPT and JsonParser output_processors.

    Users need these arguments to initialize the ReActAgent:
    - tools: a list of tools to use to complete the task. Each tool is a function or a function tool.
    - max_steps: the maximum number of steps the agent can take to complete the task.
    - All other arguments are inherited from Generator such as model_client, model_kwargs, prompt, output_processors, etc.

    There are `examples` which is optional, a list of string examples in the prompt.

    Example:
    ```
    from core.openai_client import OpenAIClient
    from components.agent.react_agent import ReActAgent
    from core.tool_helper import FunctionTool
    # define the tools
    def multiply(a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b
    agent = ReActAgent(
    tools=[multiply, add],
    model_client=OpenAIClient,
    model_kwargs={"model": "gpt-3.5-turbo"},
    )

    Using examples:

    preset_prompt_kwargs = {"examples": examples}
    agent = ReActAgent(
    tools=[multiply, add],
    model_client=OpenAIClient,
    model_kwargs={"model": "gpt-3.5-turbo"},
    preset_prompt_kwargs=preset_prompt_kwargs,
    )
    ```
    """

    def __init__(
        self,
        # added arguments specifc to React
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
        *,
        # the following arguments are inherited from Generator
        template: str = DEFAULT_REACT_AGENT_SYSTEM_PROMPT,
        preset_prompt_kwargs: Optional[
            Dict
        ] = {},  # you can pass examples here, additionally leverage few-shot or many-shots ICL.
        output_processors: Optional[Component] = JsonParser(),
        model_client: APIClient,
        model_kwargs: Optional[Dict] = {},
    ):
        super().__init__(
            template=template,
            preset_prompt_kwargs=preset_prompt_kwargs,
            output_processors=output_processors,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        self.tools = deepcopy(tools)
        self.max_steps = max_steps

        self.additional_llm_tool = Generator(
            model_client=model_client, model_kwargs=model_kwargs
        )

        def llm_tool(input: str) -> str:
            """
            I answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple.
            """
            # use the generator to answer the query
            try:
                return self.additional_llm_tool(input=input)
            except Exception as e:
                print(f"Error using the generator: {e}")

            return None

        def finish(answer: str) -> str:
            """
            Finish the task by joinging all subqueries answers.
            """
            return answer

        self.tools.extend([llm_tool, finish])
        # convert all functions to FunctionTool, and track how to call each function, either call or acall
        self.tools = [
            (
                tool
                if isinstance(tool, FunctionTool)
                else FunctionTool.from_defaults(fn=tool)
            )
            for tool in self.tools
        ]
        # pass the tools to the prompt
        self.system_prompt.update_preset_prompt_kwargs(tools=self.tools)

        self.tools_map = {tool.metadata.name: tool for tool in self.tools}
        self.step_history: List[StepOutput] = []
        self.output_processors = output_processors

    def reset(self):
        r"""Reset the agent to start a new query."""
        self.step_history = []

    def _parse_text_response(
        self, json_obj_response: Dict[str, Any], step: int
    ) -> Optional[StepOutput]:
        """
        Parse the json output
        """
        try:
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
            print(f"fun_name: {fun_name}, args: {args}, kwargs: {kwargs}")
            fun: Union[Callable, AsyncCallable] = self.tools_map[fun_name].fn
            result = fun(*args, **kwargs)
            action_step.fun_name = fun_name
            action_step.fun_args = args
            action_step.fun_kwargs = kwargs

            action_step.observation = result
            return action_step
        except Exception as e:
            print(f"Error executing {action}: {e}")
            # pass the error as observation so that the agent can continue and correct the error in the next step
            action_step.observation = f"Error executing {action}: {e}"
            return action_step

    def _run_one_step(
        self, input: str, step: int, prompt_kwargs: Dict, model_kwargs: Dict
    ) -> str:
        """
        Run one step of the agent.
        """
        # step_history is the only per-query variable, and should not be controlled by the user
        # add the step_history to the prompt_kwargs
        prompt_kwargs["step_history"] = self.step_history

        # call the super class Generator to get the response
        response = super().call(
            input=input, prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        parsed_response = self._parse_text_response(
            json_obj_response=response, step=step
        )
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
        else:
            print(f"Failed to parse response for step {step}")
        self.step_history.append(parsed_response)

        return response

    def call(
        self,
        input: str,
        promt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        for i in range(self.max_steps):
            step = i + 1
            try:
                self._run_one_step(input, step, promt_kwargs, model_kwargs)
                if (
                    self.step_history[-1].fun_name
                    and self.step_history[-1].fun_name == "finish"
                ):
                    break
            except Exception as e:
                error_message = f"Error running step {step}: {e}"
                print(error_message)

        answer = self.step_history[-1].observation
        print(f"step_history: {self.step_history}")
        self.reset()
        return answer

    def _extra_repr(self) -> str:
        s = f"tools={self.tools}, max_steps={self.max_steps}, "
        s += super()._extra_repr()
        return s


if __name__ == "__main__":
    from components.api_client import GroqAPIClient
    import utils.setup_env

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
        return "python programming is a great way to learn programming"

    tools = [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        # FunctionTool.from_defaults(fn=search),
    ]
    llm_model_kwargs = {
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        # mistral also not good: mixtral-8x7b-32768, but with better prompt, it can still work
        "temperature": 0.0,
    }

    examples = [
        # r"""
        # User: What is 9 - 3?
        # You: {
        #     "thought": "I need to subtract 3 from 9, but there is no subtraction tool, so I ask llm_tool to answer the query.",
        #     "action": "llm_tool('What is 9 - 3?')"
        # }
        # """
    ]
    agent = ReActAgent(
        # examples=examples,
        tools=tools,
        max_steps=5,
        model_client=GroqAPIClient,
        model_kwargs=llm_model_kwargs,
    )
    print(agent)
    queries = [
        # "What is 2 times 3?",
        # "What is 3 plus 4?",
        # "What is the capital of France? and what is 4 times 5 then add 3?",  # this is actually two queries, or a multi-hop query
        "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?",
    ]
    """
    Results: mixtral-8x7b-32768, 0.9s per query
    llama3-70b-8192, 1.8s per query
    gpt-3.5-turbo, 2.2s per query
    """
    import time

    for i in range(3):
        agent = ReActAgent(
            tools=tools,
            max_steps=5,
            model_client=GroqAPIClient,
            model_kwargs=llm_model_kwargs,
        )
    print(agent.tools)

    # average_time = 0
    # for query in queries:
    #     t0 = time.time()
    #     answer = agent(query)
    #     average_time += time.time() - t0
    #     print(f"Answer: {answer}")
    # print(f"Average time: {average_time / len(queries)}")
