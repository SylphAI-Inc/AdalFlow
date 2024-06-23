"""
ReAct Agent leveraging LLM reasoning and function calling.

Source: https://arxiv.org/abs/2210.03629, published in Mar, 2023

Agent is not a model or LLM model.
Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.

The future: the agent can write your prompts too. Check out dspy: https://github.com/stanfordnlp/dspy

ReAct agent can be useful for
- Multi-hop reasoning [Q&A], including dividing the query into subqueries and answering them one by one.
- Plan the usage of the given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools.

The initial ReAct paper does not support different types of tools. We have greatly extended the flexibility of tool adaption, even including an llm tool
to answer questions that cant be answered or better be answered by llm using its world knowledge.
- Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
- 'finish' tool is defined to finish the task by joining all subqueries answers.
"""

from typing import List, Union, Callable, Optional, Any, Dict
from copy import deepcopy
import logging

from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.tool_helper import FunctionTool, AsyncCallable
from lightrag.core.string_parser import JsonParser, parse_function_call
from lightrag.core.generator import GeneratorOutput
from lightrag.core.model_client import ModelClient
from lightrag.core.types import StepOutput
from lightrag.utils.logger import printc

log = logging.getLogger(__name__)

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
{% if input_str %}
User query:
{{ input_str }}
{% endif %}
"""


# TODO: add better logging @xiaoyi


class ReActAgent(Generator):
    r"""
    ReActAgent is a type of Generator that runs multiple and sequential steps to generate the final response, with DEFAULT_REACT_AGENT_SYSTEM_PROMPT and JsonParser output_processors.

    Users need these arguments to initialize the ReActAgent:
    - tools: a list of tools to use to complete the task. Each tool is a function or a function tool.
    - max_steps: the maximum number of steps the agent can take to complete the task.
    - All other arguments are inherited from Generator such as model_client, model_kwargs, prompt, output_processors, etc.

    There are `examples` which is optional, a list of string examples in the prompt.

    Example:
        .. code-block:: python

            from lightrag.core.tool_helper import FunctionTool
            from lightrag.components.agent.react_agent import ReActAgent
            from lightrag.components.model_client import GroqAPIClient

            import time
            import dotenv
            # load evironment
            dotenv.load_dotenv(dotenv_path=".env", override=True)

            # define the tools
            def multiply(a: int, b: int) -> int:
                '''Multiply two numbers.'''
                return a * b
            def add(a: int, b: int) -> int:
                '''Add two numbers.'''
                return a + b

            tools = [
                FunctionTool.from_defaults(fn=multiply),
                FunctionTool.from_defaults(fn=add),
            ]

            # set up examples
            examples = [
                "your example, a human-like task-solving trajectory"
            ]
            # preset examples in the prompt
            preset_prompt_kwargs = {"example": examples}

            # set up llm args
            llm_model_kwargs = {
                "model": "llama3-70b-8192",
                "temperature": 0.0
            }

            # initialze an agent
            agent = ReActAgent(
                tools=tools,
                model_client=GroqAPIClient(),
                model_kwargs=llm_model_kwargs,
                max_steps=3,
                preset_prompt_kwargs=preset_prompt_kwargs
            )

            # query the agent
            queries = ["What is 3 add 4?", "3*9=?"]
            average_time = 0
            for query in queries:
                t0 = time.time()
                answer = agent(query)

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
        output_processors: Optional[Component] = None,
        model_client: ModelClient,
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
        self.output_processors = output_processors or JsonParser()

        self.additional_llm_tool = Generator(
            model_client=model_client, model_kwargs=model_kwargs
        )

        def llm_tool(input: str) -> str:
            """
            Answer any input query with llm's world knowledge. Use it as a fallback tool or when the query is simple.
            """
            # use the generator to answer the query
            prompt_kwargs = {
                "input_str": input
            }  # wrap the query input in the local prompt_kwargs
            try:
                response = self.additional_llm_tool.call(prompt_kwargs=prompt_kwargs)
                json_response = (
                    response.data if isinstance(response, GeneratorOutput) else response
                )  # get json data from GeneratorOutput
                # print(f"response: {response}, json_response: {json_response}")
                return json_response
            except Exception as e:
                # print(f"Error using the generator: {e}")
                log.error(f"Error using the generator: {e}")

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
        self.prompt.update_preset_prompt_kwargs(tools=self.tools)

        self.tools_map = {tool.metadata.name: tool for tool in self.tools}
        self.step_history: List[StepOutput] = []

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
            # print(f"Error parsing response: {e}")
            log.error(f"Error parsing response: {e}")
            return None

    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """
        Parse the action string to a function call and execute it. Update the action_step with the result.
        """
        action = action_step.action
        try:
            fun_name, args, kwargs = parse_function_call(action, self.tools_map)
            # print(f"fun_name: {fun_name}, args: {args}, kwargs: {kwargs}")
            fun: Union[Callable, AsyncCallable] = self.tools_map[fun_name].fn
            result = fun(*args, **kwargs)
            action_step.fun_name = fun_name
            action_step.fun_args = args
            action_step.fun_kwargs = kwargs

            action_step.observation = result
            return action_step
        except Exception as e:
            # print(f"Error executing {action}: {e}")
            log.error(f"Error executing {action}: {e}")
            # pass the error as observation so that the agent can continue and correct the error in the next step
            action_step.observation = f"Error executing {action}: {e}"
            return action_step

    def _run_one_step(self, step: int, prompt_kwargs: Dict, model_kwargs: Dict) -> str:
        """
        Run one step of the agent.
        """
        # step_history is the only per-query variable, and should not be controlled by the user
        # add the step_history to the prompt_kwargs
        prompt_kwargs["step_history"] = self.step_history

        # call the super class Generator to get the response
        response = super().call(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )  # response is GeneratorOutput

        # get json response data from generator output
        json_response = (
            response.data if isinstance(response, GeneratorOutput) else response
        )

        parsed_response = self._parse_text_response(
            json_obj_response=json_response, step=step
        )
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
            printc(f"step: {step}, response: {parsed_response}", color="blue")
        else:
            # print(f"Failed to parse response for step {step}")
            log.error(f"Failed to parse response for step {step}")
        self.step_history.append(parsed_response)

        return response

    def call(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        # wrap up the input in the prompt_kwargs
        prompt_kwargs["input_str"] = input
        printc(f"input_query: {input}", color="cyan")
        for i in range(self.max_steps):
            step = i + 1
            try:
                self._run_one_step(step, prompt_kwargs, model_kwargs)
                if (
                    self.step_history[-1].fun_name
                    and self.step_history[-1].fun_name == "finish"
                ):
                    break
            except Exception as e:
                error_message = f"Error running step {step}: {e}"
                # print(error_message)
                log.error(error_message)
        try:
            answer = self.step_history[-1].observation
        except Exception:
            answer = None
        printc(f"answer: {answer}", color="magneta")
        # print(f"step_history: {self.step_history}")
        log.info(f"step_history: {self.step_history}")
        self.reset()
        return answer

    def _extra_repr(self) -> str:
        s = f"tools={self.tools}, max_steps={self.max_steps}, "
        s += super()._extra_repr()
        return s
