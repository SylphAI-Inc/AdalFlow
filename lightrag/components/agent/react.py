"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Agent is not a model or LLM model.
Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.


REact agent can be useful for
- Multi-hop reasoning [Q&A], including dividing the query into subqueries and answering them one by one.
- Plan the usage of the given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools.

The initial ReAct paper does not support different types of tools. We have greatly extended the flexibility of tool adaption, even including an llm tool
to answer questions that cant be answered or better be answered by llm using its world knowledge.
- Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
- 'finish' tool is defined to finish the task by joining all subqueries answers.

Reference:
[1] LLM Agent survey: https://github.com/Paitesanshi/LLM-Agent-Survey
"""

from typing import List, Union, Callable, Optional, Any, Dict
from copy import deepcopy
import logging


from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.tool_helper import FunctionTool, AsyncCallable
from lightrag.core.string_parser import JsonParser, parse_function_call
from lightrag.core.types import StepOutput, GeneratorOutput
from lightrag.core.model_client import ModelClient
from lightrag.utils.logger import printc


log = logging.getLogger(__name__)

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<<SYS>>
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
{# Step History #}
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


class ReActAgent(Generator):
    __doc__ = r"""ReActAgent is a subclass of Generator that runs multiple and sequential functional call steps to generate the final response.

    Users need to set up:
    - tools: a list of tools to use to complete the task. Each tool is a function or a function tool.
    - max_steps: the maximum number of steps the agent can take to complete the task.
    - use_llm_as_fallback: a boolean to decide whether to use an additional LLM model as a fallback tool to answer the query.
    - model_client: the model client to use to generate the response.
    - model_kwargs: the model kwargs to use to generate the response.

    For the generator, the default arguments are:
    (1) default prompt: DEFAULT_REACT_AGENT_SYSTEM_PROMPT
    (2) default output_processors: JsonParser

    There are `examples` which is optional, a list of string examples in the prompt.

    Example:

    .. code-block:: python
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
    """

    def __init__(
        self,
        # added arguments specifc to React
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
        add_llm_as_fallback: bool = True,
        *,
        # the following arguments are inherited from Generator
        model_client: ModelClient,
        model_kwargs: Dict = {},
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict] = {},
        output_processors: Optional[Component] = None,
    ):
        assert "model" in model_kwargs, "model must be provided in model_kwargs"
        assert model_client, "model_client must be provided"
        # assert tools and len(tools) > 0, "At least one tool must be provided"

        template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT
        super().__init__(
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_processors,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        self.tools = deepcopy(tools)
        self.max_steps = max_steps
        self.output_processors = output_processors or JsonParser()

        self._additional_llm_tool = (
            Generator(model_client=model_client, model_kwargs=model_kwargs)
            if add_llm_as_fallback
            else None
        )

        def llm_tool(input: str) -> str:
            """
            I answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple.
            """
            # use the generator to answer the query
            try:
                output: GeneratorOutput = self._additional_llm_tool(
                    prompt_kwargs={"input_str": input}
                )
                response = output.data if output else None
                return response
            except Exception as e:
                log.error(f"Error using the generator: {e}")
                print(f"Error using the generator: {e}")

            return None

        def finish(answer: str) -> str:
            """
            Finish the task by joinging all subqueries answers.
            """
            return answer

        if add_llm_as_fallback:
            self.tools.append(llm_tool)
        self.tools.append(finish)

        # convert all functions to FunctionTool, and track how to call each function, either call or acall
        self.tools = [
            (tool if isinstance(tool, FunctionTool) else FunctionTool(fn=tool))
            for tool in self.tools
        ]
        # pass the tools to the prompt
        self.prompt.update_prompt_kwargs(tools=self.tools)

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
            log.error(f"Error parsing response: {e}")
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
        response: GeneratorOutput = super().call(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        parsed_response = self._parse_text_response(
            json_obj_response=response.data, step=step
        )
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
            printc(f"Step {step}: \n{parsed_response}\n_______\n", color="blue")
        else:
            log.error(f"Failed to parse response for step {step}")
        self.step_history.append(parsed_response)

        return response

    def call(
        self,
        input: str,
        promt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        prompt_kwargs = deepcopy(promt_kwargs)
        prompt_kwargs["input_str"] = input
        printc(f"input_query: {input}", color="red")
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
                log.error(f"Error running step {step}: {e}")

        answer = self.step_history[-1].observation
        printc(f"answer:\n {answer}", color="green")
        log.info(f"step_history: {self.step_history}")
        self.reset()
        return answer

    def _extra_repr(self) -> str:
        s = f"tools={self.tools}, max_steps={self.max_steps}, "
        s += super()._extra_repr()
        return s


if __name__ == "__main__":
    from components.model_client import GroqAPIClient
    from lightrag.utils import setup_env  # noqa

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
        FunctionTool(fn=multiply),
        FunctionTool(fn=add),
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
    # agent = ReActAgent(
    #     # examples=examples,
    #     tools=tools,
    #     max_steps=5,
    #     model_client=GroqAPIClient,
    #     model_kwargs=llm_model_kwargs,
    # )
    # print(agent)
    queries = [
        # "What is 2 times 3?",
        # "What is 3 plus 4?",
        # "What is the capital of France? and what is 4 times 5 then add 3?",  # this is actually two queries, or a multi-hop query
        # "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?",
        "Give me 5 words rhyming with cool, and make a 4-sentence poem using them",
    ]
    """
    Results: mixtral-8x7b-32768, 0.9s per query
    llama3-70b-8192, 1.8s per query
    gpt-3.5-turbo, 2.2s per query
    """
    import time

    tools = []
    for i in range(3):
        agent = ReActAgent(
            tools=[],
            max_steps=5,
            model_client=GroqAPIClient(),
            model_kwargs=llm_model_kwargs,
        )
    # print(agent.tools)

    average_time = 0
    for query in queries:
        t0 = time.time()
        answer = agent(query)
        average_time += time.time() - t0
    print(f"Average time: {average_time / len(queries)}")
