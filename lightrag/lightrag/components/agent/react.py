"""Implementation of ReAct."""

from typing import List, Union, Callable, Optional, Any, Dict
from copy import deepcopy
import logging


from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.func_tool import FunctionTool, AsyncCallable
from lightrag.core.tool_manager import ToolManager
from lightrag.components.output_parsers import JsonOutputParser
from lightrag.core.types import (
    StepOutput,
    GeneratorOutput,
    Function,
    FunctionOutput,
    FunctionExpression,
)
from lightrag.core.model_client import ModelClient
from lightrag.utils.logger import printc


log = logging.getLogger(__name__)

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<<SYS>>
{# role/task description #}
You are a helpful assistant.
Answer the user's query using the tools provided below with minimal steps and maximum accuracy.
{# REACT instructions #}
Each step you will read the previous Thought, Action, and Observation(execution result of the action) and then provide the next Thought and Action.
{# Tools #}
{% if tools %}
<TOOLS>
You available tools are:
{# tools #}
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
{# output is always more robust to use json than string #}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
<TASK_SPEC>
{# Specifications TODO: preference between the usage of llm tool vs the other tool #}
- For simple queries: Directly call the ``finish`` action and provide the answer.
- For complex queries:
    - Step 1: Read the user query and potentially divide it into subqueries. And get started with the first subquery.
    - Call one available tool at a time to solve each subquery/subquestion. \
    - At step 'finish', join all subqueries answers and finish the task.
Remember:
- Action must call one of the above tools with name. It can not be empty.
- You will always end with 'finish' action to finish the task. The answer can be the final answer or failure message.
</TASK_SPEC>
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
{% if input_str %}
User query:
{{ input_str }}
{% endif %}
{# Step History #}
{% if step_history %}
<STEPS>
{% for history in step_history %}
Step {{ loop.index }}.
{
 "thought": "{{history.thought}}",
 "action": "{{history.action.action}}",
}
"Observation": "{{history.observation}}"
------------------------
{% endfor %}
</STEPS>
{% endif %}
"""


class ReActAgent(Component):
    __doc__ = r"""ReActAgent uses generator as a planner that runs multiple and sequential functional call steps to generate the final response.

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
        from components.agent.react import ReActAgent
        from core.func_tool import FunctionTool
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

    Reference:
    [1] https://arxiv.org/abs/2210.03629, published in Mar, 2023.
    """

    def __init__(
        self,
        # added arguments specifc to React
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
        add_llm_as_fallback: bool = True,
        *,
        # the following arguments are mainly for the planner
        model_client: ModelClient,
        model_kwargs: Dict = {},
    ):
        super().__init__()
        template = DEFAULT_REACT_AGENT_SYSTEM_PROMPT

        self.max_steps = max_steps

        self.add_llm_as_fallback = add_llm_as_fallback

        self._init_tools(tools, model_client, model_kwargs)

        ouput_data_class = FunctionExpression
        example = FunctionExpression.from_function(
            thought="I have finished the task.",
            func=self._finish,
            answer="final answer: 'answer'",
        )
        output_parser = JsonOutputParser(data_class=ouput_data_class, example=example)
        prompt_kwargs = {
            "tools": self.tool_manager.yaml_definitions,
            "output_format_str": output_parser.format_instructions(),
        }
        self.planner = Generator(
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_parser,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )

        self.step_history: List[StepOutput] = []

    def _init_tools(
        self,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        model_client: ModelClient,
        model_kwargs: Dict,
    ):
        r"""Initialize the tools."""
        tools = deepcopy(tools)
        _additional_llm_tool = (
            Generator(model_client=model_client, model_kwargs=model_kwargs)
            if self.add_llm_as_fallback
            else None
        )

        def llm_tool(input: str) -> str:
            """I answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple."""
            # use the generator to answer the query
            try:
                output: GeneratorOutput = _additional_llm_tool(
                    prompt_kwargs={"input_str": input}
                )
                response = output.data if output else None
                return response
            except Exception as e:
                log.error(f"Error using the generator: {e}")
                print(f"Error using the generator: {e}")

            return None

        def finish(answer: str) -> str:
            """Finish the task with answer."""
            return answer

        self._finish = finish

        if self.add_llm_as_fallback:
            tools.append(llm_tool)
        tools.append(finish)
        self.tool_manager = ToolManager(tools=tools)

    def reset(self):
        r"""Reset the agent to start a new query."""
        self.step_history = []

    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """
        Parse the action string to a function call and execute it. Update the action_step with the result.
        """
        action = action_step.action
        try:

            fun: Function = self.tool_manager.parse_function_call_expr(action)
            result: FunctionOutput = self.tool_manager.execute_function(fun)
            # TODO: optimize the action_step
            action_step.fun_name = fun.name
            action_step.fun_args = fun.args
            action_step.fun_kwargs = fun.kwargs
            action_step.observation = result.output
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

        log.debug(
            f"Running step {step} with prompt: {self.planner.prompt(**prompt_kwargs)}"
        )

        # call the super class Generator to get the response
        response: GeneratorOutput = self.planner(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        step_output: StepOutput = None
        try:
            fun_expr: FunctionExpression = FunctionExpression.from_dict(response.data)
            step_output = StepOutput(
                step=step, thought=fun_expr.thought, action=fun_expr
            )
            # print the func expr
            log.debug(f"Step {step}: {fun_expr}")

            # execute the action
            if step_output and step_output.action:
                step_output = self._execute_action(step_output)
                printc(f"Step {step}: \n{step_output}\n_______\n", color="blue")
            else:
                log.error(f"Failed to parse response for step {step}")
        except Exception as e:
            log.error(f"Error running step {step}: {e}")
            if step_output is None:
                step_output = StepOutput(step=step, thought="", action="")
            else:
                step_output.observation = f"Error running step {step}: {e}"
        self.step_history.append(step_output)

        return response

    def call(
        self,
        input: str,
        promt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        prompt_kwargs = {**promt_kwargs, "input_str": input}
        # prompt_kwargs["input_str"] = input
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
        s = f"max_steps={self.max_steps}, add_llm_as_fallback={self.add_llm_as_fallback}"
        s += super()._extra_repr()
        return s


if __name__ == "__main__":
    from components.model_client import GroqAPIClient
    from lightrag.core.types import ModelClientType
    from lightrag.utils import setup_env

    setup_env()

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

    def divide(a: float, b: float) -> float:
        """
        Divide two numbers.
        """
        return float(a) / b

    def search(query: str) -> str:
        """
        Search the web for the given query.
        """
        return "python programming is a great way to learn programming"

    tools = [
        FunctionTool(fn=multiply),
        FunctionTool(fn=add),
        FunctionTool(fn=divide),
        # FunctionTool.from_defaults(fn=search),
    ]
    llm_model_kwargs = {
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        # mistral also not good: mixtral-8x7b-32768, but with better prompt, it can still work
        "temperature": 0.0,
    }

    gpt_3_5_turbo_model_kwargs = {
        "model": "gpt-3.5-turbo",
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
        "What is the capital of France? and what is 465 times 321 then add 95297 and then divide by 13.2?",
        # "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?",
        "Give me 5 words rhyming with cool, and make a 4-sentence poem using them",
    ]
    """
    Results: mixtral-8x7b-32768, 0.9s per query
    llama3-70b-8192, 1.8s per query
    gpt-3.5-turbo, 2.2s per query
    """
    import time

    generator = Generator(
        model_client=GroqAPIClient(),
        model_kwargs=llm_model_kwargs,
    )
    # for i in range(3):
    agent = ReActAgent(
        tools=tools,
        max_steps=5,
        model_client=ModelClientType.GROQ(),
        model_kwargs=llm_model_kwargs,
    )
    # agent.llm_planner.print_prompt()
    # print(agent)

    # vs not using agent
    # print(agent.tools)

    average_time = 0
    for query in queries:
        t0 = time.time()
        answer = agent(query)
        average_time += time.time() - t0
        answer_no_agent = generator(prompt_kwargs={"input_str": query})
        print(f"Answer with agent: {answer}")
        print(f"Answer without agent: {answer_no_agent}")
    print(f"Average time: {average_time / len(queries)}")
