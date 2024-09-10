"""Implementation and optimization of React agent."""

from typing import List, Union, Callable, Optional, Any, Dict
from copy import deepcopy
import logging


from adalflow.core.generator import Generator
from adalflow.core.component import Component
from adalflow.core.func_tool import FunctionTool, AsyncCallable
from adalflow.core.tool_manager import ToolManager
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.core.types import (
    StepOutput,
    GeneratorOutput,
    Function,
    FunctionOutput,
    FunctionExpression,
)
from adalflow.core.model_client import ModelClient
from adalflow.utils.logger import printc


log = logging.getLogger(__name__)

__all__ = ["DEFAULT_REACT_AGENT_SYSTEM_PROMPT", "ReActAgent"]

# TODO: test react agent

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<SYS>
{# role/task description #}
You are a helpful assistant.
Answer the user's query using the tools provided below with minimal steps and maximum accuracy.
{# REACT instructions #}
Each step you will read the previous Thought, Action, and Observation(execution result of the action) and then provide the next Thought and Action.
{# Tools #}
{% if tools %}
<TOOLS>
You available tools are:
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
{# output format and examples for output format #}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
<TASK_SPEC>
{# Task specification to teach the agent how to think using 'divide and conquer' strategy #}
- For simple queries: Directly call the ``finish`` action and provide the answer.
- For complex queries:
    - Step 1: Read the user query and potentially divide it into subqueries. And get started with the first subquery.
    - Call one available tool at a time to solve each subquery/subquestion. \
    - At step 'finish', join all subqueries answers and finish the task.
Remember:
- Action must call one of the above tools with name. It can not be empty.
- You will always end with 'finish' action to finish the task. The answer can be the final answer or failure message.
</TASK_SPEC>
</SYS>
-----------------
User query:
{{ input_str }}
{# Step History #}
{% if step_history %}
<STEPS>
Your previous steps:
{% for history in step_history %}
Step {{ loop.index }}.
"Thought": "{{history.action.thought}}",
"Action": "{{history.action.action}}",
"Observation": "{{history.observation}}"
------------------------
{% endfor %}
</STEPS>
{% endif %}
You:"""


class ReActAgent(Component):
    __doc__ = r"""ReActAgent uses generator as a planner that runs multiple and sequential functional call steps to generate the final response.

    Users need to set up:
    - tools: a list of tools to use to complete the task. Each tool is a function or a function tool.
    - max_steps: the maximum number of steps the agent can take to complete the task.
    - use_llm_as_fallback: a boolean to decide whether to use an additional LLM model as a fallback tool to answer the query.
    - model_client: the model client to use to generate the response.
    - model_kwargs: the model kwargs to use to generate the response.
    - template: the template to use to generate the prompt. Default is DEFAULT_REACT_AGENT_SYSTEM_PROMPT.

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
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

        # Using examples:

        call_multiply = FunctionExpression.from_function(
            thought="I want to multiply 3 and 4.",



    Reference:
    [1] https://arxiv.org/abs/2210.03629, published in Mar, 2023.
    """

    # TODO: allow users to pass in a few examples. Need to be a list of FunctionExpression instances.
    def __init__(
        self,
        # added arguments specifc to React
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
        add_llm_as_fallback: bool = True,
        # TODO: the examples are just for specifying the output format, not end to end input-output examples, need further optimization
        examples: List[FunctionExpression] = [],
        *,
        # the following arguments are mainly for the planner
        model_client: ModelClient,
        model_kwargs: Dict = {},
        template: Optional[str] = None,  # allow users to customize the template
    ):
        super().__init__()
        template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT

        self.max_steps = max_steps

        self.add_llm_as_fallback = add_llm_as_fallback

        self._init_tools(tools, model_client, model_kwargs)

        ouput_data_class = FunctionExpression
        example = FunctionExpression.from_function(
            thought="I have finished the task.",
            func=self._finish,
            answer="final answer: 'answer'",
        )
        self._examples = examples + [example]

        output_parser = JsonOutputParser(
            data_class=ouput_data_class, examples=self._examples, return_data_class=True
        )
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
        self.tool_manager: ToolManager = ToolManager(tools=tools)

    def reset(self):
        r"""Reset the agent to start a new query."""
        self.step_history = []

    # TODO: add async execution
    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """Parse the action string to a function call and execute it. Update the action_step with the result."""
        action = action_step.action
        try:

            fun: Function = self.tool_manager.parse_func_expr(action)
            result: FunctionOutput = self.tool_manager.execute_func(fun)
            # TODO: optimize the action_step
            action_step.function = fun
            action_step.observation = result.output
            return action_step
        except Exception as e:
            log.error(f"Error executing {action}: {e}")
            # pass the error as observation so that the agent can continue and correct the error in the next step
            action_step.observation = f"Error executing {action}: {e}"
            return action_step

    def _run_one_step(self, step: int, prompt_kwargs: Dict, model_kwargs: Dict) -> str:
        """Run one step of the agent. Plan and execute the action for the step."""
        step_output: StepOutput = StepOutput(step=step)
        prompt_kwargs["step_history"] = self.step_history

        log.debug(
            f"Running step {step} with prompt: {self.planner.prompt(**prompt_kwargs)}"
        )

        response: GeneratorOutput = self.planner(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        if response.error:
            error_msg = f"Error planning step {step}: {response.error}"
            step_output.observation = error_msg
            log.error(error_msg)
        else:
            try:
                fun_expr: FunctionExpression = response.data
                step_output.action = fun_expr
                log.debug(f"Step {step}: {fun_expr}")

                if step_output and step_output.action:
                    step_output = self._execute_action(step_output)
                    printc(f"Step {step}: \n{step_output}\n_______\n", color="blue")
                else:
                    log.error(f"Failed to parse response for step {step}")
            except Exception as e:
                error_msg = f"Error parsing response for step {step}: {e}"
                step_output.observation = error_msg
                log.error(error_msg)

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
        printc(f"input_query: {input}", color="red")
        for i in range(self.max_steps):
            step = i + 1
            try:
                self._run_one_step(step, prompt_kwargs, model_kwargs)
                if (
                    self.step_history[-1].function
                    and self.step_history[-1].function.name == "finish"
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
        s = f"max_steps={self.max_steps}, add_llm_as_fallback={self.add_llm_as_fallback}, "
        return s
