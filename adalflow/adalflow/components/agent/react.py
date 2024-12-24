"""Implementation and optimization of React agent."""

from typing import List, Union, Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from adalflow.core.base_data_class import DataClass
from copy import deepcopy
import logging


from adalflow.core.generator import Generator
from adalflow.optim.grad_component import GradComponent
from adalflow.optim.parameter import Parameter, ParameterType
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

react_agent_task_desc = r"""{# role/task description #}
You are a helpful assistant.
Answer the user's query using the tools provided below with minimal steps and maximum accuracy.
{# REACT instructions #}
Each step you will read the previous Thought, Action, and Observation(execution result of the action) and then provide the next Thought and Action.

<START_OF_TASK_SPEC>
{# Task specification to teach the agent how to think using 'divide and conquer' strategy #}
- For simple queries: Directly call the ``finish`` action and provide the answer.
- For complex queries:
    - Step 1: Read the user query and potentially divide it into subqueries. And get started with the first subquery.
    - Call one available tool at a time to solve each subquery/subquestion. \
    - At step 'finish', join all subqueries answers and finish the task.
Remember:
- Action must call one of the above tools with name. It can not be empty.
- You will always end with 'finish' action to finish the task. The answer can be the final answer or failure message.
<END_OF_TASK_SPEC>
"""

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<START_OF_SYSTEM_PROMPT>
{{react_agent_task_desc}}
{# Tools #}
{% if tools %}
<START_OF_TOOLS>
You available tools are:
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
<END_OF_TOOLS>
{% endif %}
{# output format and examples for output format #}
<START_OF_OUTPUT_FORMAT>
{{output_format_str}}
<END_OF_OUTPUT_FORMAT>
<END_OF_SYSTEM_PROMPT>
-----------------
<START_OF_USER_QUERY>
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
<END_OF_USER_QUERY>
"""

# We have parameters react_agent_task_desc, tools, output_format_str, input_str, step_history
# react_agent_task_desc is trainable per use case
# step_history is a list to track the history, where each time it will be updated with the current step output


class AppendStepHistory(GradComponent):
    def __init__(self):
        super().__init__()

    def call(
        self, step_output: StepOutput, step_history: List[StepOutput]
    ) -> List[StepOutput]:
        """Append the step_output to the step_history."""
        if not step_history:
            step_history = []
        # make a copy step_history for better tracking
        step_history = deepcopy(step_history)

        step_history.append(step_output)
        # printc(f"step_history: {step_history}", color="yellow")
        return step_history


# class GeneroutorOutputToStepOutput(GradComponent):
#     def __init__(self):
#         super().__init__()

#     def call(
#         self,
#         generator_output: GeneratorOutput,
#         step_output: StepOutput,
#         step: int,
#         execute_action: Any,
#     ) -> StepOutput:
#         """Convert the generator output to the step output."""
#         return execute_action_fn(generator_output, step_output, step, execute_action)


# TODO: make execute_action_fn to a GradComponent to enable the training of the tools too.
def execute_action_fn(
    x: GeneratorOutput, step_output: StepOutput, step: int, execute_action: Any
) -> StepOutput:
    """Execute the action and update the step_output."""
    if x.error:
        error_msg = f"Error planning step {step}: {x.error}"
        step_output.observation = error_msg
        log.error(error_msg)
    else:
        try:
            fun_expr: FunctionExpression = x.data
            step_output.action = fun_expr
            log.debug(f"Step {step}: {fun_expr}")

            if step_output and step_output.action:
                step_output = execute_action(step_output)
                printc(f"Step {step}: \n{step_output}\n_______\n", color="blue")
                return step_output
            else:
                printc(f"Failed to parse response for step {step}", color="red")
                log.error(f"Failed to parse response for step {step}")
                return step_output
        except Exception as e:
            error_msg = f"Error parsing response for step {step}: {e}"
            step_output.observation = error_msg
            log.error(error_msg)
            printc(error_msg, color="red")
            return step_output


@dataclass
class ReActOutput(DataClass):
    r"""Similar to GeneratorOutput, but with additional step history and final answer."""

    id: Optional[str] = field(
        default=None, metadata={"desc": "The unique id of the output"}
    )
    step_history: List[StepOutput] = field(
        metadata={"desc": "The history of steps."}, default_factory=list
    )

    answer: Any = field(metadata={"desc": "The final answer."}, default=None)


class ReActAgent(GradComponent):
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
        # examples: List[FunctionExpression] = [],
        examples: Union[List[FunctionExpression], List[str]] = [],
        *,
        # the following arguments are mainly for the planner
        model_client: ModelClient,
        model_kwargs: Dict = {},
        # template for the planner
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
            "react_agent_task_desc": Parameter(
                name="react_agent_task_desc",
                data=react_agent_task_desc,
                role_desc="Task description for the ReAct agent which functions as a planner using a Large Language Model.",
                param_type=ParameterType.PROMPT,
                requires_opt=True,
            ),
        }
        self.planner = Generator(
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_parser,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )

        # added this component to the computation graph
        self.append_step_history = AppendStepHistory()

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

    def _run_one_step(
        self,
        step: int,
        prompt_kwargs: Dict,
        model_kwargs: Dict,
        id: Optional[str] = None,
        step_history: Union["Parameter", List[str]] = None,
    ) -> Union[StepOutput, Parameter]:
        """Run one step of the agent. Plan and execute the action for the step.
        Need to deal with both train and eval mode on the self.planner.
        """
        from functools import partial

        prompt_kwargs["step_history"] = step_history

        log.debug(
            f"Running step {step} with prompt: {self.planner.prompt(**prompt_kwargs)}"
        )

        response: Union[GeneratorOutput, Parameter] = self.planner(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs, id=id
        )

        # create a new step output
        step_output: StepOutput = StepOutput(step=step)

        # connecting two generators in the computation graph, it will set up self.step_history
        if isinstance(response, Parameter):

            # connect the next planner with this current response
            def map_fn(
                x: Parameter, step_output: StepOutput = step_output
            ) -> StepOutput:
                if x and hasattr(x, "full_response"):
                    return execute_action_fn(
                        x.full_response, step_output, step, self._execute_action
                    )
                else:
                    raise ValueError(
                        f"Error: {x} does not have full_response attribute."
                    )

            # Bind `step_output` to a specific value using partial
            preinitialized_map_fn = partial(map_fn, step_output=step_output)
            # execute the function and get the output

            # # connect response to append_step_history
            response.add_successor_map_fn(
                successor=self.append_step_history, map_fn=preinitialized_map_fn
            )

            step_history = self.append_step_history.forward(response, step_history)
            # connect step_history to the next planner
            step_history.add_successor_map_fn(
                successor=self.planner, map_fn=lambda x: x.data
            )
            # convert step history back to data
            return step_history

        else:
            execute_action_fn(response, step_output, step, self._execute_action)
            step_history.append(step_output)
            return step_history

    def _check_last_step(
        self, step_history: Union["Parameter", List[str]] = None
    ) -> bool:
        """Check if the last step is the finish step."""
        if not step_history:
            return True

        last_step: StepOutput = None
        if isinstance(step_history, Parameter):
            try:
                step_history = step_history.data
                last_step = step_history[-1]

            except Exception as e:
                log.error(f"Error getting data from Parameter: {e}")
                return False
        else:
            last_step = step_history[-1]

        if last_step and last_step.function and last_step.function.name == "finish":
            return True
        return False

    def _get_answer(
        self, step_history: Union["Parameter", List[str]] = None
    ) -> Union[str, "Parameter"]:
        """Get the final answer from the step history."""
        if not step_history:
            return None

        last_step: StepOutput = None
        if isinstance(step_history, Parameter):
            try:
                return step_history

            except Exception as e:
                log.error(f"Error getting data from Parameter: {e}")
                return None
        else:
            last_step = step_history[-1]

            return last_step.observation

    def call(self, *args, **kwargs):
        return self.bicall(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Parameter:
        return self.bicall(*args, **kwargs)

    def _is_step_output_last_step(self, step_output: StepOutput) -> bool:
        """Check if the step output is the last step."""
        step_output_data = (
            step_output.data if isinstance(step_output, Parameter) else step_output
        )
        if (
            step_output_data
            and step_output_data.function
            and step_output_data.function.name == "finish"
        ):
            return True
        return False

    def bicall(
        self,
        input: str,
        promt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        id: Optional[str] = None,
    ) -> Union["Parameter", ReActOutput]:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        # initialize step_history
        step_history = None
        if self.training:
            step_history = Parameter(
                data=[],
                param_type=ParameterType.INPUT,
                name="step_history",
                requires_opt=True,
            )
        else:
            step_history = []

        # set up the prompts
        prompt_kwargs = {
            **promt_kwargs,
            "input_str": input,
        }

        printc(f"input_query: {input}", color="red")
        for i in range(self.max_steps):
            step = i + 1
            try:
                step_history = self._run_one_step(
                    step, prompt_kwargs, model_kwargs, id, step_history
                )

                if self._check_last_step(step_history):
                    break

            except Exception as e:
                log.error(f"Error running step {step}: {e}")

        answer = self._get_answer(step_history)
        if self.training:
            return answer
        # wrap the output
        output = ReActOutput(step_history=step_history, id=id, answer=answer)
        return output

    def _extra_repr(self) -> str:
        s = f"max_steps={self.max_steps}, add_llm_as_fallback={self.add_llm_as_fallback}, "
        return s
