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
{# Context Variables #}
{% if context_variables %}
<START_OF_CONTEXT>
You have access to context_variables with the following keys:
{% for key, value in context_variables.items() %}
{{ key }}
------------------------
{% endfor %}
You can either pass context_variables or context_variables['key'] to the tools depending on the tool's requirements.
<END_OF_CONTEXT>
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
        self.name = "AppendStepHistory"
        self._component_desc = "Append the step_output to the step_history."

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


# class ExecuteAction(GradComponent):
#     def __init__(self):
#         super().__init__()
#         self.name = "ExecuteAction"
#         self._component_desc = "Execute the action and output the new step_output."

#     def call(
#         self,
#         response: GeneratorOutput,
#         step_output: StepOutput,
#         execute_action: Callable,
#         id: Optional[str] = None,
#     ) -> StepOutput:
#         """Parse the action string to a function call and execute it. Update the action_step with the result."""
#         step = step_output.step
#         output = execute_action_fn(response, step_output, step, execute_action, id)
#         if isinstance(output, Parameter):
#             output = output.full_response
#         return output


class FunctionOutputToStepOutput(GradComponent):
    def __init__(self):
        super().__init__()
        self.name = "FunctionOutputToStepOutput"
        self._component_desc = "Convert the FunctionOutput to StepOutput."

    def call(self, output: FunctionOutput, step_output: StepOutput) -> StepOutput:
        """Convert the FunctionOutput to StepOutput."""

        temp_result = output.output
        if isinstance(temp_result, Parameter):
            step_output.observation = temp_result.data
        else:
            step_output.observation = temp_result
        return step_output
        # step_output = StepOutput(step=step)
        # step_output.observation = output.output
        # return step_output


# TODO: make execute_action_fn to a GradComponent to enable the training of the tools too.
def execute_action_fn(
    x: GeneratorOutput, step_output: StepOutput, step: int, execute_action: Any, id=None
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
                step_output = execute_action(step_output, id)
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
        context_variables: Optional[Dict] = None,  # context variables
    ):
        super().__init__()
        template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT

        self.max_steps = max_steps

        self.add_llm_as_fallback = add_llm_as_fallback
        self.context_variables = context_variables

        tools = self._init_tools(tools, model_client, model_kwargs)
        self.tool_manager: ToolManager = ToolManager(
            tools=tools,
            additional_context={"context_variables": self.context_variables},
        )

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
            "context_variables": self.context_variables,
        }
        self.planner = Generator(
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_parser,
            model_client=model_client,
            model_kwargs=model_kwargs,
            use_cache=True,
        )

        # added this component to the computation graph
        self.append_step_history = AppendStepHistory()
        # self.execute_action = ExecuteAction()
        self.function_output_to_step_output = FunctionOutputToStepOutput()

    def _init_tools(
        self,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        model_client: ModelClient,
        model_kwargs: Dict,
    ):
        r"""Initialize the tools."""
        try:
            tools = [deepcopy(tool) for tool in tools]
        except Exception:
            from copy import copy

            tools = [copy(tool) for tool in tools]
        _additional_llm_tool = (
            Generator(model_client=model_client, model_kwargs=model_kwargs)
            if self.add_llm_as_fallback
            else None
        )

        def llm_tool(input: str, **kwargs) -> str:
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

        def finish(answer: str, **kwargs) -> str:
            """Finish the task with answer."""
            return answer

        self._finish = finish

        if self.add_llm_as_fallback:
            tools.append(llm_tool)
        tools.append(finish)
        return tools

    def _execute_action(
        self,
        action_step: StepOutput,
        response: Union[Parameter, GeneratorOutput],
        id: Optional[str] = None,
    ) -> Optional[StepOutput]:
        """Parse the action string to a function call and execute it. Update the action_step with the result."""
        # extract the action from the response

        if isinstance(response, Parameter):

            class ActionStrToStepOutput(GradComponent):
                def __init__(self):
                    super().__init__()
                    self.name = "ActionStrToStepOutput"
                    self._component_desc = "Convert the action string to StepOutput."

                def call(
                    self,
                    action_str: FunctionExpression,
                    step: int,
                    result: Union[FunctionOutput, Parameter],
                    func: Function,
                ) -> StepOutput:
                    """Convert the action string to StepOutput."""
                    step_output = StepOutput(step=step)
                    step_output.action = action_str
                    step_output.function = func
                    # printc(f"result: {result}", color="blue")
                    result = result.data if isinstance(result, Parameter) else result
                    if isinstance(result, FunctionOutput):
                        step_output.observation = (
                            result.output.data
                            if isinstance(result.output, Parameter)
                            else result.output
                        )

                    return step_output

            tmp_action_str_to_step_output = ActionStrToStepOutput()

            # printc(f"response: {response}", color="yellow")
            # TO FunctionExpression
            response.add_successor_map_fn(
                successor=self.tool_manager, map_fn=lambda x: x.full_response
            )

            func: Union[Function, Parameter] = self.tool_manager(
                expr_or_fun=response, step="parse"
            )
            printc(f"tool_manager: {self.tool_manager.training}", color="red")
            if not isinstance(func, Parameter):
                raise ValueError(f"Expected Parameter, but got {type(func)}: {func}")
            printc(f"func: {func}", color="yellow")
            # replace the id
            if isinstance(func, Parameter):
                func.data.kwargs["id"] = id

                func.add_successor_map_fn(self.tool_manager, lambda x: x.data)

            result: Parameter = self.tool_manager(expr_or_fun=func, step="execute")
            printc(f"result: {result}", color="red")
            result.add_successor_map_fn(
                successor=tmp_action_str_to_step_output, map_fn=lambda x: x.data
            )
            action_step = tmp_action_str_to_step_output.forward(
                action_str=response.data,
                step=action_step.step,
                result=result,
                func=func,
            )

            return action_step
            # except Exception as e:
            #     log.error(f"Error executing {response}: {e}")
            #     # pass the error as observation so that the agent can continue and correct the error in the next step
            #     # action_step.observation = f"Error executing {response}: {e}"
            #     # return action_step
            #     raise e
        else:
            # normal pass
            fun: Function = self.tool_manager(expr_or_fun=response.data, step="parse")
            action_step.function = fun
            result: FunctionOutput = self.tool_manager(expr_or_fun=fun, step="execute")
            action_step.observation = result.output
            return action_step

    def _run_one_step(
        self,
        step: int,
        prompt_kwargs: Dict,
        model_kwargs: Dict,
        id: Optional[str] = None,
        step_history: Union["Parameter", List[str]] = None,
    ) -> Union[List[StepOutput], Parameter]:
        """Run one step of the agent. Plan and execute the action for the step.
        Need to deal with both train and eval mode on the self.planner.
        """
        printc("start running one step", color="yellow")

        prompt_kwargs["step_history"] = step_history
        printc(
            f"prompt_kwargs 1: {prompt_kwargs}, training: {self.planner.training}",
            color="yellow",
        )

        # prompt_str = self.planner.get_prompt(**prompt_kwargs)
        # return [StepOutput(step=step, action=None, observation="test")]

        log.debug(
            f"Running step {step} with prompt: {self.planner.prompt(**prompt_kwargs)}"
        )
        try:

            response: Union[GeneratorOutput, Parameter] = self.planner(
                prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs, id=id
            )
        except Exception as e:
            log.error(f"Error planning step {step}: {e}")
            return None

        # create a new step output
        step_output: StepOutput = StepOutput(step=step)

        # connecting two generators in the computation graph, it will set up self.step_history
        if isinstance(response, Parameter):
            printc(f"response: {response}", color="yellow")

            step_output: Parameter = self._execute_action(step_output, response, id)

            printc(f"step_output: {step_output}", color="red")
            step_output.add_successor_map_fn(
                successor=self.append_step_history, map_fn=lambda x: x.data
            )
            step_history.add_successor_map_fn(
                successor=self.append_step_history, map_fn=lambda x: x.data
            )

            step_history = self.append_step_history.forward(step_output, step_history)
            # connect step_history to the next planner
            step_history.add_successor_map_fn(
                successor=self.planner, map_fn=lambda x: x.data
            )
            # convert step history back to data
            printc(f"step_history: {step_history.data}", color="yellow")
            return step_history

        else:

            step_output = self._execute_action(
                action_step=step_output, response=response, id=id
            )
            printc(f"step_output: {step_output}", color="red")
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
            # try:
            printc(f"step_history: {step_history}", color="yellow")
            step_history_data = step_history.data
            printc(f"step_history: {step_history}", color="yellow")
            last_step = step_history_data[-1]

            # except Exception as e:
            #     log.error(f"Error getting data from Parameter: {e}")
            #     return False
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
                printc(f"step_history: {step_history}", color="yellow")
                return step_history

            except Exception as e:
                log.error(f"Error getting data from Parameter: {e}")
                return None
        else:
            last_step = step_history[-1]
            printc(f"last_step: {last_step}", color="yellow")

            return last_step.observation

    def call(self, *args, **kwargs) -> ReActOutput:
        output = self.bicall(*args, **kwargs)
        if not isinstance(output, ReActOutput):
            raise ValueError(f"Expected ReActOutput, but got {type(output)}")
        return output

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
            # try:
            step_history = self._run_one_step(
                step, prompt_kwargs, model_kwargs, id, step_history
            )

            if self._check_last_step(step_history):
                break

            # except Exception as e:
            #     log.error(f"Error running step {step}: {e}")
            # step_output = StepOutput(
            #     step=step, observation=f"Error: {e}", action=None
            # )
            # step_history.append(step_output)

        answer = self._get_answer(step_history)
        if self.training:
            return answer
        # wrap the output
        output = ReActOutput(step_history=step_history, id=id, answer=answer)
        # printc(f"output: {output}", color="yellow")
        return output

    def _extra_repr(self) -> str:
        s = f"max_steps={self.max_steps}, add_llm_as_fallback={self.add_llm_as_fallback}, "
        return s


if __name__ == "__main__":
    from adalflow.components.model_client import OpenAIClient
    from adalflow.utils import setup_env
    from adalflow.core.func_tool import FunctionTool

    setup_env()

    class App(GradComponent):
        def __init__(self):
            super().__init__()
            self.llm_tool = Generator(
                model_client=OpenAIClient(),
                model_kwargs={"model": "gpt-3.5-turbo"},
            )

            def llm_as_tool(input: str, id: Optional[str] = None) -> str:
                """Used as a calculator tool."""
                printc(f"llm_as_tool: {input}", color="yellow")

                return self.llm_tool(prompt_kwargs={"input_str": input}, id=id)

            self.react_agent = ReActAgent(
                tools=[FunctionTool(llm_as_tool, component=self.llm_tool)],
                max_steps=2,
                add_llm_as_fallback=False,
                model_client=OpenAIClient(),
                model_kwargs={"model": "gpt-3.5-turbo"},
            )

        def call(self, input: str, id: Optional[str] = None) -> Union[str, "Parameter"]:
            return self.react_agent(input, id=id)

        def forward(
            self, input: str, id: Optional[str] = None
        ) -> Union[str, "Parameter"]:
            return self.react_agent(input, id=id)

    app = App()
    app.train()
    output = app("I want to multiply 3 and 4.", id="123")
    print(output)
    output.draw_graph()