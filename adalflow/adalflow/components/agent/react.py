"""Implementation and optimization of React agent.
Note: This will be deprecated soon, use Agent + Runner instead.
"""

from typing import List, Union, Callable, Optional, Any, Dict, TypeVar, Type
from dataclasses import dataclass, field
from adalflow.core.base_data_class import DataClass
import logging
import traceback


from adalflow.core.generator import Generator
from adalflow.optim.grad_component import GradComponent
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.core.func_tool import FunctionTool, AsyncCallable
from adalflow.core.tool_manager import ToolManager
from adalflow.core.component import Component
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.core.types import (
    StepOutput,
    GeneratorOutput,
    Function,
    FunctionOutput,
)
from adalflow.optim.grad_component import fun_to_grad_component
from adalflow.core.model_client import ModelClient
from adalflow.utils.logger import printc
from adalflow.core.prompt_builder import Prompt


log = logging.getLogger(__name__)
T = TypeVar("T")

__all__ = [
    "DEFAULT_REACT_AGENT_SYSTEM_PROMPT",
    "ReActAgent",
]


default_role_desc = """You are an excellent task planner."""
# Ideal answer for agent
# 1. *complete answer* when the answer is short
# 2. *concise conclusion* when the answer is directly displayed in a tool or written in a file
# 3. *structured data* when the answer needs to be structured.
react_agent_task_desc = r"""
<START_OF_TASK_SPEC>
{{role_desc}}

Answer the input query using the tools provided below with maximum accuracy.

Each step you will read the previous thought, Action(name, kwargs), and Observation(execution result of the action) and then provide the next Thought and Action.

Follow function docstring to best call the tool.
- For simple queries: Directly call the ``finish`` action and provide the answer.
- For complex queries:
    - Step 1: Read the user query and divide it into multisteps. Start with the first tool/subquery.
    - Call one tool at a time to solve each subquery/subquestion. \
    - At step 'finish', do a concise conclusion without repeat previous observation/output.
REMEMBER:
- Action MUST call one of the tools. It CANNOT be empty.
- You will ALWAYS END WITH 'finish' tool to finish the task directly with answer or failure message.
<END_OF_TASK_SPEC>
"""

# - In this case, you are working as a multi-hop retriever and your answer in finish MUST be verbatim short factoid responses from retrieved context.
# - Answer with only the exact answer phrase, not a full sentence or paragraph.

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<START_OF_SYSTEM_PROMPT>
{{task_desc}}
- You cant use more than {{max_steps}} steps. At the {{max_steps}}th current step, must finish with answer.

{# Tools #}
{% if tools %}
<START_OF_TOOLS>
Tools and instructions:
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
<END_OF_TOOLS>
{% endif %}
{# Context Variables #}
{% if context_variables is not none %}
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
{% if examples %}
<START_OF_EXAMPLES>
Examples:
{% for example in examples %}
{{example}}
------------------------
{% endfor %}
<END_OF_EXAMPLES>
{% endif %}
{#contex#}
{% if context_str %}
-------------------------
<START_OF_CONTEXT>
{{context_str}}
<END_OF_CONTEXT>
{% endif %}
<END_OF_SYSTEM_PROMPT>
-----------------
<START_OF_USER_PROMPT>
{# chat history #}
{% if chat_history_str %}
<START_OF_CHAT_HISTORY>
{{chat_history_str}}
<END_OF_CHAT_HISTORY>
{% endif %}
{# user query #}
<START_OF_USER_QUERY>
Input query:
{{ input_str }}
<END_OF_USER_QUERY>
<START_OF_STEP_HISTORY>
Current Step/Max Step: {{step_history|length + 1}} / {{max_steps}}
{# Step History #}
{% if step_history %}
<STEPS>
Your previous steps:
{% for history in step_history %}
Step {{ loop.index }}.
{% if history.action %}
{% if history.action.thought %}
"thought": "{{history.action.thought}}",
{% endif %}
"name": "{{history.action.name}},
"kwargs": {{history.action.kwargs}}",
{% endif %}
"Observation": "{{history.observation}}"
------------------------
{% endfor %}
</STEPS>
{% endif %}
<END_OF_USER_PROMPT>
"""


class CombineStepHistory(GradComponent):
    def __init__(self):
        super().__init__(desc="Extract the final answer from the step history.")

    def call(
        self,
        step_history: List[StepOutput],
        react_agent_task_desc: str,  # skip connection
        id: Optional[str] = None,
    ) -> str:
        if not step_history:
            return ""
        answer = step_history[-1].observation
        return answer


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


class ReActAgent(Component):
    __doc__ = r"""ReActAgent uses generator as a planner that runs multiple and sequential functional call steps to generate the final response.
    The planner will generate a Function data class as action for each step that includes a "thought" field.
    The execution result is stored in the "observation" field of the StepOutput data class.
    If the execution failed, it will store the error message in the "observation" field so that we can auto-optimize it to correct the error.

    The final answer can be different in training and eval mode:
    - Training: the final answer will be
    Users need to set up:
    - tools: a list of tools to use to complete the task. Each tool is a function or a function tool.
    - max_steps: the maximum number of steps the agent can take to complete the task.
    - add_llm_as_fallback: a boolean to decide whether to use an additional LLM model as a fallback tool to answer the query.
    - model_client: the model client to use to generate the response.
    - model_kwargs: the model kwargs to use to generate the response.
    - template: the template to use to generate the prompt. Default is DEFAULT_REACT_AGENT_SYSTEM_PROMPT.
    - context_variables: the context variables to use in the prompt.
    - use_cache: a boolean to decide whether to use the cache to store the generated responses for the planner.
    - debug: a boolean to decide whether to print debug information.

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
        examples: Optional[Union[List[Function], List[str]]] = None,
        *,
        # the following arguments are mainly for the planner
        model_client: ModelClient,
        model_kwargs: Dict = {},
        # template for the planner
        template: Optional[str] = None,  # allow users to customize the template
        role_desc: Optional[str] = default_role_desc,
        context_variables: Optional[Dict] = None,  # context variables
        is_thinking_model: bool = False,
        use_cache: bool = True,
        debug: bool = False,
        answer_data_type: Type[T] = str,
    ):
        super().__init__()
        template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT

        self.max_steps = max_steps

        self.add_llm_as_fallback = add_llm_as_fallback
        self.context_variables = context_variables
        self.debug = debug
        self.use_cache = use_cache
        self.answer_data_type = answer_data_type

        processed_tools = self._init_tools(tools, model_client, model_kwargs)
        self.tool_manager: ToolManager = ToolManager(
            tools=processed_tools,
            additional_context={"context_variables": self.context_variables},
        )

        ouput_data_class = Function

        self._examples = examples or []

        self.is_thinking_model = is_thinking_model
        include_fields = ["name", "kwargs"]  # thinking model already outputs thinking
        if not is_thinking_model:
            include_fields.append("thought")

        output_parser = JsonOutputParser(
            data_class=ouput_data_class,
            examples=self._examples,
            return_data_class=True,
            include_fields=include_fields,
        )

        task_desc = Prompt(
            template=react_agent_task_desc, prompt_kwargs={"role_desc": role_desc}
        ).call()
        prompt_kwargs = {
            "tools": self.tool_manager.yaml_definitions,
            "output_format_str": output_parser.format_instructions(),
            "task_desc": Parameter(
                name="react_agent_task_desc",
                data=task_desc,
                role_desc="Task instruction for the agent to plan steps to solve a question in sequential and multi-steps to get the final answer. \
                For optimizer: you need to adapt this to the current specific task.",
                param_type=ParameterType.PROMPT,
                requires_opt=True,
            ),
            # "examples": Parameter(
            #     name="examples",
            #     data=None,
            #     role_desc="Examples for the ReAct agent.",
            #     param_type=ParameterType.DEMOS,
            #     requires_opt=True,
            # ),
            "context_variables": self.context_variables,
            "max_steps": self.max_steps,
        }
        self.planner = Generator(
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_parser,
            model_client=model_client,
            model_kwargs=model_kwargs,
            use_cache=use_cache,
        )

        # besides of form the final output, it adds a skip connection to the planner task description prompt
        self.combine_step_history = CombineStepHistory()

    def _init_tools(
        self,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        model_client: ModelClient,
        model_kwargs: Dict,
    ):
        r"""Initialize the tools. Using reference or else with (copy or deepcopy) we can not set the training/eval mode for each tool."""
        processed_tools = []
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
                log.error(f"Error using the llm_tool: {e}")
                print(f"Error using the llm_tool: {e}")

            return None

        # always add **kwargs for us to track the id, __doc__ as the predecessors.
        from adalflow.optim.grad_component import fun_to_grad_component

        @fun_to_grad_component(
            desc="Finish",
            doc_string=Parameter(
                data="Finish the task with the final answer in the kwargs.",
                param_type=ParameterType.PROMPT,
                requires_opt=True,
                role_desc="Instruct the agent on how to create the final answer from the step history.",
                name="doc_string",
            ),
        )
        def finish(answer: self.answer_data_type, **kwargs) -> str:
            return answer

        self._finish = FunctionTool(fn=finish)
        processed_tools = tools.copy()
        if self.add_llm_as_fallback:
            processed_tools.append(llm_tool)
        processed_tools.append(self._finish)
        return processed_tools

    def _execute_action(
        self,
        step_output: StepOutput,
        response: Union[Parameter, GeneratorOutput],
        id: Optional[str] = None,
    ) -> Optional[StepOutput]:
        """Parse the action string to a function call and execute it. Update the step_output with the result."""

        def handle_error(response: Parameter, e: str):

            @fun_to_grad_component()
            def set_step_output_with_error(
                step_output: StepOutput, error: str, response: Any
            ):
                """Set the step_output with error."""
                step_output.observation = f"error: {error} at {response.data}"
                return step_output

            response.add_successor_map_fn(
                successor=set_step_output_with_error, map_fn=lambda x: x.data
            )
            return set_step_output_with_error.forward(step_output, e, response)

        step = step_output.step

        if isinstance(response, Parameter):

            try:

                step_output.action = response.data.data
                # if thinking exists, then add it to the action.thought field
                if response.data.thinking:
                    step_output.action.thought = response.data.thinking

                if self.debug:
                    printc(
                        f"Step test train:  {step}: {step_output.action}", color="blue"
                    )

                if isinstance(response.data.data, Function):
                    response.data.data.kwargs.update({"id": id})

                result: Union[Parameter, str] = self.tool_manager(
                    expr_or_fun=response,
                    step="execute",
                    map_fn=lambda x: x.data.data,  # Function
                )

                if isinstance(result, str):

                    @fun_to_grad_component()
                    def set_step_output_with_error(step_output: StepOutput, data: str):
                        """Set the step_output with error."""
                        step_output.observation = f"Error {data} in executing action."

                        return step_output

                    response.add_successor_map_fn(
                        successor=set_step_output_with_error,
                        map_fn=lambda x: x.data.data,
                    )
                    step_output = set_step_output_with_error.forward(
                        step_output, response
                    )

                    return step_output

            except Exception as e:
                e = f"{e} Error executing action: {response.data}"
                return handle_error(response, e)

            try:

                step_output.step = step
                step_output.observation = result.data.output

                # update the execution result to the step_output to be consistent with the eval version
                result.data = step_output
                result.role_desc = "The result of the action execution, observation is the final answer"
                result.param_type = ParameterType.OUTPUT
                return result
            except Exception as e:
                e = f"{e} Error converting function output to step output: {result.data}"

                return handle_error(response, e)

        else:

            return self._execute_action_eval_mode(
                x=response,
                step_output=step_output,
                step=step,
                id=id,
            )

    def _execute_action_eval_mode(
        self,
        x: GeneratorOutput,
        step_output: StepOutput,
        step: int,
        id=None,
    ) -> StepOutput:
        """Execute the action and update the step_output."""
        if x.error or not x.data:
            error_msg = f"Error planning step {step}: {x.error}"
            step_output.observation = error_msg
            step_output.action = None
            log.error(error_msg)
            return step_output

        else:
            try:
                fun_expr: Function = x.data
                if x.thinking:
                    fun_expr.thought = x.thinking

                step_output.action = fun_expr
                # # add id to the function
                fun_expr.kwargs.update({"id": id})

                if step_output and step_output.action:

                    # result: FunctionOutput = self.tool_manager(
                    #     expr_or_fun=x.data,  # Function
                    #     step="execute",
                    # )

                    result: FunctionOutput = self.tool_manager.execute_func(
                        func=fun_expr
                    )

                    step_output.observation = result.output
                    if self.debug:
                        printc(f"Step {step}: \n{step_output}\n_______\n", color="blue")
                    return step_output
                else:
                    if self.debug:
                        printc(f"Failed to parse response for step {step}", color="red")
                    log.error(f"Failed to parse response for step {step}")
                    return step_output
            except Exception as e:
                error_msg = f"Error parsing response for step {step}: {e}"
                step_output.observation = error_msg
                log.error(error_msg)
                if self.debug:
                    printc(error_msg, color="red")
                return step_output

    def _run_one_step(
        self,
        step: int,
        prompt_kwargs: Dict,
        model_kwargs: Dict,
        id: Optional[str] = None,
        step_history: List[StepOutput] = [],
    ) -> Union[Parameter, StepOutput]:
        """Run one step of the agent. Plan and execute the action for the step.
        Need to deal with both train and eval mode on the self.planner.
        """
        if self.debug:
            printc(f"step: {step}", color="yellow")

        step_history_value = []
        for step_output in step_history:
            if isinstance(step_output, Parameter):
                step_history_value.append(step_output.data)
            else:
                step_history_value.append(step_output)

        prompt_kwargs["step_history"] = step_history_value

        for data in step_history_value:
            if not data:
                raise ValueError(
                    f"Expected StepOutput, but got {type(data)}, all steps: {step_history_value}"
                )
            if not isinstance(data, StepOutput):
                raise ValueError(
                    f"Expected StepOutput, but got {type(data)}, all steps: {step_history_value}"
                )

        log.debug(
            f"Running step {step} with prompt: {self.planner.prompt(**prompt_kwargs)}"
        )
        try:

            response: Union[GeneratorOutput, Parameter] = self.planner(
                prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs, id=id
            )
            prompt_str = self.planner.get_prompt(**prompt_kwargs)
            printc(f"Prompt: {prompt_str}", color="yellow")

        except Exception as e:
            error_msg = f"Error happened in planner response at step {step}: {e}.\n"
            error_msg += (
                f"Prompt kwargs: {prompt_kwargs}\nModel kwargs: {model_kwargs}\n"
            )
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)

        step_output: StepOutput = StepOutput(step=step)

        try:

            if self.training and isinstance(response, Parameter):

                if not isinstance(response.data, GeneratorOutput):
                    raise ValueError(
                        f"Expected GeneratorOutput, but got {type(response.data)}, value: {response.data}"
                    )
                # Detect planner parsing errors to FunctionExpression so that the prompt can be trained to self-correct
                if not isinstance(response.data.data, Function):

                    @fun_to_grad_component()
                    def set_step_output_with_error(
                        step_output: StepOutput, data: GeneratorOutput
                    ):
                        """Set the step_output with error."""
                        step_output.observation = f"Error {data.error} in parsing response: {data.raw_response}, data type: {type(data.data)}"
                        return step_output

                    response.add_successor_map_fn(
                        successor=set_step_output_with_error,
                        map_fn=lambda x: x.data,
                    )
                    step_output = set_step_output_with_error.forward(
                        step_output, response
                    )

                else:

                    step_output: Parameter = self._execute_action(
                        step_output, response, id
                    )
                    if not isinstance(step_output, Parameter):
                        raise ValueError(
                            f"Expected Parameter, but got {type(step_output)}, value: {step_output}"
                        )
                if self.debug:
                    printc(f"step_output: {step_output.data}", color="red")
                if not isinstance(step_output, Parameter):
                    raise ValueError(
                        f"Ensure step_output to be Parameter at training mode. Got {type(step_output)}.\n\
                            Please check the observation for error details: {step_output}"
                    )

                return step_output

            else:

                step_output: StepOutput = self._execute_action(
                    step_output=step_output, response=response, id=id
                )
                if not step_output:
                    raise RuntimeError(
                        f"Error executing action at step {step}: {step_output}"
                    )

                if self.debug:
                    printc(f"step_output: {step_output}", color="red")
                return step_output
        except Exception as e:
            error_msg = f"Error during execution at step {step}: {e}.\n"
            error_msg += f"Step output: {step_output}\nResponse: {response}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)

    def _check_last_step(
        self, step_history: List[Union[StepOutput, Parameter]]
    ) -> bool:
        """Check if the last step is the finish step."""
        if not step_history:
            return True

        last_step: Union[StepOutput, Parameter] = step_history[-1]

        if isinstance(last_step, Parameter):
            last_step = last_step.data

        if (
            last_step
            and last_step.action
            and hasattr(last_step.action, "name")
            and last_step.action.name == "finish"
        ):
            return True
        return False

    def _get_answer(
        self, step_history: List[Union[StepOutput, Parameter]]
    ) -> Union[str, "Parameter"]:
        """Get the final answer from the step history.

        When in training mode, we pass the whole step_history to the backward engine to find the feedback
        """
        if not step_history:
            return None

        last_step: Union[StepOutput, Parameter] = step_history[-1]
        if isinstance(last_step, Parameter):

            answer = self.combine_step_history(
                step_history=step_history,
                id=last_step.data_id,
                react_agent_task_desc=self.planner.prompt_kwargs[
                    "react_agent_task_desc"
                ],
            )

            return answer
        else:
            from dataclasses import is_dataclass

            answer = last_step.observation
            if is_dataclass(self.answer_data_type):
                answer = self.answer_data_type.from_dict(answer)
            return answer

    def call(self, *args, **kwargs) -> ReActOutput:
        output = self.bicall(*args, **kwargs)
        if not isinstance(output, ReActOutput) or not output:
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
        input: str,  # open up to the external
        promt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        id: Optional[str] = None,
    ) -> Union["Parameter", ReActOutput]:
        r"""prompt_kwargs: additional prompt kwargs to either replace or add to the preset prompt kwargs."""
        # initialize step_history in both training and eval mode

        # set up the prompts
        prompt_kwargs = {
            **promt_kwargs,
            "input_str": input,
        }

        step_history: List[Union[StepOutput, Parameter]] = []
        if self.debug:
            printc(f"input_query: {input}", color="red")
        for i in range(self.max_steps):
            step = i + 1
            try:
                step_output = self._run_one_step(
                    step, prompt_kwargs, model_kwargs, id, step_history
                )
                if isinstance(step_output, Parameter):
                    step_output.data_id = id
                step_history.append(step_output)
                if self._check_last_step(step_history):
                    break
            except Exception as e:
                log.error(f"Error running step {step}: {e}")
                printc(f"Error running step {step}: {e}", color="red")
                raise e  # the only place to raise the error for debugging. In normal cases, the agent should not raise an error.

        answer = self._get_answer(step_history)
        if self.training:
            return answer
        # wrap the output
        output = ReActOutput(step_history=step_history, id=id, answer=answer)
        if self.debug:
            printc(f"answer: {output}", color="yellow")

        return output

    def _extra_repr(self) -> str:
        s = f"max_steps={self.max_steps}, add_llm_as_fallback={self.add_llm_as_fallback}, "
        return s


if __name__ == "__main__":
    from adalflow.components.model_client import OpenAIClient
    from adalflow.utils import setup_env
    from adalflow.core.func_tool import FunctionTool

    setup_env()

    class App(Component):
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
                tools=[FunctionTool(llm_as_tool)],
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

    # print(OutputParameter.__mro__)

    app = App()
    app.train()
    output = app("I want to multiply 3 and 4.", id="123")
    # print(output)
    printc(output, color="yellow")
    output.draw_graph()
