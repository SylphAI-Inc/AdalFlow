from typing import (
    Dict,
    List,
    Optional,
    Type,
    Any,
    TypeVar,
    Union,
    Callable,
)
from adalflow.core.func_tool import FunctionTool, AsyncCallable

from adalflow.core.component import Component
from adalflow.core.model_client import ModelClient
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.core.prompt_builder import Prompt
from adalflow.core.types import GeneratorOutput, ModelType, Function
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.utils import printc


import logging


from adalflow.components.agent.react import (
    DEFAULT_REACT_AGENT_SYSTEM_PROMPT,
    react_agent_task_desc,
)


__all__ = ["Agent"]

T = TypeVar("T")

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 10
DEFAULT_ROLE_DESC = """You are an excellent task planner."""


# NOTE: our function call supports component and its method

# Tools and instructions:
# 1.
# class_instance: component
# func_name: __call__
# func_desc: '__call__(command: ''str'', **kwargs)

#   Belongs to class: PermissionTool

#   Docstring: Run bash command with permission check.

#   '


# the context will wrap the whole component
def create_default_tool_manager(
    # Tool manager parameters
    tools: Optional[List[Any]] = None,
    context_variables: Optional[Dict] = None,  # context variables
    # set the llm tool
    add_llm_as_fallback: Optional[bool] = True,
    model_client: Optional[ModelClient] = None,
    model_kwargs: Optional[Dict[str, Any]] = {},
    answer_data_type: Optional[Type[T]] = str,  # the data type of the final answer
    **kwargs,
) -> ToolManager:
    """Create a default tool manager with the given tools, context variables, and add_llm_as_fallback."""

    # raise error when there is no model_client
    if add_llm_as_fallback and (not model_client or not model_kwargs):
        raise ValueError("model_client and model_kwargs are required")

    def _init_tools(
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        model_client: ModelClient,
        model_kwargs: Dict,
    ):
        r"""Initialize the tools. Using reference or else with (copy or deepcopy) we can not set the training/eval mode for each tool."""
        processed_tools = []
        _additional_llm_tool = (
            Generator(model_client=model_client, model_kwargs=model_kwargs)
            if add_llm_as_fallback
            else None
        )

        def llm_tool(input_str: str) -> str:
            """I answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple."""
            try:
                output: GeneratorOutput = _additional_llm_tool(
                    prompt_kwargs={"input_str": input_str}
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
                data="""
                Finish the task with the final conclusion in the answer.
                These rules MUST BE FOLLOWED:
                1. If the specified type of the answer is a Python built-in type, the answer MUST be an object of that specific builtin type.
                2. If it is not, pass in the answer as a string but with these rules:
                    - This string will be directly parsed by the caller by using AST.literal_eval, so it must be deserializable using AST.literal_eval.
                    - Once the string is deserialized, it should be able to be parsed by the caller into the provided type.
                """,
                param_type=ParameterType.PROMPT,
                requires_opt=True,
                role_desc="Instruct the agent on how to create the final answer from the step history.",
                name="doc_string",
            ),
        )
        def finish(answer: answer_data_type, **kwargs) -> Union[str, answer_data_type]:
            # returns a string that is an AST for the answer_data_type
            log.info(f"answer: {answer}, type: {type(answer)}")
            # answer will be passed as a dict
            return answer

        _finish = FunctionTool(fn=finish)

        print(_finish.definition)
        processed_tools = tools.copy() if tools else []
        if add_llm_as_fallback:
            processed_tools.append(llm_tool)
        processed_tools.append(_finish)
        return processed_tools

    # 1. create default ToolManager

    processed_tools = _init_tools(tools, model_client, model_kwargs)

    tool_manager = ToolManager(
        tools=processed_tools,
        additional_context={
            "context_variables": context_variables
        },  # TODO: optimize this
    )

    return tool_manager


def create_default_planner(
    # Tool manager parameters
    tool_manager: ToolManager,
    # Generator parameters
    model_client: Optional[ModelClient] = None,
    model_kwargs: Optional[Dict[str, Any]] = {},
    model_type: Optional[ModelType] = ModelType.LLM,
    template: Optional[
        str
    ] = None,  # allow users to update the template but cant delete any parameters
    role_desc: Optional[Union[str, Prompt]] = None,
    cache_path: Optional[str] = None,
    use_cache: Optional[bool] = False,
    # default agent parameters
    max_steps: Optional[int] = 10,
    is_thinking_model: Optional[bool] = False,
    **kwargs,
) -> Generator:
    """Create a default planner with the given model client, model kwargs, template, task desc, cache path, use cache, max steps."""

    if not model_client:
        raise ValueError("model_client and model_kwargs are required")

    template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT
    role_desc = role_desc or DEFAULT_ROLE_DESC

    # define the parser for the intermediate step, which is a Function class
    ouput_data_class = Function
    if is_thinking_model:
        # skip the CoT field
        include_fields = [
            "name",
            "kwargs",
        ]
    else:
        include_fields = ["thought", "name", "kwargs"]
    output_parser = JsonOutputParser(
        data_class=ouput_data_class,
        examples=None,
        # examples=self._examples, # TODO: add examples
        return_data_class=True,
        include_fields=include_fields,
    )

    task_desc = Prompt(
        template=react_agent_task_desc,
        prompt_kwargs={"role_desc": role_desc},
    )

    prompt_kwargs = {
        "tools": tool_manager.yaml_definitions,
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
        "context_variables": tool_manager.context_variables[
            "context_variables"
        ],  # TODO: make it more clear
        "max_steps": max_steps,  # move to the 2nd step
        "step_history": [],
    }

    # 3. create default Generator
    planner = Generator(
        model_client=model_client,
        model_kwargs=model_kwargs,
        model_type=model_type,
        template=template,
        prompt_kwargs=prompt_kwargs,
        output_processors=output_parser,
        name="agent planner",
        cache_path=cache_path,
        use_cache=use_cache,
    )

    printc(f"planner use cache: {planner.use_cache}")
    printc(f"planner cache path: {planner.cache_path}")

    return planner


class Agent(Component):
    """
    An agent is a high-level component that holds (1) a generator as task plannaer (calling tools) and (2) a tool manager to manage tools.

    It comes with default prompt template that instructs LLM (1) agentic task description (2) template on adding definitions of tools
    (3) arguments to fill in history.

    Additionally, it comes with two helper tools:
    1. finish: to finish the task
    2. additional_llm_tool: to answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple.

    Attributes:
        name (str): Name of the agent
        tool_manager (ToolManager): Stores and manages tools
        generator (Generator): Handles text generation with a language model
        the output_processors must return the type StepOutput
    """

    def __init__(
        self,
        name: str,
        # pass this if using default agent config
        tools: Optional[List[Any]] = None,
        context_variables: Optional[Dict] = None,  # context variables
        add_llm_as_fallback: Optional[bool] = True,
        # Generator parameters
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = {},
        model_type: Optional[
            ModelType
        ] = ModelType.LLM,  # by default use LLM reasoning model
        template: Optional[
            str
        ] = None,  # allow users to update the template but cant delete any parameters
        role_desc: Optional[Union[str, Prompt]] = None, # support both str and prompte template
        cache_path: Optional[str] = None,
        use_cache: Optional[bool] = True,
        # default agent parameters
        answer_data_type: Optional[Type[T]] = str,  # the data type of the final answer
        max_steps: Optional[int] = 10,
        is_thinking_model: Optional[bool] = False,  # support thinking model in agent
        # for fully customize the agent
        tool_manager: Optional[
            ToolManager
        ] = None,  # pass this if using custom tool manager
        planner: Optional[Generator] = None,  # pass this if using custom planner
        **kwargs,
    ):
        """Initialize agent with required components.

        The agent internal classes have three main parts:
        (1) name
        (2) planner
        (3) tool_manager

        These three parts are essential for

        Args:
            name: Unique identifier for the agent

            # Tool manager parameters
            tools: List of tools to be managed by the ToolManager
            additional_context: Additional context to be added to the tool manager

            # Generator parameters
            model_client: Initialized ModelClient instance for the generator
            model_kwargs: Model configuration parameters (e.g., model name, temperature)
            template: Optional template for the generator prompt
            prompt_kwargs: Additional prompt parameters (The prompt_kwargs must incldue details on )
            output_processors: Optional output processors for the generator (should haeve Function as the output type)
            cache_path: Path for caching generator outputs
            use_cache: Whether to use caching for generator outputs
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        # name the agent
        self.name = name

        # planner or model_client exists

        self.tool_manager = tool_manager or create_default_tool_manager(
            tools=tools,
            context_variables=context_variables,
            add_llm_as_fallback=add_llm_as_fallback,
            model_client=model_client,
            model_kwargs=model_kwargs,
            answer_data_type=answer_data_type,
        )

        self.planner = planner or create_default_planner(
            tool_manager=self.tool_manager,
            model_client=model_client,
            model_kwargs=model_kwargs,
            model_type=model_type,
            template=template,
            role_desc=role_desc,
            cache_path=cache_path,
            use_cache=use_cache,
            max_steps=max_steps,
            is_thinking_model=is_thinking_model,
        )
        self.answer_data_type = answer_data_type  # save the final answer data type for the runner to communicate
        self.max_steps = max_steps
        self.is_thinking_model = is_thinking_model

        # check
        if not self.tool_manager:
            raise ValueError("Tool manager must be provided to the agent.")
        if not self.planner:
            raise ValueError("Planner must be provided to the agent. ")
        if self.max_steps < 1:
            raise ValueError("Max steps must be greater than 0.")

    def is_training(self) -> bool:
        return self.planner.training

    def get_prompt(self, **kwargs) -> str:
        """Get formatted prompt using generator's prompt template.

        Args:
            **kwargs: Additional arguments to pass to the generator's get_prompt

        Returns:
            Formatted prompt string
        """
        return self.planner.get_prompt(**kwargs)
