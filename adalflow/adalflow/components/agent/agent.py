"""Agent component for building conversational AI agents with tool integration."""

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
from adalflow.core.functional import get_type_schema
from adalflow.core.component import Component
from adalflow.core.model_client import ModelClient
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.core.prompt_builder import Prompt
from adalflow.core.types import GeneratorOutput, ModelType, Function
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.utils import printc

from adalflow.components.agent.prompts import (
    DEFAULT_ADALFLOW_AGENT_SYSTEM_PROMPT,
    adalflow_agent_task_desc,
)

import logging


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

        processed_tools = tools.copy() if tools else []
        if add_llm_as_fallback and _additional_llm_tool:
            processed_tools.append(llm_tool)
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
    answer_data_type: Optional[Type[T]] = str,
    **kwargs,
) -> Generator:
    """Create a default planner with the given model client, model kwargs, template, task desc, cache path, use cache, max steps."""

    if not model_client:
        raise ValueError("model_client and model_kwargs are required")

    template = template or DEFAULT_ADALFLOW_AGENT_SYSTEM_PROMPT
    role_desc = role_desc or DEFAULT_ROLE_DESC

    # define the parser for the intermediate step, which is a Function class
    ouput_data_class = Function
    if is_thinking_model:
        # skip the CoT field
        include_fields = ["name", "kwargs", "_is_answer_final", "_answer"]
    else:
        include_fields = ["thought", "name", "kwargs", "_is_answer_final", "_answer"]
    output_parser = JsonOutputParser(
        data_class=ouput_data_class,
        examples=None,
        # examples=self._examples, # TODO: add examples
        return_data_class=True,
        include_fields=include_fields,
    )

    task_desc = Prompt(
        template=adalflow_agent_task_desc,
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
        # "task_desc": task_desc,
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
        "answer_type_schema": get_type_schema(answer_data_type),
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

    log.debug(f"planner use cache: {planner.use_cache}")
    log.debug(f"planner cache path: {planner.cache_path}")

    return planner


class Agent(Component):
    """A high-level agentic component that orchestrates AI planning and tool execution.

    The Agent combines a Generator-based planner for task decomposition with a ToolManager
    for function calling. It uses a ReAct (Reasoning and Acting) architecture to iteratively
    plan steps and execute tools to solve complex tasks.

    The Agent comes with default prompt templates for agentic reasoning, automatic tool
    definition integration, and step history tracking. It includes built-in helper tools:
    - llm_tool: Fallback tool using LLM world knowledge for simple queries

    Architecture:
        Agent contains two main components:
        1. Planner (Generator): Plans and reasons about next actions using an LLM
        2. ToolManager: Manages and executes available tools/functions

    Attributes:
        name (str): Unique identifier for the agent instance
        tool_manager (ToolManager): Manages available tools and their execution
        planner (Generator): LLM-based planner for task decomposition and reasoning
        answer_data_type (Type): Expected type for the final answer output
        max_steps (int): Maximum number of planning steps allowed
        is_thinking_model (bool): Whether the underlying model supports chain-of-thought
    """

    def __init__(
        self,
        name: str,
        # pass this if using default agent config
        tools: Optional[List[Any]] = None,
        context_variables: Optional[Dict] = None,  # context variables
        add_llm_as_fallback: Optional[bool] = False,
        # Generator parameters
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = {},
        model_type: Optional[
            ModelType
        ] = ModelType.LLM,  # by default use LLM reasoning model
        template: Optional[
            str
        ] = None,  # allow users to update the template but cant delete any parameters
        role_desc: Optional[
            Union[str, Prompt]
        ] = None,  # support both str and prompte template
        cache_path: Optional[str] = None,
        use_cache: Optional[bool] = True,
        # default agent parameters
        answer_data_type: Optional[Type[T]] = str,  # the data type of the final answer
        max_steps: Optional[int] = 10,
        is_thinking_model: Optional[bool] = False,  # when thinking model turned on, it disables the CoT field in the output
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

        # save the following parameters to recreate the planner
        self.model_client = model_client
        self.model_kwargs = model_kwargs
        self.model_type = model_type

        # set the template
        self.template = template or DEFAULT_ADALFLOW_AGENT_SYSTEM_PROMPT
        self.role_desc = role_desc or DEFAULT_ROLE_DESC

        self.cache_path = cache_path
        self.use_cache = use_cache

        # set the default max steps
        self.max_steps = max_steps or DEFAULT_MAX_STEPS
        self.is_thinking_model = is_thinking_model

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
            answer_data_type=answer_data_type,
        )
        self.answer_data_type = answer_data_type  # save the final answer data type for the runner to communicate

        # check
        if not self.tool_manager:
            raise ValueError("Tool manager must be provided to the agent.")
        if not self.planner:
            raise ValueError("Planner must be provided to the agent. ")
        if self.max_steps < 1:
            raise ValueError("Max steps must be greater than 0.")
        
    def flip_thinking_model(self):
        """Toggle the thinking model state."""
        self.is_thinking_model = not self.is_thinking_model
        log.debug(f"Thinking model is now {'enabled' if self.is_thinking_model else 'disabled'}.")

        # we have to recrate the planner
        self.planner = create_default_planner(
            tool_manager=self.tool_manager,
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            model_type=self.model_type,
            template=self.template,
            role_desc=self.role_desc,
            cache_path=self.cache_path,
            use_cache=self.use_cache,
            max_steps=self.max_steps,
            is_thinking_model=self.is_thinking_model,
            answer_data_type=self.answer_data_type,
        )


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
