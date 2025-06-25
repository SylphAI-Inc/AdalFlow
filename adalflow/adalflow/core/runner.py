from adalflow.core.component import Component
from adalflow.core.agent import Agent

from typing import (
    Generator as GeneratorType,
    Dict,
    Optional,
    List,
    Any,
    TypeVar,
    Union,
    Tuple,
)
from adalflow.core.types import GeneratorOutput, FunctionOutput, StepOutput
from adalflow.optim.parameter import Parameter
from adalflow.core.types import Function
from pydantic import BaseModel
import logging
import json


__all__ = ["Runner"]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)  # Changed to use Pydantic BaseModel


# @dataclass
# class RunnerConfig:
#     """Configuration for the Runner class.

#     Attributes:
#         output_type: Optional Pydantic data class type
#         stream_parser: Optional stream parser
#     """

#     output_type: Optional[Type[T]] = None
#     stream_parser: Optional["StreamParser"] = None


class Runner(Component):
    """A runner class that executes an Agent instance with multi-step execution.

    It internally maintains a planner LLM and an executor and adds a LLM call to the executor as a tool for the planner.

    The output to the planner agent  call is expected to be a Function object. The planner iterates through at most
    max_steps unless the planner sets the action to "finish" then the planner returns the final response.

    If the user optionally specifies the output_type then the Runner parses the Function object to the output_type.

    Attributes:
        planner (Agent): The agent instance to execute
        config (RunnerConfig): Configuration for the runner
        max_steps (int): Maximum number of steps to execute
    """

    def __init__(
        self,
        agent: Agent,
        # stream_parser: Optional["StreamParser"] = None,
        # output_type: Optional[Type[T]] = None,
        # max_steps: int = 10,
        **kwargs,
    ) -> None:
        """Initialize runner with an agent and configuration.

        Args:
            agent: The agent instance to execute
            stream_parser: Optional stream parser
            output_type: Optional Pydantic data class type
            max_steps: Maximum number of steps to execute
        """
        super().__init__(**kwargs)
        self.agent = agent
        # executor = executor if executor else self._init_executor()
        # self.executor = executor
        self.max_steps = agent.max_steps
        self.answer_data_type = agent.answer_data_type
        # self.config = RunnerConfig(
        #     stream_parser=stream_parser,
        #     output_type=output_type,
        # )
        self.step_history = []
        # add the llm call to the executor as a tool

    def _check_last_step(self, step: Function) -> bool:
        """Check if the last step is the finish step.

        Args:
            step_history: List of previous steps

        Returns:
            bool: True if the last step is a finish step
        """

        # assert (
        #     isinstance(step, Function),
        #     f"Expected Function, but got {type(step)}, value: {step}",
        # )

        # Check if it's a finish step
        if step.name == "finish":
            return True

        return False

    def _process_data(self, data: Dict[str, Any], id: Optional[str] = None) -> T:
        """Process the generator output data field and convert to the specified pydantic data class of output_type.

        Args:
            data: The data to process
            id: Optional identifier for the output

        Returns:
            str: The processed data as a string
        """
        if not self.answer_data_type:
            return data

        try:
            # expect data.observation to be a strict
            # Convert to Pydantic model

            if isinstance(data, str):
                try:
                    data = json.loads(data.replace("'", '"')) 
                    log.info(data)
                except json.JSONDecodeError:
                    log.error(f"Failed to parse string as JSON: {data}")
                    return f"Error: Invalid JSON string: {data}"

            # return only the 'properties key of the object and then load into the class 
            
            # TODO make more robust 
            def recursive_parse(data): 
                # Return primitive types as-is
                if isinstance(data, (str, int, float, bool, type(None))):
                    return data

                # Handle dictionaries
                if isinstance(data, dict):
                    if "properties" in data: 
                        return recursive_parse(data["properties"])
                    return {k: recursive_parse(v) for k, v in data.items()}

                # Handle other iterables (including lists, tuples, sets, etc.)
                from collections.abc import Iterable, Sequence
                if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
                    # For sequences (lists, tuples), preserve the type
                    if isinstance(data, Sequence):
                        return type(data)(recursive_parse(item) for item in data)
                    # For non-sequence iterables (sets, generators), convert to list
                    return [recursive_parse(item) for item in data]

                # Return as-is if not an iterable we handle
                return data

            data = recursive_parse(data)

            model_output = self.answer_data_type(**data)

            return model_output

        except Exception as e:
            log.error(f"Failed to parse output: {e}")
            return f"Error processing output: {str(e)}"

    def call(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # if some call use a different config
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Tuple[List[GeneratorOutput], Any]:
        """Execute the planner synchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request

        Returns:
            The generator output of type specified in self.answer_data_type
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        step_count = 0
        last_output = None

        # set maximum number of steps for the planner into the prompt 
        prompt_kwargs["max_steps"] = self.max_steps

        while step_count < self.max_steps:
            try:
                # Execute one step
                output = self.agent.planner.call(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )

                if not isinstance(output, GeneratorOutput):
                    raise ValueError(
                        f"Expected GeneratorOutput, but got {type(output)}, value: {output}"
                    )

                function = output.data

                # assert (
                #     isinstance(function, Function),
                #     f"Expected Function type, but got {type(function)}, value: {function}",
                # )

                function_results = self._tool_execute(function)
                last_output = function_results.output

                step_ouput: StepOutput = StepOutput(
                    step=step_count, function=function, observation=function_results.output
                )
                self.step_history.append(step_ouput)

                if self._check_last_step(function):
                    last_output = self._process_data(function_results.output)
                    break

                # Add function results to prompt for next step
                if "step_history" not in prompt_kwargs:
                    prompt_kwargs["step_history"] = []
                else:
                    # Format function results more clearly
                    prompt_kwargs["step_history"].append(step_ouput)
                # log.info(
                #     "The prompt with the prompt template is {}".format(
                #         self.agent.planner.get_prompt(**prompt_kwargs)
                #     )
                # )

                step_count += 1

            except Exception as e:
                log.error(f"Error in step {step_count}: {str(e)}")
                return f"Error in step {step_count}: {str(e)}"

        return self.step_history, last_output

    async def acall(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Tuple[List[GeneratorOutput], T]:
        """Execute the planner asynchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request

        Returns:
            Tuple containing:
                - List of step history (GeneratorOutput objects)
                - Final processed output
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        step_count = 0
        last_output = None

        while step_count < self.max_steps:
            try:
                # Execute one step asynchronously
                output = await self.agent.planner.acall(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )

                if not isinstance(output, GeneratorOutput):
                    raise ValueError(
                        f"Expected GeneratorOutput, but got {type(output)}, value: {output}"
                    )

                # planner generated response using generator's call (inference)
                function = output.data

                # assert (
                #     isinstance(function, Function),
                #     f"Expected Function type, but got {type(function)}, value: {function}",
                # )

                function_results = self._tool_execute(function)
                last_output = self._process_data(function_results.output)

                step_output: StepOutput = StepOutput(
                    step=step_count, function=function, observation=function_results.output
                )
                self.step_history.append(step_output)

                if self._check_last_step(function):
                    break

                # Add function results to prompt for next step
                if "step_history" not in prompt_kwargs:
                    prompt_kwargs["step_history"] = []
                else:
                    # Format function results more clearly
                    prompt_kwargs["step_history"].append(step_output)
                # log.info(
                #     "The prompt with the prompt template is {}".format(
                #         self.agent.planner.get_prompt(**prompt_kwargs)
                #     )
                # )

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                return self.step_history, error_msg

        return self.step_history, last_output

    def stream(
        self,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> GeneratorType[Any, None, None]:
        """
        Synchronously executes the planner output and stream results.
        Optionally parse and post-process each chunk.
        """
        # TODO replace Any type with StreamChunk type
        try:
            generator_output = self.call(user_query, current_objective, memory)

            if self.config.stream_parser:
                for chunk in generator_output.data:
                    yield self.config.stream_parser(chunk)
            else:
                log.error("StreamParser not specified")
                for chunk in generator_output.data:
                    yield chunk
                # TODO: need to define a StreamChunk type in library

        except Exception as e:
            log.error(f"Failed to stream generator output: {e}")
            yield generator_output
            # TODO: need to define a StreamChunk type in library

    # TODO implement async stream
    async def astream(
        self,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> GeneratorType[Any, None, None]:
        """
        Execute the planner asynchronously and stream results.
        Optionally parse and post-process each chunk.
        """
        # TODO replace Any type with StreamChunk type
        ...
        # This would require relying on the async_stream of the model_client instance of the generator and parsing that
        # using custom logic to buffer chunks and only stream when they complete a certain top-level field

    # def backward(
    #     self,
    #     response: Parameter,
    #     prompt_kwargs: Optional[Dict] = None,
    #     template: Optional[str] = None,
    #     backward_engine: Optional["Generator"] = None,
    #     id: Optional[str] = None,
    #     disable_backward_engine: bool = False,
    # ):
    #     """
    #     Run backward pass on the planner.
    #     Template is expected to be the template to guide the backward pass.
    #     """
    #     return self.planner.backward(
    #         response=response,
    #         prompt_kwargs=prompt_kwargs,
    #         template=template,
    #         backward_engine=backward_engine,
    #         id=id,
    #         disable_backward_engine=disable_backward_engine,
    #     )

    # def update_runner(
    #     self,
    #     planner: Optional[Agent] = None,
    #     stream_parser: Optional["StreamParser"] = None,
    #     output_parser: Optional[OutputParser] = None,
    #     output_class: Optional[Type] = None,
    #     context_map: Optional[Dict[str, Function]] = None,
    #     planner_config: Optional[Dict[str, Any]] = None,
    # ) -> None:
    #     """Update runner configuration where the user can optionally provide a new planner instance or
    #     a configuration of the planner if it is to be updated."""

    #     # if both planner instance and planner_config is provided update using planner
    #     if planner is not None:
    #         self.planner = planner
    #     else:
    #         if planner_config is not None:
    #             self.planner = Agent.from_config(planner_config)

    #     # keep as is if None
    #     self.config = RunnerConfig(
    #         stream_parser=stream_parser or self.config.stream_parser,
    #         output_parser=output_parser or self.config.output_parser,
    #         output_class=output_class or self.config.output_class,
    #         context_map=context_map or self.config.context_map,
    #     )

    # """
    #   internal tool to execute tools as necessary based on the generator output of call and stream_call
    # """

    def _tool_execute(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Execute a tool function through the planner's tool manager.
        Handles both sync and async functions.
        """
        return self.agent.tool_manager.call(expr_or_fun=func, step="execute")

        # except Exception as e:
        #     # TODO check map_f
        #     if func is not None and isinstance(func, Function) and map_fn is None:
        #         function_call_response = asyncio.run(
        #             self.planner.context_map[func.name](**func.kwargs)
        #         )
        #         return function_call_response
        #     else:
        #         raise ValueError(f"Error {e} executing function: {func}")

    # @classmethod
    # def from_config(cls, config: Dict[str, Any]) -> 'Runner':
    #     """Create a Runner instance from a configuration dictionary.

    #     Args:
    #         config: Configuration dictionary containing:
    #             - planner: planner configuration
    #             - stream_parser: Optional stream parser
    #             - output_parser: Optional output parser
    #             - output_class: Optional output class
    #             - context_map: Optional context map
    #             - name: Optional name for the runner

    #     Example 1: Create a Runner from a config dictionary
    #     runner_config = {
    #         'name': 'example_runner',
    #         'planner': {
    #             'name': 'example_planner',
    #             'system_prompt': 'You are a helpful assistant.',
    #             'model_client': {
    #                 'component_name': 'OpenAIClient',
    #                 'component_config': {
    #             'api_key': 'your-api-key',
    #                     'model': 'gpt-3.5-turbo'
    #                 }
    #             },
    #             'tool_manager': {
    #                 'tools': []  # List of tools would go here
    #             }
    #         },
    #         'output_parser': lambda x: {'text': x.text},  # Simple output parser
    #         'output_class': GeneratorOutput,
    #     }

    #     Returns:
    #         Configured Runner instance
    #     """

    #     # Extract planner config/instance
    #     try:
    #         planner = planner.from_config(config.get('planner'))
    #     except Exception as e:
    #         raise ValueError(f"Failed to create planner from config: {e}")

    #     # Create runner instance
    #     runner = cls(
    #         planner=planner,
    #         stream_parser=config.get('stream_parser', None),
    #         output_parser=config.get('output_parser', None),
    #         output_class=config.get('output_class', GeneratorOutput),
    #     )

    #     return runner

    # def return_state_dict(self) -> Dict[str, Any]:
    #     """Return the state of the runner as a dictionary that can be used to recreate it.

    #     Returns:
    #         Dictionary containing the runner's state
    #     """
    #     return {
    #         'planner': self.planner.return_state_dict(),
    #         'stream_parser': self.config.stream_parser,
    #         'output_parser': self.config.output_parser,
    #         'output_class': self.config.output_class,
    #         'class_name': self.__class__.__name__,
    #         'module_name': self.__class__.__module__
    #     }
