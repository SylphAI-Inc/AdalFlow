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

        # get agent requirements
        self.max_steps = agent.max_steps
        self.answer_data_type = agent.answer_data_type

        self.step_history = []
        # add the llm call to the executor as a tool

    def _check_last_step(self, step: Function) -> bool:
        """Check if the last step is the finish step.

        Args:
            step_history: List of previous steps

        Returns:
            bool: True if the last step is a finish step
        """

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
        return data
        # if not self.answer_data_type:
        #     return data

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

    @classmethod
    def _get_planner_function(self, output: GeneratorOutput) -> Function:
        """Check the planner output and return the function.

        Args:
            output: The planner output
        """
        if not isinstance(output, GeneratorOutput):
            raise ValueError(
                f"Expected GeneratorOutput, but got {type(output)}, value: {output}"
            )

        function = output.data

        if not isinstance(function, Function):
            raise ValueError(
                f"Expected Function in the data field of the GeneratorOutput, but got {type(function)}, value: {function}"
            )

        return function

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
        # reset the step history
        self.step_history = []

        # take in the query in prompt_kwargs
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        prompt_kwargs["step_history"] = (
            self.step_history
        )  # a reference to the step history

        model_kwargs = model_kwargs.copy() if model_kwargs else {}

        step_count = 0
        last_output = None

        # set maximum number of steps for the planner into the prompt
        # prompt_kwargs["max_steps"] = self.max_steps

        while step_count < self.max_steps:
            try:
                # Execute one step
                output = self.agent.planner.call(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )

                function = self._get_planner_function(output)

                # execute the tool
                function_results = self._tool_execute_sync(function)

                # create a step output
                step_ouput: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=function_results.output,
                )
                self.step_history.append(step_ouput)

                if self._check_last_step(function):
                    last_output = self._process_data(function_results.output)
                    break

                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )

                step_count += 1

            except Exception as e:
                log.error(f"Error in step {step_count}: {str(e)}")
                return f"Error in step {step_count}: {str(e)}"

        return self.step_history, last_output

    def _tool_execute_sync(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Call this in the call method.
        Handles both sync and async functions by running async ones in event loop.
        """
        import inspect
        import asyncio

        result = self.agent.tool_manager(expr_or_fun=func, step="execute")

        # Check if result is a coroutine (async tool)
        if inspect.iscoroutine(result):
            return asyncio.run(result)
        # async generator
        elif inspect.isasyncgen(result):
            # For async generators, we need to collect all yielded values
            async def collect_async_gen():
                items = []
                async for item in result:
                    items.append(item)
                # Return the final meaningful result (last item)
                return items[-1] if items else None

            return asyncio.run(collect_async_gen())
        else:
            # Sync tool result
            return result

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

        prompt_kwargs["step_history"] = (
            self.step_history
        )  # a reference to the step history

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

                function = self._get_planner_function(output)
                function_results = await self._tool_execute_async(function)

                step_output: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=function_results.output,
                )
                self.step_history.append(step_output)

                if self._check_last_step(function):
                    last_output = self._process_data(function_results.output)
                    break

                # important to ensure the prompt at each step is correct
                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                return self.step_history, error_msg

        return self.step_history, last_output

    async def _tool_execute_async(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Call this in the acall method.
        Handles both sync and async functions.
        """
        import inspect

        result = self.agent.tool_manager(expr_or_fun=func, step="execute")

        # Check if result is a coroutine (async tool)
        if inspect.iscoroutine(result):
            return await result
        # async generator
        elif inspect.isasyncgen(result):
            return await result
        else:
            # Sync tool result
            return result

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
