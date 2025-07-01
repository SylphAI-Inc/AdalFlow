

from pydantic import BaseModel
import logging
import json
import inspect
import asyncio
from dataclasses import dataclass

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
from adalflow.utils import printc
from adalflow.core.component import Component
from adalflow.core.agent import Agent



__all__ = ["Runner"]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)  # Changed to use Pydantic BaseModel

@dataclass 
class RunResultStreaming:
    step_history: List[StepOutput]
    output: T
    _run_impl_task: asyncio.Task 
    _event_queue: asyncio.Queue 

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

        result = self.agent.tool_manager(expr_or_fun=func, step="execute")

        if not isinstance(result, FunctionOutput):
            raise ValueError("Result is not a FunctionOutput")

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

                # if inspect.iscoroutine(result):
                #     function_results = await result
                # else:
                #     function_results = None
                #     async for item in result:
                #         function_results = item
                # function_results = await self._tool_execute_async(function)

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
                printc(f'agent planner prompt: {self.agent.planner.get_prompt(**prompt_kwargs)}')

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                return self.step_history, error_msg

        return self.step_history, last_output

    def astream(self, prompt_kwargs: Dict[str, Any], model_kwargs: Optional[Dict[str, Any]] = None, use_cache: Optional[bool] = None, id: Optional[str] = None):
        import asyncio
        self._event_queue = asyncio.Queue()
        self._run_impl_task = asyncio.create_task(self.impl_astream(prompt_kwargs, model_kwargs, use_cache, id))
        return self._event_queue

    async def impl_astream(
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
                 # important to ensure the prompt at each step is correct
                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )
                printc(f'agent planner prompt: {self.agent.planner.get_prompt(**prompt_kwargs)}')

                # Execute one step asynchronously
                output = await self.agent.planner.acall(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )

                function = self._get_planner_function(output)
                printc(f'function: {function}', color="yellow")
                function_result = await self._tool_execute_async(function) # everything must be wrapped in FunctionOutput 

                if not isinstance(function_result, FunctionOutput):
                    raise ValueError(f"Result must be wrapped in FunctionOutput, got {type(function_result)}")

                function_output = function_result.output 
                real_function_output = None

                if inspect.iscoroutine(function_output):
                    real_function_output = await function_output
                elif inspect.isasyncgen(function_output):
                    # handle async generator
                    printc(f"async generator detected")
                    function_results = []
                    async for item in function_output:
                        self._event_queue.put_nowait(item)
                        function_results.append(item)
                    real_function_output = function_results[-1]
                    # function_results = []
                    # async for item in function_output:
                    #     function_results.append(item)
                    #     yield item 
                    # real_function_output = function_results[-1]
                # function_results = await self._tool_execute_async(function)

                step_output: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=real_function_output,
                )
                self.step_history.append(step_output)

                if self._check_last_step(function):
                    last_output = self._process_data(real_function_output)
                    break

               
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
        Note: this version has no support for streaming.
        """

        result = await self.agent.tool_manager.execute_func_async(func=func)

        if not isinstance(result, FunctionOutput):
            raise ValueError("Result is not a FunctionOutput")
        return result



    