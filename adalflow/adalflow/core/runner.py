from pydantic import BaseModel
import logging
import inspect
import asyncio
import ast


from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    AsyncIterable,
)
from typing_extensions import TypeAlias


from adalflow.optim.parameter import Parameter
from adalflow.utils import printc
from adalflow.core.component import Component
from adalflow.core.agent import Agent

from adalflow.core.types import (
    GeneratorOutput,
    FunctionOutput,
    Function,
    StepOutput,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    ToolCallRunItem,
    ToolOutputRunItem,
    StepRunItem,
    FinalOutputItem,
    RunnerStreamingResult,
    RunnerResult,
    QueueCompleteSentinel,
    ToolOutput,
)
from adalflow.core.functional import _is_pydantic_dataclass, _is_adalflow_dataclass


__all__ = ["Runner"]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)  # Changed to use Pydantic BaseModel

BuiltInType: TypeAlias = Union[str, int, float, bool, list, dict, tuple, set, None]
PydanticDataClass: TypeAlias = Type[BaseModel]
AdalflowDataClass: TypeAlias = Type[
    Any
]  # Replace with your actual Adalflow dataclass type if available


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
        ctx: Optional[Dict] = None,
        max_steps: Optional[int] = None,
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
        self.max_steps = max_steps 
        if max_steps is None:
            self.max_steps = agent.max_steps
        else:
            # overwrite the agent's max_steps
            self.agent.max_steps = max_steps
        self.answer_data_type = agent.answer_data_type or str

        self.step_history: List[StepOutput] = []

    def _check_last_step(self, step: Function) -> bool:
        """Check if the last step is the finish step."""
        if step.name == "finish":
            return True

        return False

    # TODO: improved after the finish function is refactored
    def _process_data(
        self,
        data: Union[BuiltInType, PydanticDataClass, AdalflowDataClass],
        id: Optional[str] = None,
    ) -> T:
        """Process the generator output data field and convert to the specified pydantic data class of output_type.

        Args:
            data: The data to process
            id: Optional identifier for the output

        Returns:
            str: The processed data as a string
        """

        try:
            model_output = None
            log.info(f"answer_data_type: {type(self.answer_data_type)}")
            if _is_pydantic_dataclass(self.answer_data_type):
                # data should be a string that represents a dictionary
                log.info(
                    f"initial answer returned by finish when user passed a pydantic type: {data}, type: {type(data)}"
                )
                data = str(data)
                dict_obj = ast.literal_eval(data)
                log.info(
                    f"initial answer after being evaluated using ast: {dict_obj}, type: {type(dict_obj)}"
                )
                model_output = self.answer_data_type(**dict_obj)
            elif _is_adalflow_dataclass(self.answer_data_type):
                # data should be a string that represents a dictionary
                log.info(
                    f"initial answer returned by finish when user passed a adalflow dataclass type: {data}, type: {type(data)}"
                )
                data = str(data)
                dict_obj = ast.literal_eval(data)
                log.info(
                    f"initial answer after being evaluated using ast: {dict_obj}, type: {type(dict_obj)}"
                )
                model_output = self.answer_data_type.from_dict(dict_obj)
            else:  # expect data to be a python built_in_type
                log.info(
                    f"type of answer is neither a pydantic dataclass or adalflow dataclass, answer before being casted again for safety: {data}, type: {type(data)}"
                )
                try:
                    # if the data is a python built_in_type then we can return it directly
                    # as the prompt passed to the LLM requires this
                    if not isinstance(data, self.answer_data_type):
                        raise ValueError(
                            f"Expected data of type {self.answer_data_type}, but got {type(data)}"
                        )
                    model_output = data
                except Exception as e:
                    log.error(
                        f"Failed to parse output: {data}, {e} for answer_data_type: {self.answer_data_type}"
                    )
                    model_output = None
                    raise ValueError(f"Error processing output: {str(e)}")

            if not model_output:
                raise ValueError(f"Failed to parse output: {data}")

            return model_output

        except Exception as e:
            log.error(f"Error processing output: {str(e)}")
            raise ValueError(f"Error processing output: {str(e)}")

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
    ) -> RunnerResult:
        """Execute the planner synchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request

        Returns:
            RunnerResult containing step history and final processed output
        """
        # reset the step history
        self.step_history = []

        # take in the query in prompt_kwargs
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        prompt_kwargs["step_history"] = (
            self.step_history
        )  # a reference to the step history
        # set maximum number of steps for the planner into the prompt
        prompt_kwargs["max_steps"] = self.max_steps

        model_kwargs = model_kwargs.copy() if model_kwargs else {}

        step_count = 0
        last_output = None



        while step_count < self.max_steps:
            try:
                printc(
                    f"agent planner prompt: {self.agent.planner.get_prompt(**prompt_kwargs)}"
                )
                # call the planner
                output = self.agent.planner.call(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )
                printc(f"planner output: {output}", color="yellow")

                function = self._get_planner_function(output)
                printc(f"function: {function}", color="yellow")

                function_results = self._tool_execute_sync(function)

                # observation is what llm see about the output, and user can control it using ToolOutput
                # output is the real output data 

                function_output = function_results.output
                function_output_observation = function_output
                if isinstance(function_output, ToolOutput) and hasattr(function_output, "observation"):
                    function_output_observation = function_output.observation

                # create a step output
                step_ouput: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=function_output_observation,
                )
                self.step_history.append(step_ouput)

                if self._check_last_step(function):
                    processed_data = self._process_data(function_output)
                    # Wrap final output in RunnerResponse
                    last_output = RunnerResult(
                        answer=processed_data,
                        step_history=self.step_history.copy(),
                    )
                    break

                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                error_response = RunnerResult(
                    error=error_msg,
                    step_history=self.step_history.copy(),
                )
                return error_response

        return last_output

    def _tool_execute_sync(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Call this in the call method.
        Handles both sync and async functions by running async ones in event loop.
        """

        result = self.agent.tool_manager.execute_func(func=func)

        if not isinstance(result, FunctionOutput):
            raise ValueError("Result is not a FunctionOutput")

        return result

    async def acall(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> RunnerResult:
        """Execute the planner asynchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request

        Returns:
            RunnerResponse containing step history and final processed output
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}

        prompt_kwargs["step_history"] = (
            self.step_history
        )  # a reference to the step history
        # set maximum number of steps for the planner into the prompt
        prompt_kwargs["max_steps"] = self.max_steps

        model_kwargs = model_kwargs.copy() if model_kwargs else {}

        step_count = 0
        last_output = None

        while step_count < self.max_steps:
            try:
                printc(
                    f"agent planner prompt: {self.agent.planner.get_prompt(**prompt_kwargs)}"
                )
                # Execute one step asynchronously
                output = await self.agent.planner.acall(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )
                printc(f"planner output: {output}", color="yellow")

                function = self._get_planner_function(output)
                printc(f"function: {function}", color="yellow")

                function_results = await self._tool_execute_async(function)

                function_output = function_results.output
                function_output_observation = function_output

                if isinstance(function_output, ToolOutput) and hasattr(function_output, "observation"):
                    function_output_observation = function_output.observation

                step_output: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=function_output_observation,
                )
                self.step_history.append(step_output)

                if self._check_last_step(function):
                    processed_data = self._process_data(function_output)
                    # Wrap final output in RunnerResult
                    last_output = RunnerResult(
                        answer=processed_data,
                        step_history=self.step_history.copy(),
                    )
                    break

                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                error_response = RunnerResult(
                    error=error_msg,
                    step_history=self.step_history.copy(),
                )
                return error_response

        return last_output

    def astream(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> RunnerStreamingResult:
        """
        Execute the runner asynchronously with streaming support.

        Returns:
            RunnerStreamingResult: A streaming result object with stream_events() method
        """
        result = RunnerStreamingResult()
        result._run_task = asyncio.create_task(
            self.impl_astream(prompt_kwargs, model_kwargs, use_cache, id, result)
        )
        return result

    async def impl_astream(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
        streaming_result: Optional[RunnerStreamingResult] = None,
    ) -> None:
        """Execute the planner asynchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}

        prompt_kwargs["step_history"] = self.step_history
        # a reference to the step history
        # set maximum number of steps for the planner into the prompt
        prompt_kwargs["max_steps"] = self.max_steps

        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        step_count = 0

        while step_count < self.max_steps:
            try:
                # important to ensure the prompt at each step is correct
                log.debug(
                    "The prompt with the prompt template is {}".format(
                        self.agent.planner.get_prompt(**prompt_kwargs)
                    )
                )
                printc(
                    f"agent planner prompt: {self.agent.planner.get_prompt(**prompt_kwargs)}"
                )

                # when it's streaming, the output will be an async generator
                output = await self.agent.planner.acall(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id,
                )

                if not isinstance(output, GeneratorOutput):
                    raise ValueError("The output is not a GeneratorOutput")

                if isinstance(output.raw_response, AsyncIterable):
                    # Streaming llm call - iterate through the async generator
                    async for event in output.raw_response:
                        wrapped_event = RawResponsesStreamEvent(data=event) # raw response wrapper 
                        streaming_result._event_queue.put_nowait(wrapped_event)

                else:
                    # yield the final planner response
                    if output.error is not None:
                        wrapped_event = RawResponsesStreamEvent(error=output.error)
                    else:
                        wrapped_event = RawResponsesStreamEvent(data=output.data) # wrap on the data field to be the final output, the data might be null
                    streaming_result._event_queue.put_nowait(wrapped_event)

                # asychronously consuming the raw response will
                # update the data field of output with the result of the output processor

                function = self._get_planner_function(output)
                printc(f"function: {function}", color="yellow")

                # Emit tool call event
                tool_call_item = ToolCallRunItem(data=function)
                tool_call_id  = tool_call_item.id
                tool_call_event = RunItemStreamEvent(
                    name="agent.tool_call_start", item=tool_call_item
                )
                streaming_result._event_queue.put_nowait(tool_call_event)

                function_result = await self._tool_execute_async(
                    function
                )  # everything must be wrapped in FunctionOutput

                if not isinstance(function_result, FunctionOutput):
                    raise ValueError(
                        f"Result must be wrapped in FunctionOutput, got {type(function_result)}"
                    )

                function_output = function_result.output
                # TODO: function needs a stream_events
                real_function_output = None


                # TODO: validate when the function is a generator

                if inspect.iscoroutine(function_output):
                    real_function_output = await function_output
                elif inspect.isasyncgen(function_output):
                    function_results = []
                    async for item in function_output:
                        streaming_result._event_queue.put_nowait(item)
                        function_results.append(item)
                    real_function_output = function_results[-1]

                # create call complete 
                call_complete_event = RunItemStreamEvent(
                    name="agent.tool_call_complete",
                    item=ToolOutputRunItem(
                        id=tool_call_id,
                        data=FunctionOutput(
                            name=function.name,
                            input=function,
                            output=real_function_output,
                        )
                    ),
                )
                streaming_result._event_queue.put_nowait(call_complete_event)

                function_output = real_function_output
                function_output_observation = function_output

                if isinstance(function_output, ToolOutput) and hasattr(function_output, "observation"):
                    function_output_observation = function_output.observation

                step_output: StepOutput = StepOutput(
                    step=step_count,
                    action=function,
                    function=function,
                    observation=function_output_observation,
                )
                self.step_history.append(step_output)

                # Emit step completion event
                step_item = StepRunItem(data=step_output)
                step_event = RunItemStreamEvent(
                    name="agent.step_complete", item=step_item
                )
                streaming_result._event_queue.put_nowait(step_event)

                if self._check_last_step(function):
                    processed_data = self._process_data(real_function_output)
                    printc(f"processed_data: {processed_data}", color="yellow")

                    # Create RunnerResult for the final output
                    runner_result = RunnerResult(
                        answer=processed_data,
                        step_history=self.step_history.copy(),
                    )

                    # Emit execution complete event
                    final_output_item = FinalOutputItem(data=runner_result)
                    final_output_event = RunItemStreamEvent(
                        name="agent.execution_complete", item=final_output_item
                    )
                    streaming_result._event_queue.put_nowait(final_output_event)

                    # Store final result and completion status
                    streaming_result.answer = processed_data
                    streaming_result.step_history = self.step_history.copy()
                    streaming_result._is_complete = True
                    break

                step_count += 1

            except Exception as e:
                error_msg = f"Error in step {step_count}: {str(e)}"
                log.error(error_msg)
                # Store error result and completion status
                streaming_result.step_history = self.step_history.copy()
                streaming_result._is_complete = True
                streaming_result.exception = error_msg

                # Emit error as FinalOutputItem to queue
                error_final_item = FinalOutputItem(error=error_msg)
                error_event = RunItemStreamEvent(
                    name="runner_finished", item=error_final_item
                )
                streaming_result._event_queue.put_nowait(error_event)

                # end the streaming result's event queue
                streaming_result._event_queue.put_nowait(QueueCompleteSentinel())
                return error_msg

        # Signal completion of streaming
        streaming_result._event_queue.put_nowait(QueueCompleteSentinel())

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
