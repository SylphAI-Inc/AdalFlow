"""Agent runner component for managing and executing agent workflows."""

from pydantic import BaseModel
import logging
import inspect
import asyncio
import uuid
import json
from datetime import datetime

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    AsyncIterable,
)
from typing_extensions import TypeAlias
import sys


from adalflow.optim.parameter import Parameter
from adalflow.utils import printc
from adalflow.core.component import Component
from adalflow.components.agent.agent import Agent

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
    ToolCallActivityRunItem,
    UserQuery,
    AssistantResponse,
)
from adalflow.apps.permission_manager import PermissionManager
from adalflow.components.memory.memory import ConversationMemory
from adalflow.core.functional import _is_pydantic_dataclass, _is_adalflow_dataclass
from adalflow.tracing import (
    runner_span,
    tool_span,
    response_span,
    step_span,
)


__all__ = ["Runner"]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)  # Changed to use Pydantic BaseModel


def _is_unrecoverable_error(error: Optional[str]) -> bool:  # pragma: no cover
    """Check if an error string indicates an unrecoverable error.
    
    Unrecoverable errors include:
    - HTTP 400: Bad request (e.g., context too long)
    - HTTP 429: Rate limit exceeded  
    - HTTP 404: Model not found
    - "Connection error": Network connection issues
    
    This is marked as uncoverable for testing purposes.
    
    Args:
        error: Error string to check
        
    Returns:
        True if the error is unrecoverable, False otherwise
    """
    if not error:
        return False
    
    # Check for connection error string pattern (case insensitive)
    if "connection error" in error.lower():
        return True
    
    # Check for HTTP error codes
    if "400" in error or "429" in error or "404" in error:
        return True
    
    return False

BuiltInType: TypeAlias = Union[str, int, float, bool, list, dict, tuple, set, None]
PydanticDataClass: TypeAlias = Type[BaseModel]
AdalflowDataClass: TypeAlias = Type[
    Any
]  # Replace with your actual Adalflow dataclass type if available


# The runner will create tool call request, add a unique call id.
# TODO: move this to repo adalflow/agent
class Runner(Component):
    """Executes Agent instances with multi-step iterative planning and tool execution.

    The Runner orchestrates the execution of an Agent through multiple reasoning and action
    cycles. It manages the step-by-step execution loop where the Agent's planner generates
    Function calls that get executed by the ToolManager, with results fed back into the
    planning context for the next iteration.

    Execution Flow:
        1. Initialize step history and prompt context
        2. For each step (up to max_steps):
           a. Call Agent's planner to get next Function
           b. Execute the Function using ToolManager
           c. Add step result to history
           d. Check if Function is "finish" to terminate
        3. Process final answer to expected output type

    The Runner supports both synchronous and asynchronous execution modes, as well as
    streaming execution with real-time event emission. It includes comprehensive tracing
    and error handling throughout the execution pipeline.

    Attributes:
        agent (Agent): The Agent instance to execute
        max_steps (int): Maximum number of execution steps allowed
        answer_data_type (Type): Expected type for final answer processing
        step_history (List[StepOutput]): History of all execution steps
        ctx (Optional[Dict]): Additional context passed to tools
    """

    def __init__(
        self,
        agent: Agent,
        ctx: Optional[Dict] = None,
        max_steps: Optional[int] = None, # this will overwrite the agent's max_steps
        permission_manager: Optional[PermissionManager] = None,
        conversation_memory: Optional[ConversationMemory] = None,
        **kwargs,
    ) -> None:
        """Initialize runner with an agent and configuration.

        Args:
            agent: The agent instance to execute
            stream_parser: Optional stream parser
            output_type: Optional Pydantic data class type
            max_steps: Maximum number of steps to execute
            permission_manager: Optional permission manager for tool approval
            conversation_memory: Optional conversation memory
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.tool_manager = agent.tool_manager
        self.permission_manager = permission_manager
        # pass the tool_manager to the permission_manager
        if permission_manager is not None:
            permission_manager.set_tool_manager(self.tool_manager)

        self.conversation_memory = conversation_memory

        self.use_conversation_memory = conversation_memory is not None

        # get agent requirements
        self.max_steps = max_steps
        if max_steps is None:
            self.max_steps = agent.max_steps
        else:
            # overwrite the agent's max_steps
            self.agent.max_steps = max_steps
        self.answer_data_type = agent.answer_data_type or str

        self.step_history: List[StepOutput] = []

        # add ctx (it is just a reference, and only get added to the final response)
        # assume intermediate tool is gonna modify the ctx
        self.ctx = ctx

        # Initialize permission manager
        self._init_permission_manager()

        # Initialize cancellation flag
        self._cancelled = False
        self._cancel_callbacks = []
        self._current_task = None  # Track the current running task
        self._current_streaming_result = None  # Track the current streaming result

        # support thinking model
        self.is_thinking_model = agent.is_thinking_model if hasattr(agent, 'is_thinking_model')  else False
        
        # Token tracking
        self._token_consumption: Dict[str, Any] = {
            'total_prompt_tokens': 0,
            'current_step_tokens': 0,
            'steps_token_history': [],
            'last_total_tokens': 0  # Track last total to calculate step difference
        }

    def _init_permission_manager(self):
        """Initialize the permission manager and register tools that require approval."""
        if self.permission_manager and hasattr(self.agent, "tool_manager"):
            # Iterate through tools in the ComponentList
            for tool in self.agent.tool_manager.tools:
                if hasattr(tool, "definition") and hasattr(tool, "require_approval"):
                    tool_name = tool.definition.func_name
                    self.permission_manager.register_tool(
                        tool_name, tool.require_approval
                    )

    def set_permission_manager(
        self, permission_manager: Optional[PermissionManager]
    ) -> None:
        """Set or update the permission manager after runner initialization.

        Args:
            permission_manager: The permission manager instance to use for tool approval
        """
        self.permission_manager = permission_manager
        # Re-initialize to register tools with the new permission manager
        self._init_permission_manager()

        # pass the tool_manager to the permission_manager
        if permission_manager is not None:
            permission_manager.set_tool_manager(self.tool_manager)



    def is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancelled

    def reset_cancellation(self) -> None:
        """Reset the cancellation flag for a new execution."""
        self._cancelled = False
    
    def get_token_consumption(self) -> Dict[str, Any]:
        """Get the current token consumption statistics.
        
        Returns:
            Dict containing token consumption data:
            - total_prompt_tokens: Total tokens consumed across all steps
            - current_step_tokens: Tokens from the most recent step
            - steps_token_history: List of token counts per step
        """
        return self._token_consumption.copy()
    
    def _update_token_consumption(self) -> None:
        """Update token consumption statistics by checking the planner's accumulated token count.
        
        Since the generator accumulates tokens, we calculate the step tokens as the difference
        from the last recorded total.
        """
        if hasattr(self.agent.planner, 'estimated_token_count'):
            current_total = self.agent.planner.estimated_token_count
            step_tokens = current_total - self._token_consumption['last_total_tokens']
            
            self._token_consumption['current_step_tokens'] = step_tokens
            self._token_consumption['total_prompt_tokens'] = current_total
            self._token_consumption['steps_token_history'].append(step_tokens)
            self._token_consumption['last_total_tokens'] = current_total
            
            return step_tokens
        return 0

    def register_cancel_callback(self, callback) -> None:
        """Register a callback to be called when execution is cancelled."""
        self._cancel_callbacks.append(callback)

    async def cancel(self) -> None:
        """Cancel the current execution.

        This will stop the current execution but preserve state like memory.
        """
        log.info("Runner.cancel() called - setting cancelled flag")
        self._cancelled = True

        # Try to emit a test event if we have a streaming result
        if hasattr(self, '_current_streaming_result') and self._current_streaming_result:
            try:
                cancel_received_event = RunItemStreamEvent(
                    name="runner.cancel_received",
                    item=FinalOutputItem(
                        data={
                        "status": "cancel_received",
                        "message": "Cancel request received",
                    })
                )
                self._current_streaming_result.put_nowait(cancel_received_event)
                log.info("Emitted cancel_received event")
            except Exception as e:
                log.error(f"Failed to emit cancel_received event: {e}")

        # Cancel the current streaming task if it exists
        if self._current_task and not self._current_task.done():
            log.info(f"Cancelling runner task: {self._current_task}")
            self._current_task.cancel()

            # Create a task to wait for cancellation to complete
            await self._wait_for_cancellation()

    async def _wait_for_cancellation(self):
        """Wait for task to be cancelled with timeout."""
        if self._current_task:
            try:
                # Wait up to 1 second for task to cancel gracefully
                await asyncio.wait_for(
                    self._current_task,
                    timeout=1.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Task didn't cancel in time or was cancelled - that's ok
                pass

    def _check_last_step(self, step: Function) -> bool:
        """Check if the last step has is_answer_final set to True."""
        if hasattr(step, "_is_answer_final") and step._is_answer_final:
            return True

        return False

    def _get_final_answer(self, function: Function) -> Any:
        """Get and process the final answer from the function."""
        if hasattr(function, "_answer"):
            return self._process_data(function._answer)
        return None


    def _create_runner_result(self, answer: Any, step_history, error: Optional[str] = None,  ) -> RunnerResult:
        """Create a RunnerResult object with the final answer and error."""
        return RunnerResult(
            answer=answer,
            step_history=step_history.copy(),
            error=error,
            # ctx=self.ctx,
        )
    def _create_execution_complete_stream_event(self, streaming_result: RunnerStreamingResult, final_output_item: FinalOutputItem):
        """Complete the streaming execution by adding a sentinel."""
        final_output_event = RunItemStreamEvent(
            name="agent.execution_complete",
            item=final_output_item,
        )
        streaming_result.put_nowait(final_output_event)

        runner_result: RunnerResult = final_output_item.data

        # set up the final answer
        streaming_result.answer = runner_result.answer if runner_result else None
        streaming_result.step_history = self.step_history.copy()
        streaming_result._is_complete = True

    def _add_assistant_response_to_memory(self, final_output_item: FinalOutputItem):
        # add the assistant response to the conversation memory
        if self.use_conversation_memory and self.conversation_memory._pending_user_query is not None:
            self.conversation_memory.add_assistant_response(
                AssistantResponse(
                    response_str=final_output_item.data.answer,
                    metadata={
                        "step_history": final_output_item.data.step_history.copy()
                    },
                )
            )

    def create_response_span(self, runner_result, step_count: int, streaming_result: RunnerStreamingResult, runner_span_instance, workflow_status: str = "stream_completed"):

        runner_span_instance.span_data.update_attributes(
            {
                "steps_executed": step_count + 1,
                "final_answer": runner_result.answer,
                "workflow_status": workflow_status,
            }
        )

        # Create response span for tracking final streaming result
        with response_span(
            answer=runner_result.answer,
            result_type=type(runner_result.answer).__name__,
            execution_metadata={
                "steps_executed": step_count + 1,
                "max_steps": self.max_steps,
                "workflow_status": workflow_status,
                "streaming": True,
            },
            response=runner_result,
        ):
            pass




    async def _process_stream_final_step(
        self,
        answer: Any,
        step_count: int,
        streaming_result,
        runner_span_instance
    ) -> FinalOutputItem:
        """Process the final step and trace it."""
        # processed_data = self._get_final_answer(function)
        # printc(f"processed_data: {processed_data}", color="yellow")

        # Runner result is the same as the sync/async call result

        runner_result = self._create_runner_result(
            answer=answer,
            step_history=self.step_history,
        )

        # Update runner span with final results
        # self.create_response_span(
        #     runner_result=runner_result,
        #     step_count=step_count,
        #     streaming_result=streaming_result,
        #     runner_span_instance=runner_span_instance,
        #     workflow_status="stream_completed",
        # )

        # Emit execution complete event
        final_output_item = FinalOutputItem(data=runner_result)
        self._create_execution_complete_stream_event(
            streaming_result, final_output_item
        )
        # add the assistant response to the conversation memory
        self._add_assistant_response_to_memory(final_output_item)
        return final_output_item

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

            # returns a dictionary in this case
            if _is_pydantic_dataclass(self.answer_data_type):
                log.info(
                    f"initial answer returned by finish when user passed a pydantic type: {data}, type: {type(data)}"
                )
                # if it has not yet been deserialized then deserialize into dictionary using json loads
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in data: {e}")
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected dict after JSON parsing, got {type(data)}")
                log.info(
                    f"initial answer after being evaluated using json: {data}, type: {type(data)}"
                )
                # data should be a string that represents a dictionary
                model_output = self.answer_data_type(**data)
            elif _is_adalflow_dataclass(self.answer_data_type):
                log.info(
                    f"initial answer returned by finish when user passed a adalflow type: {data}, type: {type(data)}"
                )

                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in data: {e}")
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected dict after JSON parsing, got {type(data)}")
                log.info(
                    f"initial answer after being evaluated using json: {data}, type: {type(data)}"
                )
                # data should be a string that represents a dictionary
                model_output = self.answer_data_type.from_dict(data)
            else:  # expect data to be a python built_in_type
                log.info(
                    f"type of answer is neither a pydantic dataclass or adalflow dataclass, answer before being casted again for safety: {data}, type: {type(data)}"
                )
                data = self.answer_data_type(
                    data
                )  # directly cast using the answer_data_type
                if not isinstance(data, self.answer_data_type):
                    raise ValueError(
                        f"Expected data of type {self.answer_data_type}, but got {type(data)}"
                    )
                model_output = data

            if not model_output:
                raise ValueError(f"Failed to parse output: {data}")

            return model_output

        except Exception as e:
            log.error(f"Error processing output: {str(e)}")
            raise ValueError(f"Error processing output: {str(e)}")

    @classmethod
    def _get_planner_function(self, output: GeneratorOutput) -> Optional[Function]:
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
            # can still self-recover in the agent for formatting.
            # raise ValueError(
            #     f"Expected Function in the data field of the GeneratorOutput, but got {type(function)}, value: {function}"
            # )
            return None

        return function

    def call(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,  # global run id
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
        # Create runner span for tracing
        with runner_span(
            runner_id=id or f"runner_{hash(str(prompt_kwargs))}",
            max_steps=self.max_steps,
            workflow_status="starting",
        ) as runner_span_instance:
            # reset the step history
            self.step_history = []

            # take in the query in prompt_kwargs
            prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
            prompt_kwargs["step_history"] = (
                self.step_history
            )  # a reference to the step history

            if self.use_conversation_memory:
                # Reset any pending query state before starting a new query
                self.conversation_memory.reset_pending_query()

                prompt_kwargs["chat_history_str"] = self.conversation_memory()
                # save the user query to the conversation memory

                # meta data is all keys in the list of context_str
                query_metadata = {"context_str": prompt_kwargs.get("context_str", None)}
                self.conversation_memory.add_user_query(
                    UserQuery(
                        query_str=prompt_kwargs.get("input_str", None),
                        metadata=query_metadata,
                    )
                )

            # set maximum number of steps for the planner into the prompt
            prompt_kwargs["max_steps"] = self.max_steps

            model_kwargs = model_kwargs.copy() if model_kwargs else {}

            step_count = 0
            last_output = None
            current_error = None

            while step_count < self.max_steps:
                try:
                    log.debug(f"Running step {step_count + 1}/{self.max_steps} with prompt_kwargs: {prompt_kwargs}")
                    # Create step span for each iteration
                    with step_span(
                        step_number=step_count, action_type="planning"
                    ) as step_span_instance:       

                        # Call the planner first to get the output
                        output = self.agent.planner.call(
                            prompt_kwargs=prompt_kwargs,
                            model_kwargs=model_kwargs,
                            use_cache=use_cache,
                            id=id,
                        )
                        
                        # Track token usage
                        step_tokens = self._update_token_consumption()
                        if step_tokens > 0:
                            log.debug(f"Step {step_count} - Prompt tokens: {step_tokens}, Total: {self._token_consumption['total_prompt_tokens']}")

                        log.debug(f"planner output: {output}")

                        # consistency with impl_astream, break if output is not a Generator Output 
                        if not isinstance(output, GeneratorOutput):
                            # Create runner finish event with error and stop the loop
                            current_error = (
                                f"Expected GeneratorOutput, but got {output}"
                            )
                            # add this to the step history
                            step_output = StepOutput(
                                step=step_count,
                                action=None,
                                function=None,
                                observation=current_error,
                            )
                            self.step_history.append(step_output)
                            break

                        function = output.data

                        log.debug(f"function: {function}")
                        if function is None:
                            error_msg = f"Run into error: {output.error}, raw response: {output.raw_response}"
                            # Handle recoverable vs unrecoverable errors
                            if output.error is not None:
                                if _is_unrecoverable_error(output.error):
                                    # Unrecoverable errors: context too long, rate limit, model not found
                                    current_error = output.error
                                    step_output = StepOutput(
                                        step=step_count,
                                        action=None,
                                        function=None,
                                        observation=f"Unrecoverable error: {output.error}",
                                    )
                                    self.step_history.append(step_output)
                                    break  # Stop execution for unrecoverable errors
                            # Recoverable errors: JSON format errors, parsing errors, etc.
                            current_error = output.error
                            step_output = StepOutput(
                                step=step_count,
                                action=None,
                                function=None,
                                observation=current_error,
                            )
                            self.step_history.append(step_output)
                            step_count += 1
                            continue  # Continue to next step for recoverable errors
                        
                        # start to process correct function
                        function.id = str(uuid.uuid4()) # add function id
                        thinking = output.thinking if hasattr(output, 'thinking') else None
                        if thinking is not None and self.is_thinking_model:
                            function.thought = thinking
                            

                        if self._check_last_step(function):
                            processed_data = self._process_data(function._answer)
                            # Wrap final output in RunnerResult
                            last_output = RunnerResult(
                                answer=processed_data,
                                step_history=self.step_history.copy(),
                                # ctx=self.ctx,
                            )

                            # Add assistant response to conversation memory
                            if self.use_conversation_memory:
                                self.conversation_memory.add_assistant_response(
                                    AssistantResponse(
                                        response_str=processed_data,
                                        metadata={
                                            "step_history": self.step_history.copy()
                                        },
                                    )
                                )

                            step_count += 1  # Increment step count before breaking
                            break

                        step_output: Optional[StepOutput] = None

                        # Create tool span for function execution
                        with tool_span(
                            tool_name=function.name,
                            function_name=function.name,
                            function_args=function.args,
                            function_kwargs=function.kwargs,
                        ) as tool_span_instance:
                            function_results = self._tool_execute_sync(function)
                            # Update span attributes using update_attributes for MLflow compatibility
                            tool_span_instance.span_data.update_attributes(
                                {"output_result": function_results.output}
                            )

                        function_output = function_results.output
                        real_function_output = None
                        
                        # Handle generator outputs in sync call
                        if inspect.iscoroutine(function_output):
                            # For sync call, we need to run the coroutine
                            real_function_output = asyncio.run(function_output)
                        elif inspect.isasyncgen(function_output):
                            # Collect all values from async generator
                            async def collect_async_gen():
                                collected_items = []
                                async for item in function_output:
                                    if isinstance(item, ToolCallActivityRunItem):
                                        # Skip activity items
                                        continue
                                    else:
                                        collected_items.append(item)
                                return collected_items
                            real_function_output = asyncio.run(collect_async_gen())
                        elif inspect.isgenerator(function_output):
                            # Collect all values from sync generator
                            collected_items = []
                            for item in function_output:
                                if isinstance(item, ToolCallActivityRunItem):
                                    # Skip activity items
                                    continue
                                else:
                                    collected_items.append(item)
                            real_function_output = collected_items
                        else:
                            real_function_output = function_output
                        
                        # Use the processed output
                        function_output = real_function_output
                        function_output_observation = function_output
                        if isinstance(function_output, ToolOutput) and hasattr(
                            function_output, "observation"
                        ):
                            function_output_observation = function_output.observation

                        # create a step output
                        step_output = StepOutput(
                            step=step_count,
                            action=function,
                            function=function,
                            observation=function_output_observation,
                        )

                        # Update step span with results
                        step_span_instance.span_data.update_attributes(
                            {
                                "tool_name": function.name,
                                "tool_output": function_results,
                                "is_final": self._check_last_step(function),
                                "observation": function_output_observation,
                            }
                        )

                        log.debug(
                            "The prompt with the prompt template is {}".format(
                                self.agent.planner.get_prompt(**prompt_kwargs)
                            )
                        )
                        self.step_history.append(step_output)
                        step_count += 1

                except Exception as e:
                    error_msg = f"Error in step {step_count}: {str(e)}"
                    log.error(error_msg)

                    # Create response span for error tracking
                    with response_span(
                        answer=error_msg,
                        result_type="error",
                        execution_metadata={
                            "steps_executed": step_count,
                            "max_steps": self.max_steps,
                            "workflow_status": "failed",
                        },
                        response=None,
                    ):
                        pass

                    # Continue to next step instead of returning
                    step_count += 1
                    current_error = error_msg
                    break

            # Update runner span with final results
            # Update runner span with completion info using update_attributes
            runner_span_instance.span_data.update_attributes(
                {
                    "steps_executed": step_count,
                    "final_answer": last_output.answer if last_output else None,
                    "workflow_status": "completed",
                }
            )

            # Create response span for tracking final result
            with response_span(
                answer=(
                    last_output.answer
                    if last_output
                    else f"No output generated after {step_count} steps (max_steps: {self.max_steps})"
                ),
                result_type=(
                    type(last_output.answer).__name__ if last_output else "no_output"
                ),
                execution_metadata={
                    "steps_executed": step_count,
                    "max_steps": self.max_steps,
                    "workflow_status": "completed" if last_output else "incomplete",
                },
                response=last_output,  # can be None if Runner has not finished in the max steps
            ):
                pass

            # Always return a RunnerResult, even if no successful completion
            return last_output or RunnerResult(
                answer=current_error or f"No output generated after {step_count} steps (max_steps: {self.max_steps})",
                step_history=self.step_history.copy(),
                error=current_error,
            )

    def _tool_execute_sync(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Call this in the call method.
        Handles both sync and async functions by running async ones in event loop.
        Includes permission checking if permission_manager is configured.
        """

        # execute permission and blocking mechanism in check_permission
        # TODO: permission manager might be better to be put inside of tool manager
        if self.permission_manager:

            result = asyncio.run(self.permission_manager.check_permission(func))

            # Handle both old (2 values) and new (3 values) return formats
            if len(result) == 3:
                allowed, modified_func, _ = result
            else:
                allowed, modified_func = result

            if not allowed:
                return FunctionOutput(
                    name=func.name,
                    input=func,
                    output=ToolOutput(
                        output="Tool execution cancelled by user",
                        observation="Tool execution cancelled by user",
                        display="Permission denied",
                        status="cancelled",
                    ),
                )

            # Use modified function if user edited it
            func = modified_func or func

        result = self.agent.tool_manager.execute_func(func=func)

        if not isinstance(result, FunctionOutput):
            raise ValueError("Result is not a FunctionOutput")

        # check error
        if result.error is not None:
            log.warning(f"Error in tool execution: {result.error}")
        # TODO: specify how to handle this error

        return result

    # support both astream and non-stream
    async def acall(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Optional[RunnerResult]:
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


        workflow_status = "starting"
        runner_id = id or f"async_runner_{hash(str(prompt_kwargs))}"

        

        # Create runner span for tracing
        with runner_span(
            runner_id=runner_id,
            max_steps=self.max_steps,
            workflow_status= workflow_status,
        ) as runner_span_instance:
            # Reset cancellation flag at start of new execution
            self.reset_cancellation()

            self.step_history = []
            prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}

            prompt_kwargs["step_history"] = (
                self.step_history
            )  # a reference to the step history

            if self.use_conversation_memory:
                # Reset any pending query state before starting a new query
                self.conversation_memory.reset_pending_query()

                prompt_kwargs["chat_history_str"] = self.conversation_memory()
                # save the user query to the conversation memory

                # meta data is all keys in the list of context_str
                query_metadata = {"context_str": prompt_kwargs.get("context_str", None)}
                self.conversation_memory.add_user_query(
                    UserQuery(
                        query_str=prompt_kwargs.get("input_str", None),
                        metadata=query_metadata,
                    )
                )

            # set maximum number of steps for the planner into the prompt
            prompt_kwargs["max_steps"] = self.max_steps

            model_kwargs = model_kwargs.copy() if model_kwargs else {}

            step_count = 0
            last_output = None
            current_error = None

            while step_count < self.max_steps and not self.is_cancelled():
                try:
                    log.debug(f"Running async step {step_count + 1}/{self.max_steps} with prompt_kwargs: {prompt_kwargs}")

                    # Create step span for each iteration
                    with step_span(
                        step_number=step_count, action_type="async_planning"
                    ) as step_span_instance:

                        log.debug(f"Running async step {step_count + 1}/{self.max_steps} with prompt_kwargs: {prompt_kwargs}")
                        
                        if self.is_cancelled():
                            raise asyncio.CancelledError("Execution cancelled by user")

                        # Call the planner first to get the output
                        output: GeneratorOutput = await self.agent.planner.acall(
                            prompt_kwargs=prompt_kwargs,
                            model_kwargs=model_kwargs,
                            use_cache=use_cache,
                            id=id,
                        )
                        
                        # Track token usage
                        step_tokens = self._update_token_consumption()
                        if step_tokens > 0:
                            log.debug(f"Step {step_count} - Prompt tokens: {step_tokens}, Total: {self._token_consumption['total_prompt_tokens']}")

                        log.debug(f"planner output: {output}")

                        if not isinstance(output, GeneratorOutput):
                            # Create runner finish event with error and stop the loop
                            current_error = (
                                f"Expected GeneratorOutput, but got {type(output)}"
                            )
                            # create a step output for the error
                            step_output = StepOutput(
                                step=step_count,
                                action=None,
                                function=None,
                                observation=current_error,
                            )
                            self.step_history.append(step_output)
                            step_count += 1
                            break
 


                        function = output.data

                        log.debug(f"function: {function}")
                        if function is None:
                            error_msg = f"Run into error: {output.error}, raw response: {output.raw_response}"
                            # Handle recoverable vs unrecoverable errors
                            if output.error is not None:
                                if _is_unrecoverable_error(output.error):
                                    # Unrecoverable errors: context too long, rate limit, model not found
                                    current_error = output.error
                                    step_output = StepOutput(
                                        step=step_count,
                                        action=None,
                                        function=None,
                                        observation=f"Unrecoverable error: {output.error}",
                                    )
                                    self.step_history.append(step_output)
                                    step_count += 1
                                    break  # Stop execution for unrecoverable errors
                            # Recoverable errors: JSON format errors, parsing errors, etc.
                            current_error = output.error
                            step_output = StepOutput(
                                step=step_count,
                                action=None,
                                function=None,
                                observation=current_error,
                            )
                            self.step_history.append(step_output)
                            step_count += 1

                            continue  # Continue to next step for recoverable errors`



                    thinking = output.thinking if hasattr(output, 'thinking') else None
                    if function is not None:
                        # add a function id
                        function.id = str(uuid.uuid4())
                        if thinking is not None and self.is_thinking_model:
                            function.thought = thinking
                            
                            

                    if self._check_last_step(function):
                        answer = self._get_final_answer(function)
                        # Wrap final output in RunnerResult
                        last_output = RunnerResult(
                            answer=answer,
                            step_history=self.step_history.copy(),
                            error=current_error,
                            # ctx=self.ctx,
                        )

                        # Add assistant response to conversation memory
                        if self.use_conversation_memory:
                            self.conversation_memory.add_assistant_response(
                                AssistantResponse(
                                    response_str=answer,
                                    metadata={
                                        "step_history": self.step_history.copy()
                                    },
                                )
                            )



                        step_count += 1  # Increment step count before breaking
                            
                        break

                    # Create tool span for function execution
                    with tool_span(
                        tool_name=function.name,
                        function_name=function.name,
                        function_args=function.args,
                        function_kwargs=function.kwargs,
                    ) as tool_span_instance:
                        function_results = await self._tool_execute_async(
                            func=function
                        )
                        function_output = function_results.output
                        # add the process of the generator and async generator
                        real_function_output = None
                        
                        # Handle generator outputs similar to astream implementation
                        if inspect.iscoroutine(function_output):
                            real_function_output = await function_output
                        elif inspect.isasyncgen(function_output):
                            # Collect all values from async generator
                            collected_items = []
                            async for item in function_output:
                                if isinstance(item, ToolCallActivityRunItem):
                                    # Skip activity items in acall
                                    continue
                                else:
                                    collected_items.append(item)
                            # Use collected items as output
                            real_function_output = collected_items
                        elif inspect.isgenerator(function_output):
                            # Collect all values from sync generator
                            collected_items = []
                            for item in function_output:
                                if isinstance(item, ToolCallActivityRunItem):
                                    # Skip activity items in acall
                                    continue
                                else:
                                    collected_items.append(item)
                            # Use collected items as output
                            real_function_output = collected_items
                        else:
                            real_function_output = function_output
                        
                        # Use the processed output
                        function_output = real_function_output
                        function_output_observation = function_output

                        if isinstance(function_output, ToolOutput) and hasattr(
                            function_output, "observation"
                        ):
                            function_output_observation = (
                                function_output.observation
                            )

                        # Update tool span attributes using update_attributes for MLflow compatibility
                        tool_span_instance.span_data.update_attributes(
                            {"output_result": function_output}
                        )

                        step_output: StepOutput = StepOutput(
                            step=step_count,
                            action=function,
                            function=function,
                            observation=function_output_observation,
                        )
                        self.step_history.append(step_output)

                        # Update step span with results
                        step_span_instance.span_data.update_attributes(
                            {
                                "tool_name": function.name,
                                "tool_output": function_results,
                                "is_final": self._check_last_step(function),
                                "observation": function_output_observation,
                            }
                        )

                        log.debug(
                            "The prompt with the prompt template is {}".format(
                                self.agent.planner.get_prompt(**prompt_kwargs)
                            )
                        )

                        step_count += 1

                except Exception as e:
                    error_msg = f"Error in step {step_count}: {str(e)}"
                    log.error(error_msg)

                    # Create response span for error tracking
                    with response_span(
                        answer=error_msg,
                        result_type="error",
                        execution_metadata={
                            "steps_executed": step_count,
                            "max_steps": self.max_steps,
                            "workflow_status": "failed",
                        },
                        response=None,
                    ):
                        pass

                    # Continue to next step instead of returning
                    step_count += 1
                    current_error = error_msg
                    break

            # Update runner span with final results
            # Update runner span with completion info using update_attributes
            runner_span_instance.span_data.update_attributes(
                {
                    "steps_executed": step_count,
                    "final_answer": last_output.answer if last_output else None,
                    "workflow_status": "completed",
                }
            )

            # Create response span for tracking final result
            with response_span(
                answer=(
                    last_output.answer
                    if last_output
                    else f"No output generated after {step_count} steps (max_steps: {self.max_steps})"
                ),
                result_type=(
                    type(last_output.answer).__name__ if last_output else "no_output"
                ),
                execution_metadata={
                    "steps_executed": step_count,
                    "max_steps": self.max_steps,
                    "workflow_status": "completed" if last_output else "incomplete",
                },
                response=last_output,  # can be None if Runner has not finished in the max steps
            ):
                pass

            # Always return a RunnerResult, even if no successful completion
            return last_output or RunnerResult(
                answer=current_error or f"No output generated after {step_count} steps (max_steps: {self.max_steps})",
                step_history=self.step_history.copy(),
                error=current_error,
            )
        
    
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
        # Cancel any previous task that might still be running
        # TODO might have problems of overwriting and cancelling other tasks if we call await astream two times asychronously with the same runner / agent instance.
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            log.info("Cancelled previous streaming task")
            # Don't wait for cancellation here - just cancel and move on
            self._current_task = None

        # Reset cancellation flag for new execution
        self._cancelled = False

        result = RunnerStreamingResult()
        # Store the streaming result so we can emit events to it during cancellation
        self._current_streaming_result = result

        self.reset_cancellation()

        # Store the task so we can cancel it if needed
        self._current_task = asyncio.get_event_loop().create_task(
            self.impl_astream(prompt_kwargs, model_kwargs, use_cache, id, result)
        )
        result._run_task = self._current_task
        return result

    async def impl_astream(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
        streaming_result: Optional[RunnerStreamingResult] = None,
    ) -> None:
        """
        Behave exactly the same as `acall` but with streaming support.

        - GeneratorOutput will be emitted as RawResponsesStreamEvent

        - StepOutput will be emitted as RunItemStreamEvent with name "agent.step_complete".

        - Finally, there will be a FinalOutputItem with the final answer or error.

        Execute the planner asynchronously for multiple steps with function calling support.

        At the last step the action should be set to "finish" instead which terminates the sequence

        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
        """
        workflow_status: Literal["streaming", "stream_completed", "stream_failed", "stream_incomplete"] = "streaming"
        # Create runner span for tracing streaming execution
        with runner_span(
            runner_id=id or f"stream_runner_{hash(str(prompt_kwargs))}",
            max_steps=self.max_steps,
            workflow_status= workflow_status,
        ) as runner_span_instance:
            
            # Reset cancellation flag at start of new execution
            self.step_history = []
            prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}

            prompt_kwargs["step_history"] = self.step_history
            if self.use_conversation_memory:
                # Reset any pending query state before starting a new query
                self.conversation_memory.reset_pending_query()

                prompt_kwargs["chat_history_str"] = self.conversation_memory()
                # save the user query to the conversation memory

                # meta data is all keys in the list of context_str
                query_metadata = {"context_str": prompt_kwargs.get("context_str", None)}
                self.conversation_memory.add_user_query(
                    UserQuery(
                        query_str=prompt_kwargs.get("input_str", None),
                        metadata=query_metadata,
                    )
                )
            # a reference to the step history
            # set maximum number of steps for the planner into the prompt
            prompt_kwargs["max_steps"] = self.max_steps

            model_kwargs = model_kwargs.copy() if model_kwargs else {}
            step_count = 0
            final_output_item = None
            current_error = None

            # whenever we have the final output, we break the loop, this includes
            # (1) final_answer (check final step)
            # (2) unrecoverable error in llm planner
            # (3) any exception

            # for normal, we will have raw_response_event, request_permission, tool_call_start, tool_call_activity, tool_call_complete, step_complete
            # for error, we can skip any step but will always have step_complete []


            # ToolOutput
            # has three status: success, error, canceled
            while step_count < self.max_steps and not self.is_cancelled():
                try:
                    # Create step span for each streaming iteration
                    # error handing: when run into any error, it creates a runner finish event. and stops the loop
                    # it should directly sent the execution complete with error event
                    with step_span(
                        step_number=step_count, action_type="stream_planning"
                    ) as step_span_instance:
                        # important to ensure the prompt at each step is correct
                        log.debug(
                            "The prompt with the prompt template is {}".format(
                                self.agent.planner.get_prompt(**prompt_kwargs)
                            )
                        )

                        # Check cancellation before calling planner
                        # TODO seems slightly unnecessary we are calling .cancel on the task in cancel which will raise this exception regardless unless we want to terminate earlier by checking the cancelled field
                        if self.is_cancelled():
                            raise asyncio.CancelledError("Execution cancelled by user")

                        # when it's streaming, the output will be an async generator
                        output: GeneratorOutput = await self.agent.planner.acall(
                            prompt_kwargs=prompt_kwargs,
                            model_kwargs=model_kwargs,
                            use_cache=use_cache,
                            id=id,
                        )
                        
                        # Track token usage
                        step_tokens = self._update_token_consumption()
                        if step_tokens > 0:
                            log.debug(f"Step {step_count} - Prompt tokens: {step_tokens}, Total: {self._token_consumption['total_prompt_tokens']}")

                        if not isinstance(output, GeneratorOutput):
                            # Create runner finish event with error and stop the loop
                            error_msg = (
                                f"Expected GeneratorOutput, but got {type(output)}"
                            )
                            final_output_item = FinalOutputItem(
                                error=error_msg,
                            )
                            workflow_status = "stream_failed"
                            current_error = error_msg
                            # create a step output for the error
                            step_output = StepOutput(
                                step=step_count,
                                action=None,
                                function=None,
                                observation=current_error,
                            )
                            self.step_history.append(step_output)
                            step_count += 1
                            break


                        # handle the generator output data and error
                        wrapped_event = None

                        if isinstance(output.raw_response, AsyncIterable):
                            log.debug(
                                f"Streaming raw response from planner: {output.raw_response}"
                            )
                            # Streaming llm call - iterate through the async generator
                            async for event in output.raw_response:
                                # TODO seems slightly unnecessary we are calling .cancel on the task in cancel which will raise this exception regardless
                                if self.is_cancelled():
                                    raise asyncio.CancelledError("Execution cancelled by user")
                                wrapped_event = RawResponsesStreamEvent(data=event)
                                streaming_result.put_nowait(wrapped_event)

                        else: # non-streaming cases
                            # yield the final planner response
                            if output.data is None:

                                # recoverable errors, continue to create stepout
                                current_error = output.error 
                                # wrap the error in a RawResponsesStreamEvent
                                wrapped_event = RawResponsesStreamEvent(
                                    data=None,  # no data in this case
                                    error= output.error,
                                )
                                streaming_result.put_nowait(wrapped_event)

        
                                step_output = StepOutput(
                                    step=step_count,
                                    action=None,
                                    function=None,
                                    observation=current_error,
                                )
                                # emit the step complete event with error which matches the step_output
                                step_item = StepRunItem(data=step_output)
                                step_complete_event = RunItemStreamEvent(
                                    name="agent.step_complete",
                                    item=step_item,
                                )
                                streaming_result.put_nowait(step_complete_event)
                                self.step_history.append(step_output)

                                if output.error is not None:
  
                                    if _is_unrecoverable_error(output.error): # context too long or rate limite, not recoverable
                                        # 404 model not exist
                                        # create a final output item with error and stop the loop
                                        final_output_item = FinalOutputItem(
                                            error=output.error,
                                        )
                                        workflow_status = "stream_failed"
                                        current_error = output.error
                                        step_output = StepOutput(
                                            step=step_count,
                                            action=None,
                                            function=None,
                                            observation=f"Unrecoverable error: {output.error}",
                                        )
                                        self.step_history.append(step_output)
                                        step_count += 1
                                        break
                                step_count += 1
                                continue  # continue to next step
                                    

                            # normal functions
                            wrapped_event = RawResponsesStreamEvent(
                                data=output.data, 
                            )  # wrap on the data field to be the final output, the data might be null
                            streaming_result.put_nowait(wrapped_event)

                        # asychronously consuming the raw response will
                        # update the data field of output with the result of the output processor

                        # handle function output 

                        function = output.data # here are the recoverable errors, should continue to step output
                        thinking = output.thinking # check the reasoning model response
                        if thinking is not None and self.is_thinking_model:
                            # if the thinking is not None, we will add it to the function
                            if function is not None and isinstance(function, Function):
                                function.thought = thinking

                        function.id = str(uuid.uuid4()) # add function id 
                        function_result = None
                        function_output_observation = None

                        if thinking is not None and self.is_thinking_model:
                            function.thought = thinking

                        # TODO: simplify this
                        tool_call_id = function.id
                        tool_call_name = function.name
                        log.debug(f"function: {function}")

                        if self._check_last_step(function): # skip stepoutput 
                            answer = self._get_final_answer(function)
                            final_output_item = await self._process_stream_final_step(
                                answer=answer,
                                step_count=step_count,
                                streaming_result=streaming_result,
                                runner_span_instance=runner_span_instance,
                            )
                            workflow_status = "stream_completed"
                            break

                        # Check if permission is required and emit permission event
                        # TODO: trace the permission event

                        function_output_observation = None
                        function_result = None
                        print("function name", function.name)
                        complete_step = False
                        if (
                            self.permission_manager
                            and self.permission_manager.is_approval_required(
                                function.name
                            )
                        ):
                            permission_event = (
                                self.permission_manager.create_permission_event(
                                    function
                                )
                            )
                            # there is an error
                            if isinstance(permission_event, ToolOutput):
                                # need a tool complete event
                                function_result = FunctionOutput(
                                    name=function.name,
                                    input=function,
                                    output=permission_event,
                                )
                                tool_complete_event = RunItemStreamEvent(
                                    name="agent.tool_call_complete",
                                    # error is already tracked in output
                                    # TODO: error tracking is not needed in RunItem, it is tracked in the tooloutput status.
                                    item=ToolOutputRunItem(
                                        data=function_result,
                                        id=tool_call_id,
                                        error=permission_event.observation if permission_event.status == "error" else None, # error message sent to the frontend
                                    ),
                                )
                                streaming_result.put_nowait(tool_complete_event)
                                function_output_observation = permission_event.observation
                                complete_step = True
                            else:
                                permission_stream_event = RunItemStreamEvent(
                                    name="agent.tool_permission_request",
                                    item=permission_event,
                                )
                                streaming_result.put_nowait(permission_stream_event)
                        if not complete_step:
                            # Execute the tool with streaming support
                            function_result, function_output, function_output_observation = await self.stream_tool_execution(
                                function=function,
                                tool_call_id=tool_call_id,
                                tool_call_name=tool_call_name,
                                streaming_result=streaming_result,
                            )

                        # Add step to history for approved tools (same as non-permission branch)
                        step_output: StepOutput = StepOutput(
                            step=step_count,
                            action=function,
                            function=function,
                            observation=function_output_observation,
                        )
                        self.step_history.append(step_output)

                        # Update step span with results (for both recoverable errors and normal function execution)
                        step_span_instance.span_data.update_attributes(
                            {
                                "tool_name": function.name if function else None,
                                "tool_output": function_result,
                                "is_final": self._check_last_step(function),
                                "observation": function_output_observation,
                            }
                        )

                        # Emit step completion event (with error if any)
                        step_item = StepRunItem(data=step_output)
                        step_event = RunItemStreamEvent(
                            name="agent.step_complete", item=step_item
                        )
                        streaming_result.put_nowait(step_event)
                        step_count += 1

                except asyncio.CancelledError:
                    # Handle cancellation gracefully
                    cancel_msg = "Execution cancelled by user"
                    log.info(cancel_msg)

                    # Emit cancellation event so frontend/logs can see it
                    cancel_event = RunItemStreamEvent(
                        name="runner.cancelled",
                        item=FinalOutputItem(data={
                            "status": "cancelled",
                            "message": cancel_msg,
                            "step_count": step_count,
                        })
                    )
                    streaming_result.put_nowait(cancel_event)

                    # Store cancellation result
                    streaming_result.answer = cancel_msg
                    streaming_result.step_history = self.step_history.copy()
                    streaming_result._is_complete = True 

                    # Add cancellation response to conversation memory
                    if self.use_conversation_memory:
                        self.conversation_memory.add_assistant_response(
                            AssistantResponse(
                                response_str="I apologize, but the execution was cancelled by the user.",
                                metadata={
                                    "step_history": self.step_history.copy(),
                                    "status": "cancelled",
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                        )

                    # Signal completion and break
                    streaming_result.put_nowait(QueueCompleteSentinel())
                    break

                except Exception as e:
                    # these excepts should almost never happen
                    error_msg = f"Error in step {step_count}: {str(e)}"
                    log.error(error_msg)

                    workflow_status = "stream_failed"
                    streaming_result._exception = error_msg

                    # Emit error as FinalOutputItem to queue
                    final_output_item = FinalOutputItem(error=error_msg)
                    # error_event = RunItemStreamEvent(
                    #     name="runner_finished", item=error_final_item
                    # )
                    current_error = error_msg
                    break

            # If loop terminated without creating a final output item, create our own
            # TODO this might be redundant
            if final_output_item is None:
                # Create a RunnerResult with incomplete status
                runner_result = RunnerResult(
                    answer=f"No output generated after {step_count} steps (max_steps: {self.max_steps})",
                    error=current_error,
                    step_history=self.step_history.copy(),

                )
                final_output_item = FinalOutputItem(data=runner_result)

                workflow_status = "stream_incomplete"
                current_error = f"No output generated after {step_count} steps (max_steps: {self.max_steps})"


            runner_span_instance.span_data.update_attributes(
                {
                    "steps_executed": step_count,
                    "final_answer": final_output_item.data.answer if final_output_item.data else None,
                    "workflow_status": workflow_status,
                }
            )

            # create runner result with or without error

            runner_result = RunnerResult(
                answer=final_output_item.data.answer if final_output_item.data else None,
                step_history=self.step_history.copy(),
                error=current_error,
            )

            self._create_execution_complete_stream_event(
                streaming_result=streaming_result,
                final_output_item=final_output_item,
            )

            # create response span for final output
            # if workflow_status  in ["stream_incomplete", "stream_failed"]:
            self.create_response_span(
                runner_result=runner_result,
                step_count=step_count,
                streaming_result=streaming_result,
                runner_span_instance=runner_span_instance,
                workflow_status=workflow_status,
            )

            # Signal completion of streaming
            streaming_result.put_nowait(QueueCompleteSentinel())

    async def _tool_execute_async(
        self,
        func: Function,
        streaming_result: Optional[RunnerStreamingResult] = None,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Call this in the acall method.
        Handles both sync and async functions.
        Note: this version has no support for streaming.
        Includes permission checking if permission_manager is configured.
        """

        # Check permission before execution
        if self.permission_manager:
            result = await self.permission_manager.check_permission(func)
            # Handle both old (2 values) and new (3 values) return formats
            if len(result) == 3:
                allowed, modified_func, _ = result
            else:
                allowed, modified_func = result

            if not allowed:
                return FunctionOutput(
                    name=func.name,
                    input=func,
                    output=ToolOutput(
                        output="Tool execution cancelled by user",
                        observation="Tool execution cancelled by user",
                        display="Permission denied",
                        status="cancelled",
                    ),
                )

            # Use modified function if user edited it
            func = modified_func or func

        # Emit tool call event
        if streaming_result is not None:
            tool_call_item = ToolCallRunItem(data=func, id=func.id)
            tool_call_event = RunItemStreamEvent(
                name="agent.tool_call_start", item=tool_call_item
            )
            streaming_result.put_nowait(tool_call_event)
        
        # if streaming_result is not None:
        #     result = await self.agent.tool_manager.execute_func_astream(func=func)
        # else:
        result = await self.agent.tool_manager.execute_func_async(func=func)

        if not isinstance(result, FunctionOutput):
            raise ValueError("Result is not a FunctionOutput")
        return result

    async def stream_tool_execution(
        self,
        function: Function,
        tool_call_id: str,
        tool_call_name: str,
        streaming_result: RunnerStreamingResult,
    ) -> tuple[Any, Any, Any]:
        """
        Execute a tool/function call with streaming support and proper event handling.

        This method handles:
        - Tool span creation for tracing
        - Async generator support for streaming results
        - Tool activity events
        - Tool completion events
        - Error handling and observation extraction

        Args:
            function: The Function object to execute
            tool_call_id: Unique identifier for this tool call
            tool_call_name: Name of the tool being called
            streaming_result: Queue for streaming events

        Returns:
            tuple: (function_output, function_output_observation)
        """
        # Create tool span for streaming function execution
        with tool_span(
            tool_name=tool_call_name,
            function_name=function.name,  # TODO fix attributes
            function_args=function.args,
            function_kwargs=function.kwargs,
        ) as tool_span_instance:

            # TODO: inside of FunctionTool execution, it should ensure the types of async generator item
            # to be either ToolCallActivityRunItem or ToolOutput(maybe)
            # Call activity might be better designed

            function_result = await self._tool_execute_async(
                func=function, streaming_result=streaming_result
            )  # everything must be wrapped in FunctionOutput

            if not isinstance(function_result, FunctionOutput):
                raise ValueError(
                    f"Result must be wrapped in FunctionOutput, got {type(function_result)}"
                )

            function_output = function_result.output
            real_function_output = None

            # TODO: validate when the function is a generator

            if inspect.iscoroutine(function_output):
                real_function_output = await function_output
            elif inspect.isasyncgen(function_output):
                async for item in function_output:
                    if isinstance(item, ToolCallActivityRunItem):
                        # add the tool_call_id to the item
                        item.id = tool_call_id
                        tool_call_event = RunItemStreamEvent(
                            name="agent.tool_call_activity", item=item
                        )
                        streaming_result.put_nowait(tool_call_event)
                    else:
                        real_function_output = item

            elif inspect.isgenerator(function_output):
                for item in function_output:
                    if isinstance(item, ToolCallActivityRunItem):
                        # add the tool_call_id to the item
                        item.id = tool_call_id
                        tool_call_event = RunItemStreamEvent(
                            name="agent.tool_call_activity", item=item
                        )
                        streaming_result.put_nowait(tool_call_event)
                    else:
                        real_function_output = item
            else:
                real_function_output = function_output

            # create call complete
            call_complete_event = RunItemStreamEvent(
                name="agent.tool_call_complete",
                item=ToolOutputRunItem(
                    id=tool_call_id,
                    data=FunctionOutput(
                        name=function.name,
                        input=function,
                        output=real_function_output,
                    ),
                ),
            )
            streaming_result.put_nowait(call_complete_event)

            function_output = real_function_output
            function_output_observation = function_output

            if isinstance(function_output, ToolOutput) and hasattr(
                function_output, "observation"
            ):
                function_output_observation = (
                    function_output.observation
                )
            # Update tool span attributes using update_attributes for MLflow compatibility

            tool_span_instance.span_data.update_attributes(
                {"output_result": real_function_output}
            )

            return function_result, function_output, function_output_observation
