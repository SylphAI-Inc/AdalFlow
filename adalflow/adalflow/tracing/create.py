"""
AdalFlow tracing creation functions compatible with OpenAI Agents SDK.

This module provides functions to create traces and spans that follow the OpenAI Agents SDK
patterns for maximum compatibility with existing tracing backends like MLflow.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python/blob/main/src/agents/tracing/create.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Dict, Union

logger = logging.getLogger(__name__)
from .setup import GLOBAL_TRACE_PROVIDER
from .span_data import (
    AdalFlowRunnerSpanData,
    AdalFlowGeneratorSpanData,
    AdalFlowToolSpanData,
    AdalFlowResponseSpanData,
    AdalFlowStepSpanData,
    CustomSpanData,
)
from .spans import Span
from .traces import Trace

if TYPE_CHECKING:
    pass


def trace(
    workflow_name: str,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    disabled: bool = False,
) -> Trace:
    """
    Create a new trace. The trace will not be started automatically; you should either use
    it as a context manager (`with trace(...):`) or call `trace.start()` + `trace.finish()`
    manually.

    In addition to the workflow name and optional grouping identifier, you can provide
    an arbitrary metadata dictionary to attach additional user-defined information to
    the trace.

    Args:
        workflow_name: The name of the logical app or workflow. For example, you might provide
            "code_bot" for a coding agent, or "customer_support_agent" for a customer support agent.
        trace_id: The ID of the trace. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_trace_id()` to generate a trace ID, to guarantee that IDs are
            correctly formatted.
        group_id: Optional grouping identifier to link multiple traces from the same conversation
            or process. For instance, you might use a chat thread ID.
        metadata: Optional dictionary of additional metadata to attach to the trace.
        disabled: If True, we will return a Trace but the Trace will not be recorded. This will
            not be checked if there's an existing trace and `even_if_trace_running` is True.

    Returns:
        The newly created trace object.
    """
    current_trace = GLOBAL_TRACE_PROVIDER.get_current_trace()
    if current_trace:
        logger.warning(
            "Trace already exists. Creating a new trace, but this is probably a mistake."
        )

    return GLOBAL_TRACE_PROVIDER.create_trace(
        name=workflow_name,
        trace_id=trace_id,
        group_id=group_id,
        metadata=metadata,
        disabled=disabled,
    )


def get_current_trace() -> Optional[Trace]:
    """Returns the currently active trace, if present."""
    return GLOBAL_TRACE_PROVIDER.get_current_trace()


def get_current_span() -> Optional[Span[Any]]:
    """Returns the currently active span, if present."""
    return GLOBAL_TRACE_PROVIDER.get_current_span()


# ══════════════════════════════════════════════════════════════════════════════
# General Span Creation Functions
# ══════════════════════════════════════════════════════════════════════════════


def custom_span(
    name: str,
    data: Optional[Dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    """Create a new custom span for general-purpose tracing.

    This span can be used for any custom tracing needs in AdalFlow workflows.
    Use this to track custom workflow steps, debug custom components,
    or add domain-specific tracing information.

    Args:
        name: The name of the custom span.
        data: Optional dictionary of custom data to attach to the span.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created custom span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=CustomSpanData(
            name=name,
            data=data or {},
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AdalFlow-Specific Span Creation Functions
# ══════════════════════════════════════════════════════════════════════════════


def runner_span(
    runner_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    steps_executed: Optional[int] = None,
    final_answer: Optional[str] = None,
    workflow_status: Optional[str] = None,
    execution_time: Optional[float] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[AdalFlowRunnerSpanData]:
    """Create a new AdalFlow runner span for tracing workflow execution.

    This span tracks the complete lifecycle of multi-step agent workflows in AdalFlow.
    Use this to monitor overall workflow performance, track success/failure rates,
    and analyze step completion patterns.

    Args:
        runner_id: The ID of the runner instance.
        max_steps: The maximum number of steps allowed in the workflow.
        steps_executed: The number of steps executed so far.
        final_answer: The final answer produced by the workflow.
        workflow_status: The current status of the workflow (e.g., "running", "completed", "failed").
        execution_time: The total execution time in seconds.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created runner span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AdalFlowRunnerSpanData(
            runner_id=runner_id,
            max_steps=max_steps,
            steps_executed=steps_executed,
            final_answer=final_answer,
            workflow_status=workflow_status,
            execution_time=execution_time,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def generator_span(
    generator_id: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    generator_state_logger: Optional[Any] = None,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
    prompt_template_with_keywords: Optional[str] = None,
    raw_response: Optional[str] = None,
    api_response: Optional[Any] = None,
    generation_time_in_seconds: Optional[float] = None,
    token_usage: Optional[Dict[str, int]] = None,
    final_response: Optional[Any] = None,
    api_kwargs: Optional[Dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[AdalFlowGeneratorSpanData]:
    """Create a new AdalFlow generator span for tracing LLM or model generation.

    This span tracks the execution of a generator, such as an LLM call, and captures
    all relevant context including inputs, outputs, and performance metrics.

    Args:
        generator_id: Unique identifier for the generator instance.
        model_kwargs: Parameters used to configure the model.
        generator_state_logger: Optional logger for generator state changes.
        prompt_kwargs: Input parameters and context for the generation.
        prompt_template_with_keywords: The rendered prompt template with keywords filled in.
        raw_response: The raw response from the generator.
        api_response: The processed response from the API.
        generation_time: Time taken for generation in seconds.
        token_usage: Dictionary tracking token usage statistics.
        final_response: The final response data after any post-processing.
        api_kwargs: The API kwargs used for the model call.
        span_id: Optional custom span ID. If not provided, one will be generated.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created generator span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AdalFlowGeneratorSpanData(
            generator_id=generator_id,
            model_kwargs=model_kwargs,
            generator_state_logger=generator_state_logger,
            prompt_kwargs=prompt_kwargs,
            prompt_template_with_keywords=prompt_template_with_keywords,
            raw_response=raw_response,
            api_response=api_response,
            generation_time_in_seconds=generation_time_in_seconds,
            token_usage=token_usage,
            final_response=final_response,
            api_kwargs=api_kwargs,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def tool_span(
    tool_name: Optional[str] = None,
    function_name: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
    function_args: Optional[Dict[str, Any]] = None,
    function_kwargs: Optional[Dict[str, Any]] = None,
    output_result: Optional[Any] = None,
    execution_time: Optional[float] = None,
    error_info: Optional[Dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[AdalFlowToolSpanData]:
    """Create a new AdalFlow tool span for tracing tool/function execution.

    This span tracks function calls, tool usage, and external integrations in AdalFlow.
    Use this to monitor tool performance and reliability, debug tool integration issues,
    track tool usage patterns, and analyze tool effectiveness in workflows.

    Args:
        tool_name: The name of the tool being executed.
        function_name: The specific function name being called.
        input_params: The input parameters passed to the tool/function (deprecated, use function_args/function_kwargs).
        function_args: The positional arguments passed to the function.
        function_kwargs: The keyword arguments passed to the function.
        output_result: The result returned by the tool/function.
        execution_time: The time taken for execution in seconds.
        error_info: Any error information if the tool execution failed.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created tool span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AdalFlowToolSpanData(
            tool_name=tool_name,
            function_name=function_name,
            input_params=input_params,
            function_args=function_args,
            function_kwargs=function_kwargs,
            output_result=output_result,
            execution_time=execution_time,
            error_info=error_info,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def response_span(
    result_type: Optional[str] = None,
    execution_metadata: Optional[Dict[str, Any]] = None,
    response: Optional[Any] = None,
    answer: Optional[Any] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[AdalFlowResponseSpanData]:
    """Create a new AdalFlow response span for tracking final workflow results.

    This span tracks the final result of AdalFlow workflows, capturing
    the output that gets returned to the user. Use this to monitor
    end-to-end workflow success rates, analyze result quality,
    and track response patterns.

    Args:
        final_result: The final result produced by the workflow.
        result_type: The type of the result (e.g., "string", "object", "error").
        execution_metadata: Additional metadata about the execution.
        response: The response object (for OpenAI SDK compatibility).
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created response span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AdalFlowResponseSpanData(
            result_type=result_type,
            execution_metadata=execution_metadata,
            response=response,
            answer=answer,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )


def step_span(
    step_number: Optional[int] = None,
    action_type: Optional[str] = None,
    observation: Optional[Any] = None,
    is_final: bool = False,
    tool_name: Optional[str] = None,
    tool_output: Optional[Any] = None,
    execution_time: Optional[float] = None,
    error_info: Optional[Dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent: Optional[Union[Trace, Span[Any]]] = None,
    disabled: bool = False,
) -> Span[AdalFlowStepSpanData]:
    """Create a new AdalFlow step span for tracking individual workflow steps.

    This span tracks individual steps within multi-step agent workflows in AdalFlow.
    Use this to monitor step-by-step execution, debug workflow logic,
    track step completion patterns, and analyze individual step performance.

    Args:
        step_number: The step number in the workflow sequence.
        action_type: The type of action being performed (e.g., "planning", "tool_execution").
        observation: The result or observation from this step.
        is_final: Whether this is the final step in the workflow.
        tool_name: The name of the tool being executed in this step.
        tool_output: The output from executing the tool in this step.
        execution_time: The time taken to execute this step in seconds.
        error_info: Any error information if the step failed.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created step span.
    """
    return GLOBAL_TRACE_PROVIDER.create_span(
        span_data=AdalFlowStepSpanData(
            step_number=step_number,
            action_type=action_type,
            observation=observation,
            is_final=is_final,
            tool_name=tool_name,
            tool_output=tool_output,
            execution_time=execution_time,
            error_info=error_info,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )
