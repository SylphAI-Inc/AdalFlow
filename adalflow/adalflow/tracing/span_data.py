from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Dict, List, Union

if TYPE_CHECKING:
    from openai.types.responses import Response, ResponseInputItemParam

"""
Classes directly from OpenAI Agent Library SpanData

References:
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python/blob/main/src/agents/tracing/span_data.py
"""


class SpanData(abc.ABC):
    """
    Represents span data in the trace.
    """

    @abc.abstractmethod
    def export(self) -> Dict[str, Any]:
        """Export the span data as a dictionary."""
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Return the type of the span."""
        pass


class AgentSpanData(SpanData):
    """
    Represents an Agent Span in the trace.
    Includes name, handoffs, tools, and output type.
    """

    __slots__ = ("name", "handoffs", "tools", "output_type")

    def __init__(
        self,
        name: str,
        handoffs: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        output_type: Optional[str] = None,
    ):
        self.name = name
        self.handoffs: Optional[List[str]] = handoffs
        self.tools: Optional[List[str]] = tools
        self.output_type: Optional[str] = output_type

    @property
    def type(self) -> str:
        return "agent"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "handoffs": self.handoffs,
            "tools": self.tools,
            "output_type": self.output_type,
        }


class FunctionSpanData(SpanData):
    """
    Represents a Function Span in the trace.
    Includes input, output and MCP data (if applicable).
    """

    __slots__ = ("name", "input", "output", "mcp_data")

    def __init__(
        self,
        name: str,
        input: Optional[str],
        output: Optional[Any],
        mcp_data: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.mcp_data = mcp_data

    @property
    def type(self) -> str:
        return "function"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": str(self.output) if self.output else None,
            "mcp_data": self.mcp_data,
        }


class GenerationSpanData(SpanData):
    """
    Represents a Generation Span in the trace.
    Includes input, output, model, model configuration, and usage.
    """

    __slots__ = (
        "input",
        "output",
        "model",
        "model_config",
        "usage",
    )

    def __init__(
        self,
        input: Optional[Sequence[Mapping[str, Any]]] = None,
        output: Optional[Sequence[Mapping[str, Any]]] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        self.input = input
        self.output = output
        self.model = model
        self.model_config = model_config
        self.usage = usage

    @property
    def type(self) -> str:
        return "generation"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": self.output,
            "model": self.model,
            "model_config": self.model_config,
            "usage": self.usage,
        }


class ResponseSpanData(SpanData):
    """
    Represents a Response Span in the trace.
    Includes response and input.
    """

    __slots__ = ("response", "input")

    def __init__(
        self,
        response: Optional[Response] = None,
        input: Optional[Union[str, List[ResponseInputItemParam]]] = None,
    ) -> None:
        self.response = response
        # This is not used by the OpenAI trace processors, but is useful for other tracing
        # processor implementations
        self.input = input

    @property
    def type(self) -> str:
        return "response"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "response_id": self.response.id if self.response else None,
        }


class HandoffSpanData(SpanData):
    """
    Represents a Handoff Span in the trace.
    Includes source and destination agents.
    """

    __slots__ = ("from_agent", "to_agent")

    def __init__(self, from_agent: Optional[str], to_agent: Optional[str]):
        self.from_agent = from_agent
        self.to_agent = to_agent

    @property
    def type(self) -> str:
        return "handoff"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
        }


class CustomSpanData(SpanData):
    """
    Represents a Custom Span in the trace.
    Includes name and data property bag.
    """

    __slots__ = ("name", "data")

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.data = data

    @property
    def type(self) -> str:
        return "custom"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
        }


class GuardrailSpanData(SpanData):
    """
    Represents a Guardrail Span in the trace.
    Includes name and triggered status.
    """

    __slots__ = ("name", "triggered")

    def __init__(self, name: str, triggered: bool = False):
        self.name = name
        self.triggered = triggered

    @property
    def type(self) -> str:
        return "guardrail"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "triggered": self.triggered,
        }


class TranscriptionSpanData(SpanData):
    """
    Represents a Transcription Span in the trace.
    Includes input, output, model, and model configuration.
    """

    __slots__ = (
        "input",
        "input_format",
        "output",
        "model",
        "model_config",
    )

    def __init__(
        self,
        input: Optional[str] = None,
        input_format: Optional[str] = "pcm",
        output: Optional[str] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
    ):
        self.input = input
        self.input_format = input_format
        self.output = output
        self.model = model
        self.model_config = model_config

    @property
    def type(self) -> str:
        return "transcription"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "input": {
                "data": self.input or "",
                "format": self.input_format,
            },
            "output": self.output,
            "model": self.model,
            "model_config": self.model_config,
        }


class SpeechSpanData(SpanData):
    """
    Represents a Speech Span in the trace.
    Includes input, output, model, model configuration, and first content timestamp.
    """

    __slots__ = (
        "input",
        "output",
        "output_format",
        "model",
        "model_config",
        "first_content_at",
    )

    def __init__(
        self,
        input: Optional[str] = None,
        output: Optional[str] = None,
        output_format: Optional[str] = "pcm",
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        first_content_at: Optional[str] = None,
    ):
        self.input = input
        self.output = output
        self.output_format = output_format
        self.model = model
        self.model_config = model_config
        self.first_content_at = first_content_at

    @property
    def type(self) -> str:
        return "speech"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": {
                "data": self.output or "",
                "format": self.output_format,
            },
            "model": self.model,
            "model_config": self.model_config,
            "first_content_at": self.first_content_at,
        }


class SpeechGroupSpanData(SpanData):
    """
    Represents a Speech Group Span in the trace.
    """

    __slots__ = "input"

    def __init__(
        self,
        input: Optional[str] = None,
    ):
        self.input = input

    @property
    def type(self) -> str:
        return "speech_group"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
        }


class MCPListToolsSpanData(SpanData):
    """
    Represents an MCP List Tools Span in the trace.
    Includes server and result.
    """

    __slots__ = (
        "server",
        "result",
    )

    def __init__(
        self, server: Optional[str] = None, result: Optional[List[str]] = None
    ):
        self.server = server
        self.result = result

    @property
    def type(self) -> str:
        return "mcp_tools"

    def export(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "server": self.server,
            "result": self.result,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AdalFlow-Specific Span Data Classes (mapped to OpenAI format)
# ══════════════════════════════════════════════════════════════════════════════

"""
These classes are either created using custom spans of the OpenAI agents span data types
or are subclasses

References:
- OpenAI Agents SDK CustomSpanData: https://github.com/openai/openai-python/blob/main/src/openai/agents/spans.py
- MLflow OpenAI Trace Processor: https://mlflow.org/docs/latest/llms/openai/index.html
"""


class AdalFlowRunnerSpanData(CustomSpanData):
    """
    Represents the top-level Runner execution span in AdalFlow.
    Tracks complete lifecycle of multi-step agent workflows.
    Maps to OpenAI's AgentSpanData format.

    The caller should update span attributes by using update_attributes() method.
    """

    __slots__ = (
        "runner_id",
        "max_steps",
        "steps_executed",
        "final_answer",
        "workflow_status",
        "execution_time",
        "data",
    )

    def __init__(
        self,
        runner_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        steps_executed: Optional[int] = None,
        final_answer: Optional[str] = None,
        workflow_status: Optional[str] = None,
        execution_time: Optional[float] = None,
    ):
        # Initialize with data that will be used by MLflow
        self.data = {
            "runner_id": runner_id,
            "max_steps": max_steps,
            "steps_executed": steps_executed,
            "final_answer": final_answer,
            "workflow_status": workflow_status,
            "execution_time": execution_time,
        }
        super().__init__(
            name=f"AdalFlow-Runner-{runner_id}" if runner_id else "AdalFlow-Runner",
            data=self.data,
        )

        self.runner_id = runner_id
        self.max_steps = max_steps
        self.steps_executed = steps_executed
        self.final_answer = final_answer
        self.workflow_status = workflow_status
        self.execution_time = execution_time

    def export(self) -> Dict[str, Any]:
        base_export = super().export()
        return base_export

    def update_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Update span attributes directly for MLflow compatibility.
        Note: MLflow's OpenAI Trace processor reads attributes directly from the data attribute
        instead of using export(). This is a known limitation of the current MLflow implementation.
        """
        # Update instance attributes using setattr for slots
        for key, value in attributes.items():
            if key in self.__slots__:
                setattr(self, key, value)

            # Update the custom span's data attribute which is exported
            self.data[key] = value


class AdalFlowGeneratorSpanData(CustomSpanData):
    """
    Represents Generator execution spans in AdalFlow.
    Tracks LLM generation, processing, and output formatting using generator_state_logger.
    Maps to OpenAI's CustomSpanData format.

    The caller should update span attributes by using update_attributes() method.
    """

    __slots__ = (
        "generator_id",
        "generator_state_logger",
        "model_kwargs",
        "api_kwargs",
        "prompt_kwargs",
        "prompt_template_with_keywords",
        "raw_response",
        "api_response",
        "generation_time_in_seconds",
        "token_usage",
        "final_response",  # renamed data as there the custom span data already has a data in __slots__
        "data",
    )

    def __init__(
        self,
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
    ):
        # Initialize with data that will be used by MLflow
        self.data = {
            "generator_id": generator_id,
            "model_kwargs": model_kwargs,
            "prompt_kwargs": prompt_kwargs,
            "api_kwargs": api_kwargs,
            "prompt_template_with_keywords": prompt_template_with_keywords,
            "raw_response": raw_response,
            "api_response": api_response,
            "generation_time_in_seconds": generation_time_in_seconds,
            "token_usage": token_usage,
            "final_response": final_response,
        }

        super().__init__(
            name=(
                f"AdalFlow-Generator-{generator_id}"
                if generator_id
                else "AdalFlow-Generator"
            ),
            data=self.data,
        )

        self.generator_id = generator_id
        self.model_kwargs = model_kwargs
        self.generator_state_logger = generator_state_logger
        self.prompt_kwargs = prompt_kwargs
        self.prompt_template_with_keywords = prompt_template_with_keywords
        self.raw_response = raw_response
        self.api_response = api_response
        self.generation_time_in_seconds = generation_time_in_seconds
        self.token_usage = token_usage
        self.final_response = final_response
        self.api_kwargs = api_kwargs

    def update_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Update span attributes directly for MLflow compatibility.
        Note: MLflow's OpenAI Trace processor reads attributes directly from the data attribute
        instead of using export(). This is a known limitation of the current MLflow implementation.
        """
        # 1) Dump everything into the object’s namespace
        for key, value in attributes.items():
            if key in self.__slots__:
                setattr(self, key, value)

            # Update the custom span's data attribute which is exported
            self.data[key] = value

    def export(self) -> Dict[str, Any]:
        base_export = super().export()
        return base_export


class AdalFlowResponseSpanData(CustomSpanData):
    """
    Represents AdalFlow Response spans that track the final result.
    Based on CustomSpanData but specialized for AdalFlow final outputs.


    The caller should update span attributes by using update_attributes() method.
    """

    __slots__ = (
        "answer",
        "result_type",
        "execution_metadata",
        "response",
        "data",
    )

    def __init__(
        self,
        answer: Optional[Any] = None,
        result_type: Optional[str] = None,
        execution_metadata: Optional[Dict[str, Any]] = None,
        response: Optional[Any] = None,
    ):
        # Create data dict for CustomSpanData
        self.data = {
            "answer": answer,
            "result_type": result_type,
            "execution_metadata": execution_metadata,
            "response": response,
        }
        super().__init__(name="response", data=self.data)

        self.answer = answer
        self.result_type = result_type
        self.execution_metadata = execution_metadata
        self.response = response

    def update_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Update span attributes directly for MLflow compatibility.
        Note: MLflow's OpenAI Trace processor reads attributes directly from the data attribute
        instead of using export(). This is a known limitation of the current MLflow implementation.
        """
        # 1) Dump everything into the object’s namespace
        for key, value in attributes.items():
            if key in self.__slots__:
                setattr(self, key, value)

            # Update the custom span's data attribute which is exported
            self.data[key] = value

    def export(self) -> Dict[str, Any]:
        base_export = super().export()
        return base_export


class AdalFlowStepSpanData(CustomSpanData):
    """
    Represents a Step Span in AdalFlow workflow execution.
    Tracks individual steps within a multi-step agent workflow.

    The caller should update span attributes by using update_attributes() method.
    """

    __slots__ = (
        "step_number",
        "action_type",
        "observation",
        "is_final",
        "tool_name",
        "tool_output",
        "execution_time",
        "error_info",
        "data",
    )

    def __init__(
        self,
        step_number: Optional[int] = None,
        action_type: Optional[str] = None,
        observation: Optional[Any] = None,
        is_final: bool = False,
        tool_name: Optional[str] = None,
        tool_output: Optional[Any] = None,
        execution_time: Optional[float] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ):
        # Create data dictionary for CustomSpanData
        self.data = {
            "step_number": step_number,
            "action_type": action_type,
            "observation": observation,
            "is_final": is_final,
            "tool_name": tool_name,
            "tool_output": tool_output,
            "execution_time": execution_time,
            "error_info": error_info,
        }

        step_name = f"step-{step_number}" if step_number is not None else "step"
        super().__init__(name=step_name, data=self.data)

        # Set individual attributes
        self.step_number = step_number
        self.action_type = action_type
        self.observation = observation
        self.is_final = is_final
        self.tool_name = tool_name
        self.tool_output = tool_output
        self.execution_time = execution_time
        self.error_info = error_info

    def update_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Update span attributes directly for MLflow compatibility.
        Note: MLflow's OpenAI Trace processor reads attributes directly from the data attribute
        instead of using export(). This is a known limitation of the current MLflow implementation.
        """
        # 1) Dump everything into the object’s namespace
        for key, value in attributes.items():
            if key in self.__slots__:
                setattr(self, key, value)

            # Update the custom span's data attribute which is exported
            self.data[key] = value

    def export(self) -> Dict[str, Any]:
        base_export = super().export()
        return base_export


class AdalFlowToolSpanData(CustomSpanData):
    """
    Represents Tool execution spans in AdalFlow.
    Tracks function calls, tool usage, and external integrations.
    Inherits OpenAI's CustomSpanData format.

    The caller should update span attributes by using update_attributes() method.
    """

    __slots__ = (
        "tool_name",
        "function_name",
        "input_params",
        "function_args",
        "function_kwargs",
        "output_result",
        "execution_time",
        "error_info",
        "data",
    )

    def __init__(
        self,
        tool_name: Optional[str] = None,
        function_name: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        function_args: Optional[Dict[str, Any]] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        output_result: Optional[Any] = None,
        execution_time: Optional[float] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ):
        # Map to OpenAI function format
        function_display_name = (
            f"{tool_name}"
            if tool_name 
            else (tool_name or function_name or "unknown")
        )

        # Create data dictionary for CustomSpanData
        self.data = {
            "tool_name": tool_name,
            "function_name": function_name,
            "input_params": str(input_params) if input_params else None,
            "function_args": str(function_args) if function_args else None,
            "function_kwargs": str(function_kwargs) if function_kwargs else None,
            "output_result": str(output_result) if output_result else None,
            # "execution_time": execution_time,
            "error_info": error_info,
        }

        super().__init__(name=function_display_name, data=self.data)

        self.tool_name = tool_name
        self.function_name = function_name
        self.input_params = input_params
        self.function_args = function_args
        self.function_kwargs = function_kwargs
        self.output_result = output_result
        self.execution_time = execution_time
        self.error_info = error_info

    def update_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Update span attributes directly for MLflow compatibility.
        Note: MLflow's OpenAI Trace processor reads attributes directly from the data attribute
        instead of using export(). This is a known limitation of the current MLflow implementation.
        """
        # 1) Dump everything into the object’s namespace
        for key, value in attributes.items():
            if key in self.__slots__:
                setattr(self, key, value)

            # Update the custom span's data attribute which is exported
            self.data[key] = value

    def export(self) -> Dict[str, Any]:
        base_export = super().export()
        return base_export
