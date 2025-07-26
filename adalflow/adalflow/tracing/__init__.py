"""
AdalFlow tracing module with OpenAI Agents SDK compatible interface.

This module provides tracing capabilities for AdalFlow components following
the OpenAI Agents SDK patterns for maximum compatibility with existing
observability backends.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-python/tree/main/src/openai/agents
- OpenAI Tracing Interface: https://platform.openai.com/docs/guides/agents/tracing
"""

from typing import Optional

# Legacy generator tracing (existing)
from .generator_state_logger import GeneratorStateLogger
from .generator_call_logger import GeneratorCallLogger
from .decorators import trace_generator_states, trace_generator_call

# New OpenAI-compatible tracing interface
from .spans import Span, NoOpSpan, SpanImpl
from .traces import Trace, NoOpTrace, TraceImpl
from .processor_interface import TracingProcessor, TracingExporter
from .span_data import (
    SpanData,
    AdalFlowRunnerSpanData,
    AdalFlowGeneratorSpanData,
    AdalFlowToolSpanData,
    AdalFlowResponseSpanData,
)

# from .span_data import SpanData, GeneratorSpanData

from .create import (
    trace,
    runner_span,
    generator_span,
    tool_span,
    custom_span,
    response_span,
    step_span,
)
from .scope import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
    gen_span_id,
    gen_trace_id,
)
from .setup import GLOBAL_TRACE_PROVIDER
from .mlflow_integration import (
    enable_mlflow_local,
    enable_mlflow_local_with_server,
    get_mlflow_server_command,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions for managing trace processors
# ═══════════════════════════════════════════════════════════════════════════════


def set_trace_processors(processors):
    """Set the global trace processors."""
    GLOBAL_TRACE_PROVIDER.set_processors(processors)


def add_trace_processor(processor):
    """Add a trace processor to the global list."""
    GLOBAL_TRACE_PROVIDER.register_processor(processor)


def get_trace_processors():
    """Get the current list of trace processors."""
    # Return a copy of the processors from the multi-processor
    return list(GLOBAL_TRACE_PROVIDER._multi_processor._processors)


def set_tracing_disabled(disabled: bool) -> None:
    """Enable or disable tracing globally."""
    GLOBAL_TRACE_PROVIDER.set_disabled(disabled)


def is_tracing_disabled() -> bool:
    """Check if tracing is disabled."""
    return GLOBAL_TRACE_PROVIDER._disabled


def set_tracing_export_api_key(api_key: Optional[str]) -> None:
    """Set the API key for tracing export."""
    import os

    # Store in environment variable for compatibility
    if api_key:
        os.environ["ADALFLOW_TRACING_API_KEY"] = api_key
    elif "ADALFLOW_TRACING_API_KEY" in os.environ:
        del os.environ["ADALFLOW_TRACING_API_KEY"]


__all__ = [
    # Legacy generator tracing
    "trace_generator_states",
    "trace_generator_call",
    "GeneratorStateLogger",
    "GeneratorCallLogger",
    # Core classes
    "Span",
    "NoOpSpan",
    "SpanImpl",
    "Trace",
    "NoOpTrace",
    "TraceImpl",
    "TracingProcessor",
    "TracingExporter",
    # Span data classes
    "SpanData",
    "AdalFlowRunnerSpanData",
    "AdalFlowGeneratorSpanData",
    "AdalFlowToolSpanData",
    "AdalFlowResponseSpanData",
    # Span creation functions
    "trace",
    "runner_span",
    "generator_span",
    "tool_span",
    "custom_span",
    "response_span",
    "step_span",
    # Scope management
    "get_current_span",
    "get_current_trace",
    "set_current_span",
    "set_current_trace",
    "gen_span_id",
    "gen_trace_id",
    # Processor management
    "set_trace_processors",
    "add_trace_processor",
    "set_tracing_disabled",
    "set_tracing_export_api_key",
    "get_trace_processors",
    # MLflow integration
    "enable_mlflow_local",
    "enable_mlflow_local_with_server",
    "get_mlflow_server_command",
]
