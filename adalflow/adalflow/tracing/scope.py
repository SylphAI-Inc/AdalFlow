"""
AdalFlow tracing scope management module with OpenAI Agents SDK compatibility.

This module provides context management for traces and spans, maintaining current
execution context following OpenAI Agents SDK patterns.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-python/tree/main/src/openai/agents
- OpenAI Tracing Interface: https://platform.openai.com/docs/guides/agents/tracing
"""

# Holds the current active span
import contextvars
import logging
import uuid
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .spans import Span
    from .traces import Trace

_current_span: contextvars.ContextVar["Span[Any] | None"] = contextvars.ContextVar(
    "current_span", default=None
)

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)


class Scope:
    """
    Manages the current span and trace in the context.
    """

    @classmethod
    def get_current_span(cls) -> "Span[Any] | None":
        return _current_span.get()

    @classmethod
    def set_current_span(
        cls, span: "Span[Any] | None"
    ) -> "contextvars.Token[Span[Any] | None]":
        return _current_span.set(span)

    @classmethod
    def reset_current_span(cls, token: "contextvars.Token[Span[Any] | None]") -> None:
        _current_span.reset(token)

    @classmethod
    def get_current_trace(cls) -> "Trace | None":
        return _current_trace.get()

    @classmethod
    def set_current_trace(
        cls, trace: "Trace | None"
    ) -> "contextvars.Token[Trace | None]":
        logger.debug(f"Setting current trace: {trace.trace_id if trace else None}")
        return _current_trace.set(trace)

    @classmethod
    def reset_current_trace(cls, token: "contextvars.Token[Trace | None]") -> None:
        logger.debug("Resetting current trace")
        _current_trace.reset(token)


# Utility functions for standalone use
def get_current_span() -> "Span[Any] | None":
    """Get the current span from context."""
    return Scope.get_current_span()


def set_current_span(span: "Span[Any] | None") -> "contextvars.Token[Span[Any] | None]":
    """Set the current span in context."""
    return Scope.set_current_span(span)


def get_current_trace() -> "Trace | None":
    """Get the current trace from context."""
    return Scope.get_current_trace()


def set_current_trace(trace: "Trace | None") -> "contextvars.Token[Trace | None]":
    """Set the current trace in context."""
    return Scope.set_current_trace(trace)


def gen_span_id() -> str:
    """Generate a unique span ID."""
    return str(uuid.uuid4())


def gen_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())
