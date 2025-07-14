"""
AdalFlow tracing scope management module with OpenAI Agents SDK compatibility.

This module provides context management for traces and spans, maintaining current
execution context following OpenAI Agents SDK patterns.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python/blob/main/src/agents/tracing/scope.py
"""

# Holds the current active span
import contextvars
import logging
import uuid
from typing import Any, Optional
from .traces import Trace
from .spans import Span

logger = logging.getLogger(__name__)

_current_span: contextvars.ContextVar[Optional["Span[Any]"]] = contextvars.ContextVar(
    "current_span", default=None
)

_current_trace: contextvars.ContextVar[Optional["Trace"]] = contextvars.ContextVar(
    "current_trace", default=None
)


class Scope:
    """
    Manages the current span and trace in the context.
    """

    @classmethod
    def get_current_span(cls) -> Optional["Span[Any]"]:
        return _current_span.get()

    @classmethod
    def set_current_span(
        cls, span: Optional["Span[Any]"]
    ) -> "contextvars.Token[Optional[Span[Any]]]":
        return _current_span.set(span)

    @classmethod
    def reset_current_span(
        cls, token: "contextvars.Token[Optional[Span[Any]]]"
    ) -> None:
        _current_span.reset(token)

    @classmethod
    def get_current_trace(cls) -> Optional["Trace"]:
        return _current_trace.get()

    @classmethod
    def set_current_trace(
        cls, trace: Optional["Trace"]
    ) -> "contextvars.Token[Optional[Trace]]":
        logger.debug(f"Setting current trace: {trace.trace_id if trace else None}")
        return _current_trace.set(trace)

    @classmethod
    def reset_current_trace(cls, token: "contextvars.Token[Optional[Trace]]") -> None:
        logger.debug("Resetting current trace")
        _current_trace.reset(token)


# Utility functions for standalone use
def get_current_span() -> Optional["Span[Any]"]:
    """Get the current span from context."""
    return Scope.get_current_span()


def set_current_span(
    span: Optional["Span[Any]"],
) -> "contextvars.Token[Optional[Span[Any]]]":
    """Set the current span in context."""
    return Scope.set_current_span(span)


def get_current_trace() -> Optional["Trace"]:
    """Get the current trace from context."""
    return Scope.get_current_trace()


def set_current_trace(trace: Optional["Trace"]) -> "contextvars.Token[Optional[Trace]]":
    """Set the current trace in context."""
    return Scope.set_current_trace(trace)


def gen_span_id() -> str:
    """Generate a unique span ID."""
    return str(uuid.uuid4())


def gen_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())
