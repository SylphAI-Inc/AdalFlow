"""
AdalFlow tracing setup and providers module with OpenAI Agents SDK compatibility.

This module provides the core tracing infrastructure for AdalFlow, including trace providers
and multi-processor management following OpenAI Agents SDK patterns.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python/blob/main/src/agents/tracing/setup.py
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Dict, Union

from . import util
from .processor_interface import TracingProcessor
from .scope import Scope
from .spans import NoOpSpan, Span, SpanImpl, TSpanData
from .traces import NoOpTrace, Trace, TraceImpl


logger = logging.getLogger(__name__)


class SynchronousMultiTracingProcessor(TracingProcessor):
    """
    Forwards all calls to a list of TracingProcessors, in order of registration.
    """

    def __init__(self):
        # Using a tuple to avoid race conditions when iterating over processors
        self._processors: tuple[TracingProcessor, ...] = ()
        self._lock = threading.Lock()

    def add_tracing_processor(self, tracing_processor: TracingProcessor):
        """
        Add a processor to the list of processors. Each processor will receive all traces/spans.
        """
        with self._lock:
            self._processors += (tracing_processor,)

    def set_processors(self, processors: list[TracingProcessor]):
        """
        Set the list of processors. This will replace the current list of processors.
        """
        with self._lock:
            self._processors = tuple(processors)

    def on_trace_start(self, trace: Trace) -> None:
        """
        Called when a trace is started.
        """
        for processor in self._processors:
            processor.on_trace_start(trace)

    def on_trace_end(self, trace: Trace) -> None:
        """
        Called when a trace is finished.
        """
        for processor in self._processors:
            processor.on_trace_end(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        """
        Called when a span is started.
        """
        for processor in self._processors:
            processor.on_span_start(span)

    def on_span_end(self, span: Span[Any]) -> None:
        """
        Called when a span is finished.
        """
        for processor in self._processors:
            processor.on_span_end(span)

    def shutdown(self) -> None:
        """
        Called when the application stops.
        """
        for processor in self._processors:
            try:
                logger.debug(f"Shutting down trace processor {processor}")
            except (ValueError, OSError):
                # Logging system may be closed during shutdown
                pass
            processor.shutdown()

    def force_flush(self):
        """
        Force the processors to flush their buffers.
        """
        for processor in self._processors:
            processor.force_flush()


class TraceProvider:
    def __init__(self):
        self._multi_processor = SynchronousMultiTracingProcessor()
        # check if ADALFLOW_DISABLE_TRACING is set to true or 1
        # if disabled the provider will just return a NoOp Trace and Span
        self._disabled = os.environ.get("ADALFLOW_DISABLE_TRACING", "true").lower() in (
            "true",
            "1",
        )

    def register_processor(self, processor: TracingProcessor):
        """
        Add a processor to the list of processors. Each processor will receive all traces/spans.
        """
        self._multi_processor.add_tracing_processor(processor)

    def set_processors(self, processors: list[TracingProcessor]):
        """
        Set the list of processors. This will replace the current list of processors.
        """
        self._multi_processor.set_processors(processors)

    def get_current_trace(self) -> Optional[Trace]:
        """
        Returns the currently active trace, if any.
        """
        return Scope.get_current_trace()

    def get_current_span(self) -> Optional[Span[Any]]:
        """
        Returns the currently active span, if any.
        """
        return Scope.get_current_span()

    def set_disabled(self, disabled: bool) -> None:
        """
        Set whether tracing is disabled.
        """
        self._disabled = disabled

    def create_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
    ) -> Trace:
        """
        Create a new trace.
        """
        if self._disabled or disabled:
            logger.debug(f"Tracing is disabled. Not creating trace {name}")
            return NoOpTrace()

        trace_id = trace_id or util.gen_trace_id()

        logger.debug(f"Creating trace {name} with id {trace_id}")

        return TraceImpl(
            name=name,
            trace_id=trace_id,
            group_id=group_id,
            metadata=metadata,
            processor=self._multi_processor,
        )

    def create_span(
        self,
        span_data: TSpanData,
        span_id: Optional[str] = None,
        parent: Optional[Union[Trace, Span[Any]]] = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        """
        Create a new span.
        """
        if self._disabled or disabled:
            logger.debug(f"Tracing is disabled. Not creating span {span_data}")
            return NoOpSpan(span_data)

        if not parent:
            current_span = Scope.get_current_span()
            current_trace = Scope.get_current_trace()
            if current_trace is None:
                logger.error(
                    "No active trace. Make sure to start a trace with `trace()` first"
                    "Returning NoOpSpan."
                )
                return NoOpSpan(span_data)
            elif isinstance(current_trace, NoOpTrace) or isinstance(
                current_span, NoOpSpan
            ):
                logger.debug(
                    f"Parent {current_span} or {current_trace} is no-op, returning NoOpSpan"
                )
                return NoOpSpan(span_data)

            parent_id = current_span.span_id if current_span else None
            trace_id = current_trace.trace_id

        elif isinstance(parent, Trace):
            if isinstance(parent, NoOpTrace):
                logger.debug(f"Parent {parent} is no-op, returning NoOpSpan")
                return NoOpSpan(span_data)
            trace_id = parent.trace_id
            parent_id = None
        elif isinstance(parent, Span):
            if isinstance(parent, NoOpSpan):
                logger.debug(f"Parent {parent} is no-op, returning NoOpSpan")
                return NoOpSpan(span_data)
            parent_id = parent.span_id
            trace_id = parent.trace_id

        logger.debug(f"Creating span {span_data} with id {span_id}")

        return SpanImpl(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            processor=self._multi_processor,
            span_data=span_data,
        )

    def shutdown(self) -> None:
        if self._disabled:
            return

        try:
            try:
                logger.debug("Shutting down trace provider")
            except (ValueError, OSError):
                # Logging system may be closed during shutdown
                pass
            self._multi_processor.shutdown()
        except Exception as e:
            try:
                logger.error(f"Error shutting down trace provider: {e}")
            except (ValueError, OSError):
                # Logging system may be closed during shutdown
                pass


# Lazy initialization - provider will be created on first access
_GLOBAL_TRACE_PROVIDER = None


def get_global_trace_provider():
    """Get the global trace provider, creating it if necessary."""
    global _GLOBAL_TRACE_PROVIDER
    if _GLOBAL_TRACE_PROVIDER is None:
        _GLOBAL_TRACE_PROVIDER = TraceProvider()
    return _GLOBAL_TRACE_PROVIDER


# For backward compatibility, create a property-like access
class _GlobalProviderProxy:
    """Proxy to provide backward-compatible access to GLOBAL_TRACE_PROVIDER."""

    def __getattr__(self, name):
        return getattr(get_global_trace_provider(), name)

    def __setattr__(self, name, value):
        return setattr(get_global_trace_provider(), name, value)


GLOBAL_TRACE_PROVIDER = _GlobalProviderProxy()
