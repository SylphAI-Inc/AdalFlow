"""
AdalFlow spans module with OpenAI Agents SDK compatible interface.

This module provides span implementations for AdalFlow tracing that follow
the OpenAI Agents SDK patterns for maximum compatibility with existing
observability backends.

References:
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python/blob/main/src/agents/tracing/spans.py
"""

from __future__ import annotations

import abc
import contextvars
import logging
from typing import Any, Generic, TypeVar, Optional, Dict

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

from . import util
from .processor_interface import TracingProcessor
from .span_data import SpanData

TSpanData = TypeVar("TSpanData", bound=SpanData)


class SpanError(TypedDict):
    message: str
    data: Optional[Dict[str, Any]]


class Span(abc.ABC, Generic[TSpanData]):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def span_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData:
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """
        Start the span.

        Args:
            mark_as_current: If true, the span will be marked as the current span.
        """
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        """
        Finish the span.

        Args:
            reset_current: If true, the span will be reset as the current span.
        """
        pass

    @abc.abstractmethod
    def __enter__(self) -> Span[TSpanData]:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abc.abstractmethod
    def parent_id(self) -> Optional[str]:
        pass

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None:
        pass

    @property
    @abc.abstractmethod
    def error(self) -> Optional[SpanError]:
        pass

    @abc.abstractmethod
    def export(self) -> Optional[Dict[str, Any]]:
        pass

    @property
    @abc.abstractmethod
    def started_at(self) -> Optional[str]:
        pass

    @property
    @abc.abstractmethod
    def ended_at(self) -> Optional[str]:
        pass


class NoOpSpan(Span[TSpanData]):
    __slots__ = ("_span_data", "_prev_span_token")

    def __init__(self, span_data: TSpanData):
        self._span_data = span_data
        self._prev_span_token: Optional[
            contextvars.Token[Optional[Span[TSpanData]]]
        ] = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def span_id(self) -> str:
        return "no-op"

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> Optional[str]:
        return None

    def start(self, mark_as_current: bool = False):
        from .scope import Scope

        if mark_as_current:
            self._prev_span_token = Scope.set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        from .scope import Scope

        if reset_current and self._prev_span_token is not None:
            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Span[TSpanData]:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_current = True
        if exc_type is GeneratorExit:
            logger.debug("GeneratorExit, skipping span reset")
            reset_current = False

        self.finish(reset_current=reset_current)

    def set_error(self, error: SpanError) -> None:
        pass

    @property
    def error(self) -> Optional[SpanError]:
        return None

    def export(self) -> Optional[Dict[str, Any]]:
        return None

    @property
    def started_at(self) -> Optional[str]:
        return None

    @property
    def ended_at(self) -> Optional[str]:
        return None


class SpanImpl(Span[TSpanData]):
    __slots__ = (
        "_trace_id",
        "_span_id",
        "_parent_id",
        "_started_at",
        "_ended_at",
        "_error",
        "_prev_span_token",
        "_processor",
        "_span_data",
    )

    def __init__(
        self,
        trace_id: str,
        span_id: Optional[str],
        parent_id: Optional[str],
        processor: TracingProcessor,
        span_data: TSpanData,
    ):
        self._trace_id = trace_id
        self._span_id = span_id or util.gen_span_id()
        self._parent_id = parent_id
        self._started_at: Optional[str] = None
        self._ended_at: Optional[str] = None
        self._processor = processor
        self._error: Optional[SpanError] = None
        self._prev_span_token: Optional[
            contextvars.Token[Optional[Span[TSpanData]]]
        ] = None
        self._span_data = span_data

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def parent_id(self) -> Optional[str]:
        return self._parent_id

    def start(self, mark_as_current: bool = False):
        from .scope import Scope

        if self.started_at is not None:
            logger.warning("Span already started")
            return

        self._started_at = util.time_iso()
        self._processor.on_span_start(self)
        if mark_as_current:
            self._prev_span_token = Scope.set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        from .scope import Scope

        if self.ended_at is not None:
            logger.warning("Span already finished")
            return

        self._ended_at = util.time_iso()
        self._processor.on_span_end(self)
        if reset_current and self._prev_span_token is not None:
            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def __enter__(self) -> Span[TSpanData]:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_current = True
        if exc_type is GeneratorExit:
            logger.debug("GeneratorExit, skipping span reset")
            reset_current = False

        self.finish(reset_current=reset_current)

    def set_error(self, error: SpanError) -> None:
        self._error = error

    @property
    def error(self) -> Optional[SpanError]:
        return self._error

    @property
    def started_at(self) -> Optional[str]:
        return self._started_at

    @property
    def ended_at(self) -> Optional[str]:
        return self._ended_at

    def export(self) -> Optional[Dict[str, Any]]:
        return {
            "object": "trace.span",
            "id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self._parent_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "span_data": self.span_data.export(),
            "error": self._error,
        }
