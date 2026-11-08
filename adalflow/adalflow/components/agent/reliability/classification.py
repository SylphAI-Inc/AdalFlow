"""Structured error classification for the Runner reliability layer.

Replaces the old substring-matching ``_is_unrecoverable_error`` helper with a
real classifier that understands HTTP status codes, ``Retry-After`` headers, the
asyncio cancellation / timeout taxonomy, and the cause/context chain of
chained exceptions.

The classifier never imports any HTTP client library; it reads attributes
(``status_code`` / ``status`` / ``http_status`` / ``response.status_code`` and
``response.headers["Retry-After"]``) defensively so it works against OpenAI,
Anthropic, httpx and requests style exceptions without a hard dependency.
"""

from __future__ import annotations

import asyncio
import email.utils
import errno
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


class ErrorCategory(str, Enum):
    """Coarse-grained bucket an error falls into.

    The string values are stable so they can be persisted in checkpoints and
    compared across versions.
    """

    TRANSIENT = "transient"  # network blip, 5xx, connection reset
    RATE_LIMITED = "rate_limited"  # 429
    TIMEOUT = "timeout"  # asyncio.TimeoutError, our own timeout errors
    CONTEXT_OVERFLOW = "context_overflow"  # 400 + context phrase
    INVALID_REQUEST = "invalid_request"  # other 400 / 422
    AUTH = "auth"  # 401 / 403
    NOT_FOUND = "not_found"  # 404
    PARSE = "parse"  # JSON / output parser failures
    TOOL_ERROR = "tool_error"  # tool raised inside its body
    CANCELLED = "cancelled"  # asyncio.CancelledError - never retryable
    UNKNOWN = "unknown"


# Categories that the Runner treats as "stop the run, do not retry".
_UNRECOVERABLE_CATEGORIES = frozenset(
    {
        ErrorCategory.CONTEXT_OVERFLOW,
        ErrorCategory.INVALID_REQUEST,
        ErrorCategory.AUTH,
        ErrorCategory.NOT_FOUND,
    }
)


@dataclass
class ClassifiedError:
    """Result of classifying an error.

    ``retryable`` is the classifier's opinion; the :class:`RetryPolicy` makes the
    final decision based on its own ``retry_on`` set.
    """

    category: ErrorCategory
    retryable: bool
    status_code: int | None = None
    retry_after: float | None = None  # seconds
    message: str = ""
    exception_type: str | None = None


# A user rule is (predicate, category, retryable).  Predicate receives the raw
# error/str and returns True if it matches.
UserRule = tuple[Callable[[Any], bool], ErrorCategory, bool]


# ---------------------------------------------------------------------------
# Regex patterns for string classification (GeneratorOutput.error is a str)
# ---------------------------------------------------------------------------
_STATUS_IN_TEXT_RE = re.compile(
    r"\b(?:status|code|error|http)[^\d]{0,12}(\d{3})\b",
    re.IGNORECASE,
)
_ERROR_CODE_RE = re.compile(r"Error code:\s*(\d{3})")
_CONTEXT_OVERFLOW_PHRASES = (
    "context length",
    "maximum context",
    "too many tokens",
    "context window",
    "context is too long",
)
_TRANSIENT_PHRASES = (
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "try again",
)
# Timeout phrases are mapped to the TIMEOUT category, not TRANSIENT, so they do
# not shadow the dedicated timeout handling (and stay retryable per the default
# ``retry_on`` set which includes TIMEOUT).
_TIMEOUT_PHRASES = (
    "timed out",
    "timeout",
)
# Note: bare digit strings (e.g. "429") are intentionally NOT included here.
# Substring digit matching is exactly the false-positive bug we are fixing
# ("Processed 1400 rows" must classify as UNKNOWN); status codes are read
# from the structured HTTP-ish context via the regexes above instead.
_RATE_LIMIT_PHRASES = (
    "rate limit",
    "rate limited",
    "too many requests",
)

# OSError errnos that represent genuine network failures and therefore qualify
# as TRANSIENT.  All other OSErrors (FileNotFoundError, PermissionError, ...) are
# not network-related and must not be auto-classified as transient.
_NETWORK_ERRNOS = frozenset(
    {
        errno.ECONNRESET,
        errno.ECONNREFUSED,
        errno.ECONNABORTED,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.EPIPE,
    }
)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value (seconds or HTTP-date)."""
    if not value:
        return None
    value = value.strip()
    # Try integer/float seconds first.
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    # Fall back to HTTP-date.
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt is None:
            return None
        import datetime as _dt

        delta = (dt - _dt.datetime.now(tz=dt.tzinfo)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def _status_to_category(status: int) -> tuple[ErrorCategory, bool]:
    """Map an HTTP status code to (category, retryable)."""
    if status in (408, 409, 425, 500, 502, 503, 504):
        return ErrorCategory.TRANSIENT, True
    if status == 429:
        return ErrorCategory.RATE_LIMITED, True
    if status in (401, 403):
        return ErrorCategory.AUTH, False
    if status == 404:
        return ErrorCategory.NOT_FOUND, False
    if status in (400, 422):
        return ErrorCategory.INVALID_REQUEST, False
    # Unknown but probably-server status: be optimistic.
    if 500 <= status < 600:
        return ErrorCategory.TRANSIENT, True
    return ErrorCategory.UNKNOWN, False


class ErrorClassifier:
    """Classify exceptions and error strings into a :class:`ClassifiedError`.

    User rules registered via :meth:`register` are evaluated first and win over
    the built-in heuristics.
    """

    def __init__(self) -> None:
        self._user_rules: list[UserRule] = []

    # ------------------------------------------------------------------ public
    def register(
        self,
        predicate: Callable[[Any], bool],
        category: ErrorCategory,
        retryable: bool,
    ) -> None:
        """Register a user rule evaluated before the built-in heuristics."""
        self._user_rules.append((predicate, category, retryable))

    def classify(self, error: Union[BaseException, str, None]) -> ClassifiedError:
        """Classify an exception, error string, or ``None``."""
        if error is None:
            return ClassifiedError(
                category=ErrorCategory.UNKNOWN,
                retryable=False,
                message="",
            )
        # User rules first.
        for predicate, category, retryable in self._user_rules:
            try:
                if predicate(error):
                    return self._build(category, retryable, error)
            except Exception:  # pragma: no cover - never let a buggy rule crash us
                continue
        if isinstance(error, BaseException):
            return self._classify_exception(error)
        return self._classify_string(str(error))

    # --------------------------------------------------------------- internals
    def _build(
        self,
        category: ErrorCategory,
        retryable: bool,
        error: Any,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> ClassifiedError:
        message = str(error) if not isinstance(error, str) else error
        exc_type: str | None = None
        if isinstance(error, BaseException):
            exc_type = type(error).__name__
        return ClassifiedError(
            category=category,
            retryable=retryable,
            status_code=status_code,
            retry_after=retry_after,
            message=message,
            exception_type=exc_type,
        )

    # ------------------------------------------------------- exception handling
    def _classify_exception(self, error: BaseException) -> ClassifiedError:
        # Walk the cause/context chain (bounded depth) looking for a typed or
        # status-code signal BEFORE consulting any message text.  An outer
        # generic exception (e.g. ``ValueError("tool failed")``) must not be
        # classified via its own message as UNKNOWN while a deeper cause
        # carries the real signal.  Only if no exception in the chain yields
        # a specific classification do we fall back to string classification
        # of the OUTERMOST exception's message.
        seen: set = set()
        current: BaseException | None = error
        depth = 0
        while current is not None and depth < 8 and id(current) not in seen:
            seen.add(id(current))
            classified = self._classify_single_exception(current)
            if classified is not None:
                return classified
            current = current.__cause__ or current.__context__
            depth += 1
        # Fall back to string classification of the outermost message.
        return self._classify_string(str(error), exception=error)

    def _classify_single_exception(
        self, error: BaseException
    ) -> ClassifiedError | None:
        # Cancellation must never be retried.
        if isinstance(error, asyncio.CancelledError):
            return self._build(ErrorCategory.CANCELLED, False, error)
        # Timeouts.
        if isinstance(error, asyncio.TimeoutError) or isinstance(error, TimeoutError):
            return self._build(ErrorCategory.TIMEOUT, True, error)
        # Our own timeout errors (avoid a hard import cycle).
        cls_name = type(error).__name__
        if cls_name in {
            "StepTimeoutError",
            "ToolTimeoutError",
            "RunDeadlineExceededError",
        }:
            return self._build(ErrorCategory.TIMEOUT, False, error)
        # Network errors.  ``OSError`` is a superclass of many non-network
        # errors (FileNotFoundError, PermissionError, IsADirectoryError, ...),
        # so we must NOT treat every OSError as transient.  Only genuine
        # network failures qualify.
        if isinstance(error, ConnectionError):
            return self._build(ErrorCategory.TRANSIENT, True, error)
        if isinstance(error, socket.gaierror):
            return self._build(ErrorCategory.TRANSIENT, True, error)
        if isinstance(error, OSError):
            err = getattr(error, "errno", None)
            if err in _NETWORK_ERRNOS:
                return self._build(ErrorCategory.TRANSIENT, True, error)
            # Other OSErrors are not network-related; fall through to return
            # None so the chain walker / string classifier can decide.

        # HTTP-style exceptions: read attributes defensively.
        status = self._read_status(error)
        retry_after = self._read_retry_after(error)
        if status is not None:
            category, retryable = _status_to_category(status)
            if status == 400 and retry_after is None:
                msg = str(error).lower()
                if any(p in msg for p in _CONTEXT_OVERFLOW_PHRASES):
                    return self._build(
                        ErrorCategory.CONTEXT_OVERFLOW,
                        False,
                        error,
                        status_code=status,
                    )
            return self._build(
                category, retryable, error, status_code=status, retry_after=retry_after
            )
        # No specific signal: return None so the chain walker can dig deeper.
        return None

    @staticmethod
    def _read_status(error: BaseException) -> int | None:
        for attr in ("status_code", "status", "http_status"):
            val = getattr(error, attr, None)
            if isinstance(val, int):
                return val
        response = getattr(error, "response", None)
        if response is not None:
            for attr in ("status_code", "status"):
                val = getattr(response, attr, None)
                if isinstance(val, int):
                    return val
        return None

    @staticmethod
    def _read_retry_after(error: BaseException) -> float | None:
        response = getattr(error, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:
            return None
        return _parse_retry_after(value if isinstance(value, str) else None)

    # ---------------------------------------------------------- string handling
    def _classify_string(
        self, message: str, exception: BaseException | None = None
    ) -> ClassifiedError:
        lowered = message.lower()
        exc_type = type(exception).__name__ if exception else None

        # Extract an HTTP-ish status code embedded in the text (used below).
        status: int | None = None
        m = _ERROR_CODE_RE.search(message)
        if m:
            status = int(m.group(1))
        else:
            m = _STATUS_IN_TEXT_RE.search(message)
            if m:
                status = int(m.group(1))

        # Context overflow phrases (checked before generic 400 mapping).
        if any(p in lowered for p in _CONTEXT_OVERFLOW_PHRASES):
            return ClassifiedError(
                category=ErrorCategory.CONTEXT_OVERFLOW,
                retryable=False,
                status_code=status,
                message=message,
                exception_type=exc_type,
            )
        # Rate limit phrases.
        if any(p in lowered for p in _RATE_LIMIT_PHRASES):
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMITED,
                retryable=True,
                status_code=status,
                message=message,
                exception_type=exc_type,
            )
        # Timeout phrases -> TIMEOUT (retryable, in the default retry_on set).
        if any(p in lowered for p in _TIMEOUT_PHRASES):
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                retryable=True,
                status_code=status,
                message=message,
                exception_type=exc_type,
            )
        # Transient phrases.
        if any(p in lowered for p in _TRANSIENT_PHRASES):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                retryable=True,
                status_code=status,
                message=message,
                exception_type=exc_type,
            )

        # Status code present without a matching phrase.
        if status is not None:
            category, retryable = _status_to_category(status)
            if status == 400 and any(p in lowered for p in _CONTEXT_OVERFLOW_PHRASES):
                category, retryable = ErrorCategory.CONTEXT_OVERFLOW, False
            return ClassifiedError(
                category=category,
                retryable=retryable,
                status_code=status,
                message=message,
                exception_type=exc_type,
            )

        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            retryable=False,
            message=message,
            exception_type=exc_type,
        )


# A shared default classifier instance.  Tests register rules on a fresh one.
default_classifier = ErrorClassifier()


def is_unrecoverable(classified: ClassifiedError) -> bool:
    """Decide whether a classified error should stop the run.

    Matches the old ``_is_unrecoverable_error`` semantics: anything that is not
    retryable AND falls into one of the unrecoverable categories stops the run.
    Retries-exhausted is handled by the executor separately.
    """
    if classified.category == ErrorCategory.CANCELLED:
        return True
    return not classified.retryable and classified.category in _UNRECOVERABLE_CATEGORIES
