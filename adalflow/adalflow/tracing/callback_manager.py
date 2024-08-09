"""A simple callback manager for tracing Geneator calls"""

from typing import Callable, Literal


class CallbackManager:
    def __init__(self):
        self.callbacks = {
            "on_success": [],
            "on_failure": [],
            "on_complete": [],  # You can define more event types as needed.
        }

    def register_callback(
        self,
        event_type: Literal["on_success", "on_failure", "on_complete"],
        callback: Callable,
    ):
        """Register a callback for a specific event type."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unsupported event type: {event_type}")

    def trigger_callbacks(
        self,
        event_type: Literal["on_success", "on_failure", "on_complete"],
        *args,
        **kwargs,
    ):
        """Invoke all callbacks for a given event type."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                callback(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported event type: {event_type}")
