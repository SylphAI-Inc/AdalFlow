"""Agent components for AdalFlow."""

from .react import ReActAgent as LegacyReActAgent
from .react_agent import ReActAgent as NewReActAgent
from .base_agent import (
    BaseAgent,
    BasePlanner,
    BaseToolManager,
    BaseMemory,
    Step,
    AgentOutput,
)

__all__ = [
    "LegacyReActAgent",  # Old implementation for backward compatibility
    "NewReActAgent",  # New implementation using base agent
    "BaseAgent",  # Base agent class
    "BasePlanner",  # Base planner interface
    "BaseToolManager",  # Base tool manager interface
    "BaseMemory",  # Base memory interface
    "Step",  # Step data class
    "AgentOutput",  # Output data class
]
