from .react import (
    DEFAULT_REACT_AGENT_SYSTEM_PROMPT,
    ReActAgent,

)
from .prompts import (
    adalflow_agent_task_desc,
    DEFAULT_ADALFLOW_AGENT_SYSTEM_PROMPT,
)
from adalflow.utils.registry import EntityMapping

__all__ = [
    "ReActAgent",
    "DEFAULT_REACT_AGENT_SYSTEM_PROMPT",
    "adalflow_agent_task_desc",
    "DEFAULT_ADALFLOW_AGENT_SYSTEM_PROMPT",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
