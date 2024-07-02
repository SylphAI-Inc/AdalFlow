from .react import DEFAULT_REACT_AGENT_SYSTEM_PROMPT, ReActAgent
from lightrag.utils.registry import EntityMapping

__all__ = [
    "ReActAgent",
    "DEFAULT_REACT_AGENT_SYSTEM_PROMPT",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
