from .trainer import Trainer, AdalComponent, PromptData
from lightrag.utils.registry import EntityMapping


__all__ = [
    "Trainer",
    "AdalComponent",
    "PromptData",
]
for name in __all__:
    EntityMapping.register(name, globals()[name])
