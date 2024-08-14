from .trainer import Trainer
from .adal import AdalComponent
from adalflow.utils.registry import EntityMapping


__all__ = [
    "Trainer",
    "AdalComponent",
]
for name in __all__:
    EntityMapping.register(name, globals()[name])
