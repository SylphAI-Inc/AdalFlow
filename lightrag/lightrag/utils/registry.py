from typing import Dict, Type


class EntityMapping:
    __doc__ = r"""A registry for entities, components,classes, function.

    This can be used to configure classes, functions, or components in a registry.
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, entity_cls: Type):
        cls._registry[name] = entity_cls

    @classmethod
    def get(cls, name: str) -> Type:
        return cls._registry[name] if name in cls._registry else None

    @classmethod
    def get_all(cls) -> Dict[str, Type]:
        return cls._registry
