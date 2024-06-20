import json
import logging

from typing import Mapping, Any, Dict, Union
import enum
import inspect

from lightrag.utils.registry import EntityMapping


log = logging.getLogger(__name__)


class ObjectTypes(enum.Enum):
    CLASS = 1
    INSTANCE = 2
    TYPE = 3


def check_object(obj) -> ObjectTypes:
    if inspect.isclass(obj):
        return ObjectTypes.CLASS

    elif isinstance(obj, type):
        return ObjectTypes.TYPE
    else:
        return ObjectTypes.INSTANCE


def default(o: Any) -> Union[Dict[str, Any], str]:
    r"""Customized JSON serializer.

    (1) Support internal dataclasses and components instance serialization by calling to_dict() method.
    (2) Support internal dataclasses class serialization by calling to_dict_class() method.

    (3) Any external instance if it has to_dict() method, it will be serialized.
    (4) All other objects will be serialized as {"type": type(o).__name__, "data": str(o)}
    """
    seralized_obj = {}

    # 1. Handle the case like Component, DataClass where we have to_dict and to_dict_class
    obj_type = check_object(o)
    if obj_type == ObjectTypes.CLASS:
        log.debug(f"Object {o} is a class with name {o.__name__}")
        if hasattr(o, "to_dict_class") and callable(getattr(o, "to_dict_class")):
            try:
                seralized_obj = o.to_dict_class()
                return seralized_obj
            except Exception as e:
                log.error(f"Error serializing object {o}: {e}")
                pass
    elif obj_type == ObjectTypes.INSTANCE:
        log.debug(f"Object {o} is an instance of {o.__class__.__name__}")
        if hasattr(o, "to_dict") and callable(getattr(o, "to_dict")):
            try:
                seralized_obj = o.to_dict()
                return seralized_obj
            except Exception as e:
                log.error(f"Error serializing object {o}: {e}")
                pass
    elif obj_type == ObjectTypes.TYPE:
        log.debug(f"Object {o} is a type of {o.__name__}")
        try:
            return {"type": type(o).__name__, "data": str(o)}
        except Exception as e:
            log.error(
                f"Object of type {o.__class__.__name__} is not JSON serializable: {str(e)}"
            )
            pass

    log.debug("Fallback to serializing attributes who come from external libraries")

    # (2) Fallback to serializing attributes who come from external libraries
    try:
        return {"type": type(o).__name__, "data": str(o)}
    except Exception as e:
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable: {str(e)}"
        )


def serialize(obj: Mapping[str, Any]) -> str:
    """Serialize the object to a json string.

    Args:
        obj (Mapping[str, Any]): The object to be serialized.

    Returns:
        str: The serialized object in json format.
    """

    return json.dumps(obj, indent=4, default=default)


def deserialize(data: str) -> Any:
    """Deserialize the JSON string back to an object."""
    return json.loads(data, object_hook=_deserialize_object_hook)


def _deserialize_object_hook(d: Dict[str, Any]) -> Any:
    """Hook to deserialize objects based on their type."""
    if "type" in d and "data" in d:
        class_name = d["type"]
        class_type = EntityMapping.get(class_name)
        if class_type:
            return class_type.from_dict(d)
    return d


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert the object to a dictionary.

    Args:
        obj (Any): The object to be converted.

    Returns:
        Dict[str, Any]: The dictionary representation of the object.
    """
    return json.loads(serialize(obj))
