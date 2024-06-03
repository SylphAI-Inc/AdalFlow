import json
import logging
import os
import pickle
from typing import Mapping, Any, Optional, List, Dict


log = logging.getLogger(__name__)


def serialize(obj: Mapping[str, Any]) -> str:
    """Serialize the object to a json string.

    Args:
        obj (Mapping[str, Any]): The object to be serialized.

    Returns:
        str: The serialized object in json format.
    """

    def default(o):
        if hasattr(o, "to_dict"):
            return (
                o.to_dict()
            )  # use custom to_dict method if it exists, dataclass can be handled automatically from __dict__
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    return json.dumps(obj, indent=4, default=default)


# TODO: make this more clear
def save(obj: Mapping[str, Any], f: str = "task") -> None:
    __doc__ = r"""Save the object to a json file.

    We save two versions of the object:
    - task.json: the object itself with Parameter serialized to dict
    - task.pickle: the object itself with Parameter as is
    """

    def default(o):
        if hasattr(o, "to_dict"):
            return (
                o.to_dict()
            )  # use custom to_dict method if it exists, dataclass can be handled automatically from __dict__
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    # save the object to a json file
    json_f = f"{f}.json"

    os.makedirs(os.path.dirname(json_f) or ".", exist_ok=True)
    with open(json_f, "w") as file:
        # serialize the object with the default function
        # serialized_obj = serialize(obj)
        # file.write(serialized_obj, indent=4)
        json.dump(obj, file, indent=4, default=default)

    # save the object to a pickle file
    pickle_f = f"{f}.pickle"
    with open(pickle_f, "wb") as file:
        pickle.dump(obj, file)


def load(f: str = "task") -> Optional[Mapping[str, Any]]:
    __doc__ = r"""Load both the json and pickle files and return the object from the json file

    Args:
        f (str, optional): The file name. Defaults to "task".
    """
    # load the object from a json file
    json_f = f"{f}.json"
    pickle_f = f"{f}.pickle"
    json_obj, pickle_obj = None, None
    if not os.path.exists(json_f):
        json_obj = None
    if not os.path.exists(pickle_f):
        pickle_obj = None
    # load the object from a json file
    with open(json_f, "r") as file:
        json_obj = json.load(file)
    # load the object from a pickle file
    with open(pickle_f, "rb") as file:
        pickle_obj = pickle.load(file)
    return json_obj, pickle_obj


def load_jsonl(f: str = None) -> List[Dict[str, Any]]:
    __doc__ = r"""Load a jsonl file and return a list of dictionaries.

    Args:
        f (str, optional): The file name. Defaults to None.
    """
    try:
        import jsonlines
    except ImportError:
        raise ImportError("Please install jsonlines to use this function.")
    if not os.path.exists(f):
        log.warning(f"File {f} does not exist.")
        return []

    try:
        with jsonlines.open(f) as reader:
            return list(reader)
    except Exception as e:
        log.error(f"Error loading jsonl file {f}: {e}")
        return []


def append_to_jsonl(f: str, data: Dict[str, Any]) -> None:
    __doc__ = r"""Append data to a jsonl file.

    Args:
        f (str): The file name.
        data (Dict[str, Any]): The data to be appended.
    """
    try:
        import jsonlines
    except ImportError:
        raise ImportError("Please install jsonlines to use this function.")
    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    try:
        with jsonlines.open(f, mode="a") as writer:
            writer.write(data)
    except Exception as e:
        log.error(f"Error appending data to jsonl file {f}: {e}")


def write_list_to_jsonl(f: str, data: List[Dict[str, Any]]) -> None:
    __doc__ = r"""Write a list of dictionaries to a jsonl file.

    Args:
        f (str): The file name.
        data (List[Dict[str, Any]]): The data to be written.
    """
    try:
        import jsonlines
    except ImportError:
        raise ImportError("Please install jsonlines to use this function.")
    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    try:
        with jsonlines.open(f, mode="w") as writer:
            for d in data:
                writer.write(d)
    except Exception as e:
        log.error(f"Error writing data to jsonl file {f}: {e}")
