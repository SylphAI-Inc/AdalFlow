import json
import os
import pickle
import logging
from typing import Mapping, Any, Optional, List, Dict


from adalflow.utils.serialization import to_dict, serialize, _deserialize_object_hook

log = logging.getLogger(__name__)


def save_json(obj: Mapping[str, Any], f: str = "task.json") -> None:
    """Customized Save the object to a json file.

    Support Set.
    We encourage users first save the data as DataClass using to_dict,
    and then load it back to DataClass using from_dict.

    Args:
        obj (Mapping[str, Any]): The object to be saved.
        f (str, optional): The file name. Defaults to "task".
    """
    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    try:
        with open(f, "w") as file:
            serialized_obj = serialize(obj)
            file.write(serialized_obj)
    except IOError as e:
        raise IOError(f"Error saving object to JSON file {f}: {e}")


# def standard_save_json(obj: Mapping[str, Any], f: str = "task.json") -> None:
#     os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
#     try:
#         with open(f, "w") as file:
#             json.dump(obj, file, indent=4)
#     except IOError as e:
#         raise IOError(f"Error saving object to JSON file {f}: {e}")


def save_csv(
    obj: List[Dict[str, Any]], f: str = "task.csv", fieldnames: List[str] = None
) -> None:
    """Save the object to a csv file.

    Args:
        obj (List[Dict[str, Any]]): The object to be saved.
        f (str, optional): The file name. Defaults to "task".
    """
    import csv

    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    try:
        with open(f, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames or obj[0].keys())
            writer.writeheader()
            for row in obj:
                filtered_row = {k: v for k, v in row.items() if k in fieldnames}
                # use json.dumps to serialize the object
                for k, v in filtered_row.items():
                    if (
                        isinstance(v, dict)
                        or isinstance(v, list)
                        or isinstance(v, tuple)
                        or isinstance(v, set)
                    ):
                        filtered_row[k] = json.dumps(v)
                writer.writerow(filtered_row)
    except IOError as e:
        raise IOError(f"Error saving object to CSV file {f}: {e}")


def save_pickle(obj: Mapping[str, Any], f: str = "task.pickle") -> None:
    """Save the object to a pickle file.

    Args:
        obj (Mapping[str, Any]): The object to be saved.
        f (str, optional): The file name. Defaults to "task".
    """
    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    try:
        with open(f, "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise Exception(f"Error saving object to pickle file {f}: {e}")


def save(obj: Mapping[str, Any], f: str = "task") -> None:
    r"""Save the object to both a json and a pickle file.

    We save two versions of the object:
    - task.json: the object itself with Parameter serialized to dict
    - task.pickle: the object itself with Parameter as is
    """

    try:
        save_json(obj, f=f"{f}.json")
        save_pickle(obj, f=f"{f}.pickle")
    except Exception as e:
        raise Exception(f"Error saving object to json and pickle files: {e}")


# def load_json(f: str = "task.json") -> Optional[Mapping[str, Any]]:
#     r"""Load the object from a json file.

#     Args:
#         f (str, optional): The file name. Defaults to "task".
#     """
#     if not os.path.exists(f):
#         log.warning(f"File {f} does not exist.")
#         return None
#     try:
#         with open(f, "r") as file:
#             return json.load(file)
#     except Exception as e:
#         raise Exception(f"Error loading object from JSON file {f}: {e}")


def load_json(f: str) -> Any:
    """Customized Load a JSON file and deserialize it.

    Args:
        f (str): The file name of the JSON file to load.

    Returns:
        Any: The deserialized Python object.
    """
    if not os.path.exists(f):
        raise FileNotFoundError(f"JSON file not found: {f}")

    try:
        with open(f, "r") as file:
            data = json.load(file, object_hook=_deserialize_object_hook)
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file {f}: {e}")
    except Exception as e:
        raise IOError(f"Error loading JSON file {f}: {e}")


def load_standard_json(f: str) -> Any:
    """Standard Load a JSON file and deserialize it.
    Args:
        f (str): The file name of the JSON file to load.

    Returns:
        Any: The deserialized Python object.
    """
    if not os.path.exists(f):
        raise FileNotFoundError(f"JSON file not found: {f}")

    try:
        with open(f, "r") as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file {f}: {e}")
    except Exception as e:
        raise IOError(f"Error loading JSON file {f}: {e}")


def load_pickle(f: str = "task.pickle") -> Optional[Mapping[str, Any]]:
    r"""Load the object from a pickle file.

    Args:
        f (str, optional): The file name. Defaults to "task".
    """
    if not os.path.exists(f):
        log.warning(f"File {f} does not exist.")
        return None
    try:
        with open(f, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading object from pickle file {f}: {e}")


def load(f: str = "task") -> Optional[Mapping[str, Any]]:
    r"""Load both the json and pickle files and return the object from the json file

    Args:
        f (str, optional): The file name. Defaults to "task".
    """
    try:
        json_obj = load_json(f=f"{f}.json")
        obj = load_pickle(f=f"{f}.pickle")
        return json_obj, obj
    except Exception as e:
        raise Exception(f"Error loading object from json and pickle files: {e}")


def load_jsonl(f: str = None) -> List[Dict[str, Any]]:
    r"""Load a jsonl file and return a list of dictionaries.

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
    r"""Append data to a jsonl file.

    Used by the trace_generator_call decorator to log the generator calls.

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
            # call serialize to serialize the object
            serialized_data = to_dict(data)
            writer.write(serialized_data)
            # writer.write(data)
    except Exception as e:
        log.error(f"Error appending data to jsonl file {f}: {e}")


def write_list_to_jsonl(f: str, data: List[Dict[str, Any]]) -> None:
    r"""Write a list of dictionaries to a jsonl file.

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
