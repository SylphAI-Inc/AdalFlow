from typing import Mapping, Any, Optional
import json
import os
import pickle


def save(obj: Mapping[str, Any], f: str = "task") -> None:
    __doc__ = r"""Save the object to a json file.

    We save two versions of the object:
    - task.json: the object itself with Parameter serialized to dict
    - task.pickle: the object itself with Parameter as is
    """

    def default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    # save the object to a json file
    json_f = f"{f}.json"

    os.makedirs(os.path.dirname(json_f) or ".", exist_ok=True)
    with open(json_f, "w") as file:
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
