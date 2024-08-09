import os
import sys


def get_adalflow_default_root_path() -> str:
    r"""This will be used for storing datasets, cache, logs, trained models, etc."""
    root = None
    if sys.platform == "win32":
        root = os.path.join(os.getenv("APPDATA"), "adalflow")
    else:
        root = os.path.join(os.path.expanduser("~"), ".adalflow")
    return root
