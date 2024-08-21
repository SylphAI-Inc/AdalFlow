import os
from adalflow.utils.global_config import get_adalflow_default_root_path


def prepare_dataset_path(root: str, task_name: str) -> str:
    if root is None:
        root = os.path.join(get_adalflow_default_root_path(), "cache_datasets")

    save_path = os.path.join(root, task_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
