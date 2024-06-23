from typing import Dict, Any, List, Optional
import os
import logging


from dataclasses import dataclass, field
from datetime import datetime
import json

from lightrag.core.generator import GeneratorOutput
from lightrag.core.base_data_class import DataClass
from lightrag.utils import append_to_jsonl, load_jsonl


log = logging.getLogger(__name__)


@dataclass
class GeneratorCallRecord(DataClass):
    prompt_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    output: GeneratorOutput = field(default_factory=GeneratorOutput)
    time_stamp: str = field(default_factory=str)


class GeneratorCallLogger:
    __doc__ = r"""Log the generator calls.

    Allow multiple generators to be logged, and each with its own jsonl file.

    The log files are stored in the ./traces/ directory. If a project_name is provided,
    it will be stored in ./traces/{project_name}.

    Args:
        save_dir (str, optional): The directory to save the log files. Defaults to "./traces/".
        project_name (str, optional): The project name. Defaults to None.
    """
    _generator_names_to_files: Dict[str, str] = {}
    # TODO: a project will share the same metadata file to record each logger's metadata
    _metadata_filename = "logger_metadata.json"

    def __init__(
        self, save_dir: Optional[str] = None, project_name: Optional[str] = None
    ):
        self.filepath = save_dir or "./traces/"
        self.project_name = project_name
        if project_name:
            self.filepath = os.path.join(self.filepath, project_name)

        os.makedirs(self.filepath, exist_ok=True)
        self._metadata_filepath = os.path.join(self.filepath, self._metadata_filename)
        if os.path.exists(self._metadata_filepath):
            self.load_meta_data(self._metadata_filepath)
        else:
            self.save_meta_data(self._metadata_filepath)

    @property
    def generator_names_to_files(self) -> Dict[str, str]:
        return self._generator_names_to_files

    def get_log_location(self, name: str) -> str:
        return self._generator_names_to_files.get(name, None)

    def get_calls(self, name: str) -> List[GeneratorCallRecord]:
        r"""Get the generator call records for generator with name."""
        return self.load(name)

    def register_generator(self, name: str, filename: Optional[str] = None):
        r"""Register a generator with a name and a jsonl file to log its calls.

        Args:
            name (str): The name of the generator.
            filename (str, optional): The jsonl filename to log the calls. Defaults to {name}_call.jsonl.
        """
        if name in self._generator_names_to_files:
            log.warning(
                f"Generator {name} is already registered with jsonl file at {self._generator_names_to_files[name]}"
            )
            return
        filename = filename or f"{name}_call.jsonl"
        self._generator_names_to_files[name] = os.path.join(self.filepath, filename)
        self.save_meta_data(self._metadata_filepath)

    def save_meta_data(self, file_path: str):
        """Save the _generator_names_to_files to a json file."""
        with open(file_path, "w") as file:
            json.dump(self._generator_names_to_files, file, indent=4)

    def load_meta_data(self, file_path: str):
        """Load the _generator_names_to_files from a json file."""
        if not os.path.exists(file_path):
            log.error(f"File {file_path} does not exist.")
            return

        with open(file_path, "r") as file:
            self._generator_names_to_files = json.load(file)

    def load(self, name: str) -> List[GeneratorCallRecord]:
        r"""Load the generator call records."""
        if name not in self._generator_names_to_files:
            log.error(f"Generator {name} is not registered.")
            raise FileNotFoundError(f"Generator {name} is not registered.")

        records: List[Dict] = load_jsonl(self._generator_names_to_files[name])
        return [GeneratorCallRecord.from_dict(record) for record in records]

    def log_call(
        self,
        name: str,
        model_kwargs: Dict[str, Any],
        prompt_kwargs: Dict[str, Any],
        output: GeneratorOutput,
    ):
        r"""Log the generator call."""

        if name not in self._generator_names_to_files:
            log.error(f"Generator {name} is not registered.")
            return

        record = GeneratorCallRecord(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            output=output,
            time_stamp=datetime.now().isoformat(),
        )
        append_to_jsonl(self._generator_names_to_files[name], record.to_dict())
