from typing import Dict, Any, List, Optional
import os
import logging


from dataclasses import dataclass, field
from datetime import datetime
import json

from core.generator import GeneratorOutput
from core.data_classes import BaseDataClass
from utils import append_to_jsonl, load_jsonl


log = logging.getLogger(__name__)


@dataclass
class GeneratorCallRecord(BaseDataClass):
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
        dir (str, optional): The directory to save the log files. Defaults to "./traces/".
        project_name (str, optional): The project name. Defaults to None.
    """
    _generator_names_to_files: Dict[str, str] = {}

    def __init__(self, dir: Optional[str] = None, project_name: Optional[str] = None):
        self.dir = dir or "./traces/"
        self.project_name = project_name
        if project_name:
            self.dir = os.path.join(self.dir, project_name)

        os.makedirs(self.dir, exist_ok=True)
        self._metadata_file = os.path.join(self.dir, "generator_names_to_files.json")
        if os.path.exists(self._metadata_file):
            self.load_meta_data()

    @property
    def generator_names_to_files(self) -> Dict[str, str]:
        return self._generator_names_to_files

    def get_location(self, name: str) -> str:
        return self._generator_names_to_files.get(name, None)

    def get_calls(self, name: str) -> List[GeneratorCallRecord]:
        return self.load(name)

    def register_generator(self, name: str, filename: Optional[str] = None):
        r"""Register a generator with a name and a jsonl file to log its calls.

        Args:
            name (str): The name of the generator.
            filename (str, optional): The jsonl filename to log the calls. Defaults to None.
        """
        if name in self._generator_names_to_files:
            log.warning(
                f"Generator {name} is already registered with jsonl file at {self._generator_names_to_files[name]}"
            )
            return
        filename = filename or f"{name}_call.jsonl"
        self._generator_names_to_files[name] = os.path.join(self.dir, filename)

    def save_meta_data(self):
        """Save the _generator_names_to_files to a json file."""
        with open(os.path.join(self.dir, "generator_names_to_files.json"), "w") as file:
            json.dump(self._generator_names_to_files, file, indent=4)

    def load_meta_data(self):
        """Load the _generator_names_to_files from a json file."""
        file_path = os.path.join(self.dir, "generator_names_to_files.json")
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
        return [GeneratorCallRecord.load_from_dict(record) for record in records]

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
