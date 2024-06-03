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
    """
    _generator_names_to_files: Dict[str, str] = {}

    def __init__(self, dir: Optional[str] = None):
        self.dir = dir or "./traces/"
        self._metadata_file = os.path.join(self.dir, "generator_names_to_files.json")
        if os.path.exists(self._metadata_file):
            self.load_meta_data()
        # self.records: List[GeneratorCallRecord] = []

        os.makedirs(self.dir, exist_ok=True)

    @property
    def generator_names_to_files(self) -> Dict[str, str]:
        return self._generator_names_to_files

    @property
    def get_location(self, name: str) -> str:
        return self._generator_names_to_files.get(name, None)

    @property
    def get_calls(self, name: str) -> List[GeneratorCallRecord]:
        return self.load(name)

    def register_generator(self, name: str):
        r"""Register a generator with a name."""
        if name in self._generator_names_to_files:
            log.warning(
                f"Generator {name} is already registered with jsonl file at {self._generator_names_to_files[name]}"
            )
            return

        self._generator_names_to_files[name] = os.path.join(self.dir, f"{name}.jsonl")

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


if __name__ == "__main__":
    import json

    # def append_to_jsonl(file_path, record):
    #     """Append a record to a JSONL file."""
    #     with open(file_path, "a") as file:
    #         file.write(json.dumps(record) + "\n")

    # def append_to_json(file_path, record):
    #     """Append a record to a JSON file."""
    #     with open(file_path, "a") as file:
    #         file.write(json.dumps(record) + "\n")

    # record = {"id": 1, "data": "y" * 1024}  # Example of a long string
    # append_to_jsonl("data.jsonl", record)
