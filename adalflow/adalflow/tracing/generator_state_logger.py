from typing import Dict, Any, List, Optional, TYPE_CHECKING
import os
import logging


from dataclasses import dataclass, field
from datetime import datetime
import json

if TYPE_CHECKING:
    from adalflow.core.generator import Generator
from adalflow.core.base_data_class import DataClass
from adalflow.utils import serialize


log = logging.getLogger(__name__)


@dataclass
class GeneratorStatesRecord(DataClass):
    prompt_states: Dict[str, Any] = field(default_factory=dict)
    time_stamp: str = field(default_factory=str)

    def __eq__(self, other: Any):
        if not isinstance(other, GeneratorStatesRecord):
            return NotImplemented
        return serialize(self.prompt_states) == serialize(other.prompt_states)


class GeneratorStateLogger:
    __doc__ = r"""Log the generator states especially the prompt states update history to a file.

    Each generator should has its unique and identifiable name to be logged.
    One file can log multiple generators' states.

    We use _trace_map to store the states and track any changes and updates and save it to a file.

    Args:
        save_dir(str, optional): The directory to save the trace file. Default is "./traces/"
        project_name(str, optional): The project name. Default is None.
        filename(str, optional): The file path to save the trace. Default is "generator_state_trace.json"
    """
    _generator_names: set = set()

    # TODO: create a logger base class to avoid code duplication
    def __init__(
        self,
        save_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        self.filepath = save_dir or "./traces/"
        self.project_name = project_name
        if project_name:
            self.filepath = os.path.join(self.filepath, project_name)

        # TODO: make this a generator state instead of just the prompt as right now
        os.makedirs(self.filepath, exist_ok=True)
        self.filename = filename or "generator_state_trace.json"
        self.filepath = os.path.join(self.filepath, self.filename)

        self._trace_map: Dict[str, List[GeneratorStatesRecord]] = (
            {}  # generator_name: [prompt_states]
        )
        # load previous records if the file exists
        if os.path.exists(self.filepath):
            self.load(self.filepath)

    def get_log_location(self) -> str:
        return self.filepath

    @property
    def generator_names(self):
        return self._generator_names

    def log_prompt(self, generator: "Generator", name: str):
        r"""Log the prompt states of the generator with the given name."""
        self._generator_names.add(name)

        prompt_states: Dict = (
            generator.prompt.to_dict()
        )  # TODO: log all states of the generator instead of just the prompt

        try:

            if name not in self._trace_map:
                self._trace_map[name] = [
                    GeneratorStatesRecord(
                        prompt_states=prompt_states,
                        time_stamp=datetime.now().isoformat(),
                    )
                ]
                self.save(self.filepath)
            else:
                # compare the last record with the new record
                last_record = self._trace_map[name][-1]
                new_prompt_record = GeneratorStatesRecord(
                    prompt_states=prompt_states, time_stamp=datetime.now().isoformat()
                )

                if last_record != new_prompt_record:
                    self._trace_map[name].append(new_prompt_record)
                    self.save(self.filepath)
        except Exception as e:
            raise Exception(f"Error logging prompt states for {name}") from e

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            serialized_obj = serialize(self._trace_map)
            f.write(serialized_obj)

    def load(self, filepath: str):

        if os.stat(filepath).st_size == 0:
            logging.info(f"File {filepath} is empty.")
            return
        with open(filepath, "r") as f:
            content = f.read().strip()
            if not content:
                logging.info(f"File {filepath} is empty after stripping.")
                return
            self._trace_map = json.loads(content)
            # convert each dict record to PromptRecord
            for name, records in self._trace_map.items():
                self._trace_map[name] = [
                    GeneratorStatesRecord.from_dict(record) for record in records
                ]
