"""
(1) for generator, we track all different prompts and their corresponding arguments
"""

from typing import Dict, Any, List
import os
import logging


from dataclasses import dataclass, field
from datetime import datetime
import json

from core.generator import Generator
from core.data_classes import BaseDataClass
from utils import serialize

logger = logging.getLogger(__name__)


@dataclass
class GeneratorRecord(BaseDataClass):
    prompt_states: Dict[str, Any] = field(default_factory=dict)
    time_stamp: str = field(default_factory=str)

    def __eq__(self, other: "GeneratorRecord"):
        return serialize(self.prompt_states) == serialize(other.prompt_states)

    def __ne__(self, other: "GeneratorRecord"):
        return not self.__eq__(other)


class GeneratorLogger:
    __doc__ = r"""Log the generator states especially the prompt states update history to a file.

    Each generator should has its unique and identifiable name to be logged.
    One file can log multiple generators' states.

    We use _trace_map to store the states and track any changes and updates and save it to a file.
    """

    def __init__(
        self,
        filename: str = "./traces/generator_trace.json",
    ):
        self.filename = filename

        # self.generator_state = generator  # TODO: make this a generator state
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self._trace_map: Dict[str, List[GeneratorRecord]] = (
            {}  # generator_name: [prompt_states]
        )
        if os.path.exists(self.filename):
            self.load()

    def log_prompt(self, generator: Generator, name: str):
        r"""Log the prompt states of the generator with the given name."""
        prompt_states: Dict = (
            generator.system_prompt.to_dict()
        )  # TODO: log all states of the generator instead of just the prompt

        try:

            if name not in self._trace_map:
                self._trace_map[name] = [
                    GeneratorRecord(
                        prompt_states=prompt_states,
                        time_stamp=datetime.now().isoformat(),
                    )
                ]
                self.save()
            else:
                # compare the last record with the new record
                last_record = self._trace_map[name][-1]
                new_prompt_record = GeneratorRecord(
                    prompt_states=prompt_states, time_stamp=datetime.now().isoformat()
                )

                if last_record != new_prompt_record:
                    self._trace_map[name].append(new_prompt_record)
                    self.save()
        except Exception as e:
            raise Exception(f"Error logging prompt states for {name}") from e

    def save(self):
        with open(self.filename, "w") as f:
            serialized_obj = serialize(self._trace_map)
            f.write(serialized_obj)

    def load(self):

        if os.stat(self.filename).st_size == 0:
            logging.info(f"File {self.filename} is empty.")
            return
        with open(self.filename, "r") as f:
            content = f.read().strip()
            if not content:
                logging.info(f"File {self.filename} is empty after stripping.")
                return
            self._trace_map = json.loads(content)
            # convert each dict record to PromptRecord
            for name, records in self._trace_map.items():
                self._trace_map[name] = [
                    GeneratorRecord.load_from_dict(record) for record in records
                ]
