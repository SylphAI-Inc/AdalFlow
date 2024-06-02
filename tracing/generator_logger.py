"""
(1) for generator, we track all different prompts and their corresponding arguments
"""

from typing import Dict, Any, List
import os


from dataclasses import dataclass, field
from datetime import datetime
import json

from core.generator import Generator
from core.data_classes import BaseDataClass
from utils import serialize


@dataclass
class GeneratorRecord(BaseDataClass):
    prompt_states: Dict[str, Any] = field(default_factory=dict)
    time_stamp: str = field(default_factory=str)

    def __eq__(self, other: "GeneratorRecord"):
        return serialize(self.prompt_states) == serialize(other.prompt_states)

    def __ne__(self, other: "GeneratorRecord"):
        return not self.__eq__(other)


class GeneratorLogger:
    def __init__(
        self,
        filename: str = "./traces/generator_trace.json",
    ):
        self.filename = filename
        # self.component_name = name
        # self.generator_state = generator  # TODO: make this a generator state
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self._trace_map: Dict[str, List[GeneratorRecord]] = (
            {}
        )  # {component_name: prompt_state}
        self.load()
        print(f"trace map init: {self._trace_map}")

    def log_prompt(
        self, generator: Generator, name: str
    ):  # log a generator's prompt component
        prompt_states: Dict = (
            generator.system_prompt.to_dict()
        )  # TODO: log all states of the generator instead of just the prompt
        if name not in self._trace_map:
            self._trace_map[name] = [
                GeneratorRecord(
                    prompt_states=prompt_states, time_stamp=datetime.now().isoformat()
                )
            ]
            print(f"trace map: {self._trace_map}")
            self.save()
        else:
            # append any changes along with the time stamp
            # check if the prompt has changed
            last_record = self._trace_map[name][-1]
            new_prompt_record = GeneratorRecord(
                prompt_states=prompt_states, time_stamp=datetime.now().isoformat()
            )

            if last_record != new_prompt_record:
                self._trace_map[name].append(new_prompt_record)
                self.save()

    def save(self):
        with open(self.filename, "w") as f:
            serialized_obj = serialize(self._trace_map)
            f.write(serialized_obj)

    def load(self):
        if not os.path.exists(self.filename):
            print(f"File {self.filename} does not exist.")
            return
        if os.stat(self.filename).st_size == 0:
            print(f"File {self.filename} is empty.")
            return
        with open(self.filename, "r") as f:
            content = f.read().strip()
            if not content:
                print(f"File {self.filename} is empty after stripping.")
                return
            self._trace_map = json.loads(content)
            # convert each dict record to PromptRecord
            for name, records in self._trace_map.items():
                self._trace_map[name] = [
                    GeneratorRecord.load_from_dict(record) for record in records
                ]
