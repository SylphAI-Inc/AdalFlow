from dataclasses import dataclass, field

from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient, OpenAIClient
from prompts.outputs import YAMLOutputParser, ListOutputParser

import utils.setup_env
import yaml
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, Dict, List, Optional

"""
from dataclasses import fields, is_dataclass, MISSING

    if not is_dataclass(data_class):
        raise ValueError("Provided class is not a dataclass")
    schema = {}
    for f in fields(data_class):
        field_info = {
            "type": f.type.__name__,
            "description": f.metadata.get("description", ""),
        }

        # Determine if the field is required or optional
        if f.default is MISSING and f.default_factory is MISSING:
            field_info["required"] = True
        else:
            field_info["required"] = False
            if f.default is not MISSING:
                field_info["default"] = f.default
            elif f.default_factory is not MISSING:
                field_info["default"] = f.default_factory()

        schema[f.name] = field_info

    return schema
"""


# Define a dataclass for the YAML output schema extraction
@dataclass
class Joke:
    setup: Optional[str] = field(
        default=None, metadata={"description": "question to set up a joke"}
    )
    punchline: Optional[str] = field(
        default=None, metadata={"description": "answer to resolve the joke"}
    )

    def to_yaml_signature(self) -> str:
        """Generate a YAML signature based on field metadata descriptions."""
        # Create a dictionary to hold the descriptions
        metadata_dict = {}
        # Iterate over the fields of the dataclass
        for f in fields(self):
            # Each field's metadata 'description' is used as the value
            description = f.metadata.get("description", "No description provided")
            metadata_dict[f.name] = description

        # Convert the dictionary to a YAML string
        return yaml.dump(metadata_dict, default_flow_style=False)


joke_example = Joke(
    setup="Why did the scarecrow win an award?",
    punchline="Because he was outstanding in his field.",
)


class JokeGenerator(Component):
    def __init__(self):
        super().__init__()
        yaml_parser = YAMLOutputParser(data_class_for_yaml=Joke, example=joke_example)
        self.generator = Generator(
            model_client=GroqAPIClient,
            model_kwargs={"model": "llama3-8b-8192", "temperature": 1.0},
            preset_prompt_kwargs={
                "output_format_str": yaml_parser.format_instructions()
            },
            output_processors=yaml_parser,
        )

    def call(self, query: str, model_kwargs: dict = {}) -> dict:
        return self.generator.call(input=query, model_kwargs=model_kwargs)


if __name__ == "__main__":
    # joke_generator = JokeGenerator()
    # print(joke_generator)
    # print("show the system prompt")
    # joke_generator.generator.print_prompt()
    # print("Answer:")
    # answer = joke_generator.call("Tell me two jokes.", model_kwargs={"temperature": 1})
    # print(answer)
    # print(f"typeof answer: {type(answer)}")
    joker_class = Joke()
    print(joker_class)
    print(f"signature:\n", joker_class.to_yaml_signature())
