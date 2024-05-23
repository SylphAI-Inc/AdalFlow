from dataclasses import dataclass, field

from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient, OpenAIClient
from prompts.outputs import YAMLOutputParser, ListOutputParser

import utils.setup_env
import yaml


# Define a dataclass for the YAML output schema extraction
@dataclass
class Joke:
    setup: str = field(metadata={"description": "question to set up a joke"})
    punchline: str = field(metadata={"description": "answer to resolve the joke"})


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
    joke_generator = JokeGenerator()
    print(joke_generator)
    print("show the system prompt")
    joke_generator.generator.print_prompt()
    print("Answer:")
    answer = joke_generator.call("Tell me two jokes.", model_kwargs={"temperature": 1})
    print(answer)
    print(f"typeof answer: {type(answer)}")
