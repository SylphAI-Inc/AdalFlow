from dataclasses import dataclass, field

from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient
from prompts.outputs import YAMLOutputParser

import utils.setup_env


# Define a dataclass for the YAML output schema extraction
@dataclass
class Joke:
    setup: str = field(metadata={"description": "question to set up a joke"})
    punchline: str = field(metadata={"description": "answer to resolve the joke"})


class JokeGenerator(Component):
    def __init__(self):
        super().__init__()
        yaml_parser = YAMLOutputParser(data_class_for_yaml=Joke)
        self.generator = Generator(
            model_client=GroqAPIClient,
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": "Answer user query. "
                + yaml_parser.format_instructions()
            },
            output_processors=yaml_parser,
        )

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    joke_generator = JokeGenerator()
    print(joke_generator)
    print("show the system prompt")
    joke_generator.generator.print_prompt()
    print("Answer:")
    print(joke_generator.call("Tell me a joke."))
