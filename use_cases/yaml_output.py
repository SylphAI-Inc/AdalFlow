from dataclasses import dataclass, field

from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient, OpenAIClient
from prompts.outputs import YAMLOutputParser, ListOutputParser


from dataclasses import fields
from core.data_classes import BaseDataClass


@dataclass
class JokeOutput(BaseDataClass):
    setup: str = field(metadata={"desc": "question to set up a joke"}, default="")
    punchline: str = field(metadata={"desc": "answer to resolve the joke"}, default="")


joke_example = JokeOutput(
    setup="Why did the scarecrow win an award?",
    punchline="Because he was outstanding in his field.",
)


class JokeGenerator(Component):
    def __init__(self):
        super().__init__()
        yaml_parser = YAMLOutputParser(
            data_class_for_yaml=JokeOutput, example=joke_example
        )
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
    joker_class = JokeOutput
    print(joker_class)
    print(fields(joker_class))
    print(joker_class.__dataclass_fields__["setup"].metadata)
    print(f"signature:\n", joker_class.to_yaml_signature())
    print(f"json signature:\n", joker_class.to_json_signature())
    print(f"class schema:\n", joker_class.get_data_class_schema())
    print(joke_example)
    print(f"example yaml signature:\n", joke_example.to_yaml_signature())
    print(f"example json signature:\n", joke_example.to_json_signature())
    print(f"example schema:\n", joke_example.get_data_class_schema())
