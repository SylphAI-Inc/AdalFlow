from lightrag.core.component import Component
from lightrag.core.generator import Generator
from lightrag.components.model_client import GroqAPIClient, OpenAIClient
from lightrag.components.output_parsers import YAMLOutputParser, ListOutputParser


from lightrag.core.base_data_class import DataClass, field
from lightrag.core.types import GeneratorOutput

from lightrag.utils import setup_env


class JokeOutput(DataClass):
    setup: str = field(metadata={"desc": "question to set up a joke"}, default="")
    punchline: str = field(metadata={"desc": "answer to resolve the joke"}, default="")


joke_example = JokeOutput(
    setup="Why did the scarecrow win an award?",
    punchline="Because he was outstanding in his field.",
)


class JokeGenerator(Component):
    def __init__(self):
        super().__init__()
        yaml_parser = YAMLOutputParser(data_class=JokeOutput, example=joke_example)
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192", "temperature": 1.0},
            preset_prompt_kwargs={
                "output_format_str": yaml_parser.format_instructions()
            },
            output_processors=yaml_parser,
        )

    def call(self, query: str, model_kwargs: dict = {}) -> JokeOutput:
        response: GeneratorOutput = self.generator.call(
            prompt_kwargs={"input_str": query}, model_kwargs=model_kwargs
        )
        if response.error is None:
            output = JokeOutput.load_from_dict(response.data)
            return output
        else:
            None


if __name__ == "__main__":
    joke_generator = JokeGenerator()
    print(joke_generator)
    print("show the system prompt")
    joke_generator.generator.print_prompt()
    print("Answer:")
    answer = joke_generator.call("Tell me two jokes.", model_kwargs={"temperature": 1})
    print(answer)
    print(f"typeof answer: {type(answer)}")
