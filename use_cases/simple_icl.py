"""
We just need a very basic generator that can be used to generate text from a prompt.
"""

from typing import List

from core.generator import Generator
from core.openai_client import OpenAIClient
from core.data_classes import Example

from core.component import Component

# TODO: make the environment variable loading more robust, and let users specify the .env path
import dotenv
import random

dotenv.load_dotenv()


class SimpleICL(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "gpt-3.5-turbo"}
        self.generator = Generator(
            model_client=OpenAIClient(), model_kwargs=model_kwargs
        )
        self.generator.print_prompt()

    def get_few_shot_example_str(self, examples: List[Example], n: int) -> str:
        example_str = ""
        for example in random.sample(examples, n):
            example_str += f"Input: {example.input}\nOutput: {example.output}\n\n"
        return example_str

    def call(self, task_desc: str, query: str, example_str: str) -> str:
        return self.generator.call(
            input=query,
            prompt_kwargs={"task_desc": task_desc, "example_str": example_str},
        )


if __name__ == "__main__":
    task_desc = "Classify the sentiment of the following reviews as either Positive or Negative."

    example1 = Example(
        input="Review: I absolutely loved the friendly staff and the welcoming atmosphere!",
        output="Sentiment: Positive",
    )
    example2 = Example(
        input="Review: It was an awful experience, the food was bland and overpriced.",
        output="Sentiment: Negative",
    )
    example3 = Example(
        input="Review: What a fantastic movie! Had a great time and would watch it again!",
        output="Sentiment: Positive",
    )

    simple_icl = SimpleICL()
    example_str = simple_icl.get_few_shot_example_str(
        [example1, example2, example3], n=2
    )
    print(simple_icl)
    print(
        simple_icl.call(
            task_desc,
            "Review: The concert was a lot of fun and the band was energetic and engaging.",
            example_str,
        )
    )
