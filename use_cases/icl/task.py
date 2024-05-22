from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient
from core.prompt_builder import Prompt

from use_cases.icl.data import _COARSE_LABELS, _FINE_LABELS
from use_cases.icl.prompt import (
    CLASSIFICATION_TASK_DESC,
    OUTPUT_FORMAT_STR,
    EXAMPLES_STR,
    TEMPLATE,
)

from typing import Dict

import utils.setup_env
import re


class TRECClassifier(Component):
    r"""
    Optimizing goal is the examples_str in the prompt
    """

    def __init__(self, labels: list = _COARSE_LABELS):
        super().__init__()
        self.labels = labels
        # custome prompt with variable, use Prompt to generate it
        # the varaibles in the prompts become the model parameters to optimize
        # component and variables

        self.task_desc_prompt = Prompt(
            template=CLASSIFICATION_TASK_DESC,
            preset_prompt_kwargs={"classes": self.labels},
        )
        task_desc_str = self.task_desc_prompt()

        self.parameters = [
            {
                "component": Generator,
                "args": {
                    "model_client": GroqAPIClient,
                    "model_kwargs": {"model": "llama3-8b-8192", "temperature": 0.0},
                    "preset_prompt_kwargs": {
                        "task_desc_str": task_desc_str,
                        "output_format_str": OUTPUT_FORMAT_STR,
                    },
                },
            }
        ]
        self.generator = Generator(
            model_client=GroqAPIClient,
            model_kwargs={"model": "llama3-8b-8192"},
            template=TEMPLATE,
            preset_prompt_kwargs={
                "task_desc_str": task_desc_str,
                "output_format_str": OUTPUT_FORMAT_STR,
            },
        )

    def load_state_dict(self, state_dict: Dict):
        r"""
        generator_state_dict = {
        "preset_prompt_kwargs": {
        "examples_str": "Examples: \n\n1. What is the capital of France? \n2. Who is the president of the United States?"

        }
        state_dict = {
            "generator": {generator_state_dict}
        }
        """
        self.generator.load_state_dict(state_dict["generator"])

    def call(self, query: str) -> str:
        str_response = self.generator.call(input=query, prompt_kwargs={"input": query})

        # use re to find the first integer in the response, can be multiple digits
        re_pattern = r"\d+"
        label = re.findall(re_pattern, str_response)
        if label:
            label = int(label[0])
        else:
            label = -1

        # class_name = self.labels[label]
        return label


if __name__ == "__main__":
    # test one example
    query = "How did serfdom develop in and then leave Russia ?"
    trec_classifier = TRECClassifier(labels=_COARSE_LABELS)
    print(trec_classifier)
    trec_classifier.generator.print_prompt()
    label = trec_classifier.call(query)
    print(f"label: {label}")
