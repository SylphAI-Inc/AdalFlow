from core.component import Component, Sequential, fun_to_component
from core.generator import Generator
from components.api_client import (
    GroqAPIClient,
    OpenAIClient,
    GoogleGenAIClient,
    AnthropicAPIClient,
)
from core.prompt_builder import Prompt
from prompts.outputs import YAMLOutputParser
from core.string_parser import JsonParser

from use_cases.classification.data import (
    _COARSE_LABELS,
    _COARSE_LABELS_DESC,
)
from use_cases.classification.prompt import (
    CLASSIFICATION_TASK_DESC,
    OUTPUT_FORMAT_STR,
    TEMPLATE,
    OutputFormat,
    output_example,
    OUTPUT_FORMAT_YAML_STR,
)

from typing import Dict, Any

import utils.setup_env
import re

import logging

logger = logging.getLogger(__name__)


class TRECClassifier(Component):
    r"""
    Optimizing goal is the examples_str in the prompt
    """

    def __init__(
        self, labels: list = _COARSE_LABELS, labels_desc: list = _COARSE_LABELS_DESC
    ):
        super().__init__()
        self.labels = labels
        self.num_classes = len(labels)
        self.labels_desc = labels_desc
        labels_desc = [
            {"label": label, "desc": desc} for label, desc in zip(labels, labels_desc)
        ]
        # custome prompt with variable, use Prompt to generate it
        # the varaibles in the prompts become the model parameters to optimize
        # component and variables

        self.task_desc_prompt = Prompt(
            template=CLASSIFICATION_TASK_DESC,
            preset_prompt_kwargs={"classes": labels_desc},
        )
        self.task_desc_str = self.task_desc_prompt()

        self.parameters = [
            {
                "component": Generator,
                "args": {
                    "model_client": GroqAPIClient,
                    "model_kwargs": {"model": "llama3-8b-8192", "temperature": 0.0},
                    "preset_prompt_kwargs": {
                        "task_desc_str": self.task_desc_str,
                        "output_format_str": OUTPUT_FORMAT_STR,
                    },
                },
            }
        ]
        yaml_parser = YAMLOutputParser(
            data_class=OutputFormat,  # example=output_example
        )
        # output_str = OutputFormat.to_json_signature()
        output_str = yaml_parser.format_instructions()
        logger.debug(f"output_str: {output_str}")
        groq_model_kwargs = {
            "model": "llama3-8b-8192",  # "llama3-8b-8192",  # "llama3-8b-8192",  # "llama3-8b-8192", #gemma-7b-it not good at following yaml format
            "temperature": 0.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        openai_model_kwargs = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        google_model_kwargs = {
            "model": "gemini-1.5-pro-latest",
            "temperature": 0.0,
            "top_p": 1,
            # "frequency_penalty": 0,
            # "presence_penalty": 0,
            # "n": 1,
        }
        anthropic_model_kwargs = {
            "model": "claude-3-opus-20240229",
            "temperature": 0.0,
            "top_p": 1,
            # "frequency_penalty": 0,
            # "presence_penalty": 0,
            # "n": 1,
            "max_tokens": 1024,
        }

        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs=groq_model_kwargs,
            template=TEMPLATE,
            preset_prompt_kwargs={
                "task_desc_str": self.task_desc_str,
                # "output_format_str": Prompt(
                #     template=OUTPUT_FORMAT_YAML_STR,
                #     preset_prompt_kwargs={"include_thought": True},
                # )(),
                "output_format_str": output_str,  # OUTPUT_FORMAT_STR,
                "input_label": "Question",
            },
            trainable_params=["examples_str"],
            output_processors=Sequential(
                yaml_parser, fun_to_component(lambda x: x["class_index"])
            ),
            # output_processors=yaml_parser,
        )

    # def init_parameters(self):
    #     self.generator.examples_str.update_value()

    def call(self, query: str) -> str:
        str_response: Dict[str, Any] = self.generator.call(
            input=query, prompt_kwargs={"input": query}
        )

        # use re to find the first integer in the response, can be multiple digits
        re_pattern = r"\d+"
        # label = re.findall(re_pattern, str_response)
        # if label:
        #     label = int(label[0])
        # else:
        #     label = -1
        # if label >= self.num_classes:
        #     label = -1

        # class_name = self.labels[label]

        label = str_response
        if isinstance(label, str):
            label_match = re.findall(re_pattern, label)
            if label_match:
                label = int(label_match[0])
            else:
                label = -1
        return label


if __name__ == "__main__":
    # test one example
    query = "How did serfdom develop in and then leave Russia ?"
    trec_classifier = TRECClassifier(labels=_COARSE_LABELS)
    print(trec_classifier)
    trec_classifier.generator.print_prompt()
    label = trec_classifier.call(query)
    print(f"label: {label}")
