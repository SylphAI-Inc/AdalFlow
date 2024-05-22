from core.component import Component
from core.generator import Generator
from components.api_client import GroqAPIClient
from core.prompt_builder import Prompt

from use_cases.icl.data import _COARSE_LABELS, _FINE_LABELS
from use_cases.icl.prompt import CLASSIFICATION_TASK_DESC, OUTPUT_FORMAT_STR

import utils.setup_env


class TRECClassifier(Component):
    def __init__(self, labels: list = _COARSE_LABELS):
        super().__init__()
        self.labels = labels
        # custome prompt with variable, use Prompt to generate it
        self.task_desc_prompt = Prompt(
            template=CLASSIFICATION_TASK_DESC,
            preset_prompt_kwargs={"classes": self.labels},
        )
        task_desc_str = self.task_desc_prompt()
        self.generator = Generator(
            model_client=GroqAPIClient,
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": task_desc_str,
                "output_format_str": OUTPUT_FORMAT_STR,
            },
        )

    def call(self, query: str) -> str:
        label = int(self.generator.call(input=query))
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
