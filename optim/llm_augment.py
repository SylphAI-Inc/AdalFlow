r"""
Given the inputs and outputs, the llm augmenter will fill in any field that is missing in the output field.
Better to use more performant models to fill in the missing values.
"""

LLM_AUGMENTER_TEMPLATE = r"""Given inputs and outputs, you will fill in any field that is missing value.
- null or '' means the field is missing.
- Understand the reasoning between inputs and outputs fields. If the 'thought/reasoning' field is null, you will fill in the reasoning 
  between the inputs and existing outputs and explain it well.
- You answer will only include the missing fields along with your values
- {{yaml_format_str}}
{% if task_context_str %}
<CONTEXT>
{{task_context_str}}
</CONTEXT>
{% endif %}
<EXAMPLES>
Inputs:
Question: "Where is the capital of France?"
Outputs:
thought: null,
answer: "Paris"
Your answer:
thought: "I know the capital of France is Paris."
</EXAMPLES>


<Inputs>
{{input_str}}
</Inputs>
<Outputs>
{{output_str}}
</Outputs>
Your answer:
"""

from core.prompt_builder import Prompt
from typing import Dict, Any, Optional
from core.api_client import APIClient
from core.generator import Generator
from core.component import Component
from core.data_classes import BaseDataClass
from core.string_parser import YAMLParser
from prompts.outputs import YAML_OUTPUT_FORMAT


class LLMAugmenter(Component):
    r"""manage the generator creation and the prompt."""

    def __init__(
        self,
        model_client: APIClient,
        model_kwargs: Dict[str, Any],
        task_context_str: Optional[str] = None,
    ):
        r"""Initialize the generator with the model client and the model kwargs."""
        super().__init__()
        # overwrite temperature to 1
        model_kwargs["temperature"] = 1
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            output_processors=YAMLParser(),
            template=LLM_AUGMENTER_TEMPLATE,
            preset_prompt_kwargs={
                "task_context_str": task_context_str,
                "yaml_format_str": YAML_OUTPUT_FORMAT,
            },
        )

    def call(
        self, input_data_obj: BaseDataClass, output_data_obj: BaseDataClass
    ) -> Dict[str, Any]:
        r"""Call the generator with the input and output data objects."""
        input_str = input_data_obj.to_yaml()
        output_str = output_data_obj.to_yaml()
        print(f"Input: {input_str}")
        print(f"Output: {output_str}")

        prompt_kwargs = {"input_str": input_str, "output_str": output_str}
        outputs = self.generator(prompt_kwargs=prompt_kwargs)
        return outputs


if __name__ == "__main__":
    from components.api_client import OpenAIClient
    from use_cases.classification.prompt import (
        InputFormat,
        OutputFormat,
        CLASSIFICATION_TASK_DESC,
    )
    from use_cases.classification.data import _COARSE_LABELS_DESC, _COARSE_LABELS
    import utils.setup_env

    model_client = OpenAIClient
    model_kwargs = {"model": "gpt-4"}
    classes = [
        {"label": label, "desc": desc}
        for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
    ]
    task_template = Prompt(
        template=CLASSIFICATION_TASK_DESC,
        preset_prompt_kwargs={"classes": classes},
    )
    task_context_str = task_template()
    augmenter = LLMAugmenter(
        model_client=model_client,
        model_kwargs=model_kwargs,
        task_context_str=task_context_str,
    )

    example = {
        "text": "What is a fear of disease ?",
        "coarse_label": 1,
        "coarse_label_desc": _COARSE_LABELS_DESC[1],
    }

    input_data_obj = InputFormat(question=example["text"])
    output_data_obj = OutputFormat(
        class_index=example["coarse_label"],
        class_name=example["coarse_label_desc"],
        thought=None,
    )
    outputs = augmenter(input_data_obj, output_data_obj)
    print(outputs)
