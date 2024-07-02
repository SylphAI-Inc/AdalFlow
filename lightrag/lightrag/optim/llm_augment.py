r"""
Given the inputs and outputs, the llm augmenter will fill in any field that is missing in the output field.
Better to use more performant models to fill in the missing values.
"""

from typing import Dict, Any, Optional
import logging

from lightrag.core.model_client import ModelClient
from lightrag.core import Generator, GeneratorOutput
from lightrag.core.component import Component
from lightrag.core.base_data_class import DataClass
from lightrag.core.string_parser import YamlParser

from lightrag.components.output_parsers import YAML_OUTPUT_FORMAT

log = logging.getLogger(__name__)

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
<EXAMPLES> {#TODO: use a better example template #}
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


class LLMAugmenter(Component):
    r"""manage the generator creation and the prompt."""

    def __init__(
        self,
        model_client: ModelClient,
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
            output_processors=YamlParser(),
            template=LLM_AUGMENTER_TEMPLATE,
            preset_prompt_kwargs={
                "task_context_str": task_context_str,
                "yaml_format_str": YAML_OUTPUT_FORMAT,
            },
        )

    # TODO: return GeneratorOutput directly
    def call(
        self, input_data_obj: DataClass, output_data_obj: DataClass
    ) -> Dict[str, Any]:
        r"""Call the generator with the input and output data objects."""
        input_str = input_data_obj.to_yaml()
        output_str = output_data_obj.to_yaml()
        log.info(f"Input: {input_str}")
        log.info(f"Output: {output_str}")

        prompt_kwargs = {"input_str": input_str, "output_str": output_str}
        output: GeneratorOutput = self.generator(prompt_kwargs=prompt_kwargs)
        if not output.error:
            return output.data
        else:
            log.error(f"Error: {output.error}")
            raise ValueError(f"Error: {output.error}")
            return {}
