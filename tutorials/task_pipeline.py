from typing import Dict, Union, Optional
import re
from dataclasses import dataclass, field


import adalflow as adal
from adalflow.optim.types import ParameterType

template = r"""<START_OF_SYSTEM_PROMPT>
You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""


@adal.func_to_data_component
def parse_integer_answer(answer: str):
    """A function that parses the last integer from a string using regular expressions."""
    try:
        numbers = re.findall(r"\d+", answer)
        if numbers:
            answer = int(numbers[-1])
        else:
            answer = -1
    except ValueError:
        answer = -1
    return answer


class ObjectCountTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            output_processors=parse_integer_answer,
        )

    def bicall(self, question: str, id: str = None) -> adal.GeneratorOutput:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output


template = r"""<START_OF_SYSTEM_PROMPT>
You will answer a reasoning question. Think step by step.
{{output_format_str}}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""


@dataclass
class Sample(adal.DataClass):
    thought: str = field(
        metadata={"desc": "The reasoning thought process to reach the answer"},
    )
    answer: str = field(metadata={"desc": "The answer to the question"})
    question: Optional[str] = field(
        default=None, metadata={"desc": "The question to ask"}
    )
    __output_fields__ = ["thought", "answer"]  # formating will follow this order
    __input_fields__ = ["question"]


class ObjectCountTaskStrucutredPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        parser = adal.DataClassParser(
            data_class=Sample, return_data_class=True, format_type="yaml"
        )
        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "output_format_str": parser.get_output_format_str(),
            },
            output_processors=parser,
        )

    def bicall(self, question: str, id: str = None) -> adal.GeneratorOutput:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output


few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
{# Few shot demos #}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""


class ObjectCountTaskPipelineTrainable(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=False,
            param_type=ParameterType.PROMPT,
            instruction_to_optimizer="You can try to show examples to see if it helps.",
        )
        few_shot_demos = adal.Parameter(
            data=None,
            role_desc="To provide few shot demos to the language model",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            output_processors=parse_integer_answer,
            use_cache=True,
        )

    def bicall(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output


if __name__ == "__main__":
    model_client = adal.ModelClient(model_name="gpt2")
    model_kwargs = {"temperature": 0.7}

    from adalflow.utils import setup_env
    from adalflow.components.model_client import OpenAIClient

    setup_env()

    task_pipeline = ObjectCountTaskPipelineTrainable(
        model_client=OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
    )
    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

    task_pipeline.eval()
    output = task_pipeline(question, id="1")
    print(output)

    task_pipeline.train()
    output = task_pipeline(question, id="1")
    print(output)
