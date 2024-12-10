"""Prepare the task pipeline"""

from adalflow.optim.parameter import ParameterType

from use_cases.question_answering.bbh.data import (
    parse_integer_answer,
)


# Few shot demonstration can be less effective when performance already high
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

from typing import Dict, Union
import adalflow as adal


class ObjectCountTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            # data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
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
            template=few_shot_template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            output_processors=parse_integer_answer,
            use_cache=True,
        )

    def call(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output


def test_object_count_task():
    from use_cases.config import gpt_3_model

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"
    task_pipeline = ObjectCountTaskPipeline(**gpt_3_model)
    print(task_pipeline)

    answer = task_pipeline(question)
    print(answer)

    # set it to train mode
    task_pipeline.train()
    answer: adal.Parameter = task_pipeline(question, id="1")
    print(answer)
    print(f"full_response: {answer.full_response}")
    answer.draw_graph()


if __name__ == "__main__":
    # task = ObjectCountTask(**gpt_3_model)
    # task_original = ObjectCountTaskOriginal(**gpt_3_model)

    # question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

    # print(task(question))
    # print(task_original(question))

    test_object_count_task()
