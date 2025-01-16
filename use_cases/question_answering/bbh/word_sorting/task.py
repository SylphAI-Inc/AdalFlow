"""Prepare the task pipeline"""

from adalflow.optim.parameter import ParameterType

from use_cases.question_answering.bbh.data import (
    extract_answer,
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


class QuestionAnswerTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            data="Sort the following words alphabetically. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is the answer.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
            # instruction_to_optimizer="Try to instruct what exactly word sorting is and what to do if two words have the same starting letter.",
        )
        few_shot_demos = adal.Parameter(
            data=None,
            role_desc="To provide few shot demos to the language model",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=few_shot_template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            output_processors=extract_answer,
            use_cache=True,
        )

    def call(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm(prompt_kwargs={"input_str": question}, id=id)
        return output


def test_word_sorting_task():
    from use_cases.config import gpt_3_model
    from use_cases.question_answering.bbh.data import load_datasets

    task_pipeline = QuestionAnswerTaskPipeline(**gpt_3_model)
    print(task_pipeline)

    train_dataset, val_dataset, test_dataset = load_datasets(task_name="word_sorting")

    example = train_dataset[0]
    question = example.question
    print(example)

    answer = task_pipeline(question)
    print(answer)

    # set it to train mode
    task_pipeline.train()
    answer = task_pipeline(question, id="1")
    print(answer)
    print(f"full_response: {answer.full_response}")


if __name__ == "__main__":

    # task = ObjectCountTask(**gpt_3_model)
    # task_original = ObjectCountTaskOriginal(**gpt_3_model)

    # question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

    # print(task(question))
    # print(task_original(question))

    test_word_sorting_task()
