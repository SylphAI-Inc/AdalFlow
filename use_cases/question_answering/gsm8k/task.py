from typing import Dict, Union
import re
import adalflow as adal

template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""

system_prompt_start = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."


@adal.func_to_data_component
def parse_integer_answer(answer: str) -> str:
    try:
        numbers = re.findall(r"\d+", answer)
        if numbers:
            answer = numbers[-1]
        else:
            answer = ""
    except ValueError:
        answer = ""

    return answer


class GSM8KTask(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            data=system_prompt_start,
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
        )
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "system_prompt": system_prompt,
            },
            template=template,
            output_processors=parse_integer_answer,
            use_cache=True,
        )

    def bicall(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.generator(prompt_kwargs={"input_str": question}, id=id)
        return output


if __name__ == "__main__":
    from adalflow.utils import setup_env
    from adalflow.datasets.gsm8k import GSM8K

    setup_env()

    from use_cases.config import gpt_3_model

    task = GSM8KTask(**gpt_3_model)

    train_dataset = GSM8K(split="train", size=10)

    print("example: ", train_dataset[0])

    output = task(question=train_dataset[0].question)
    print("output: ", output)
