from typing import Dict, Union, Optional

import adalflow as adal
import re
from adalflow.datasets.trec import _COARSE_LABELS_DESC, _COARSE_LABELS
from use_cases.classification.trec_task import task_desc_template

template = r"""<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
{{input_str}}
<END_OF_USER_MESSAGE>
"""


@adal.fun_to_component
def extract_class_index_value(text: str, get_feedback=False):
    pattern = re.compile(r"Answer\s*:\s*\$?(\d+)")

    match = pattern.search(text)

    if match:
        if get_feedback:
            return match.group(1), ""
        return match.group(1)
    else:  # process the failure
        print(f"No valid CLASS_INDEX: $VALUE found in the input text: {text}")
        feedback = "No valid CLASS_INDEX: $VALUE found"
        if get_feedback:
            return text, feedback
        return text


class TRECClassifierStringOutput(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=(
                    task_desc_str,
                    r"""\n""",
                    "Respond in two lines: \n",
                    "Rational: Let's think step by step in order to produce the class_index. We ... \n",
                    "Answer: ${CLASS_INDEX} where ${CLASS_INDEX} is the class index you predict",
                ),
                role_desc="Task description with output format requirements",
            ),
        }
        print(prompt_kwargs)

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=adal.Sequential(
                extract_class_index_value, adal.IntParser()
            ),
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=f"question: {question}",
                requires_opt=False,
                role_desc="input to the LLM",
            )
        }
        return prompt_kwargs

    def call(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(question)
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output


if __name__ == "__main__":

    from benchmarks.config import gpt_3_model, load_model
    from use_cases.classification.data import load_datasets

    adal.setup_env()
    gpt_3_model = load_model(**gpt_3_model)

    task = TRECClassifierStringOutput(**gpt_3_model)

    trainset, valset, testset = load_datasets()
    for data in trainset:
        response = task.call(data.question)
        print(response)
        print(data)

        break
