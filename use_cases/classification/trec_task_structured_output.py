from typing import Dict, Union, Optional

import adalflow as adal
from adalflow.datasets.trec import _COARSE_LABELS_DESC, _COARSE_LABELS
from use_cases.classification.trec_task import task_desc_template
from use_cases.classification.data import TRECExtendedData

template = r"""<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}
{% if output_format_str is not none %}
{{output_format_str}}
{% endif %}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
{{input_str}}
<END_OF_USER_MESSAGE>
"""


class TRECClassifierStructuredOutput(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData
        self.data_class.set_task_desc(task_desc_str)

        self.parser = adal.DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="yaml"
        )

        prompt_kwargs = {
            # "system_prompt": adal.Parameter(
            #     data=self.parser.get_task_desc_str()
            #     + "\n"
            #     + self.parser.get_output_format_str(),
            #     role_desc="Task description with output format requirements",
            #     requires_opt=True,
            #     param_type=adal.ParameterType.PROMPT,
            # ),
            # NOTE: when the instruction is too long,
            # it is better to split it into two prompts it is more effective at training
            # 0.8056 val, 0.903 test
            "system_prompt": adal.Parameter(
                data=self.parser.get_task_desc_str(),
                # data="You are a classifier. Given a question, classify it into one of the following classes based on what the question is seeking:\n\nFormat: class_index. class_name, class_description\n\n0. ABBR, Abbreviation\n1. ENTY, Entity\n2. DESC, Description and abstract concept\n3. HUM, Human being\n4. LOC, Location\n5. NUM, Numeric value\n\nPay close attention to whether a question asks for specific terms, traditions, entities, or people, versus a general description or numerical detail. Do not try to answer the question:",
                # data="You are a classifier. Given a question, classify it into one of the following classes based on what the question is seeking:\n\nFormat: class_index. class_name, class_description\n\n0. ABBR, Abbreviation\n1. ENTY, Entity\n2. DESC, Description and abstract concept\n3. HUM, Human being\n4. LOC, Location\n5. NUM, Numeric value\n\nPay special attention to questions about entities versus descriptions, as well as those asking for specific terms or people. Do not try to answer the question:",
                # best  # data="You are a classifier. For each question given, classify it into one of the following classes:\n\nFormat: class_index. class_name, class_description\n\n0. ABBR, Abbreviation (includes initials)\n1. ENTY, Entity (includes products, languages, objects, etc.)\n2. DESC, Description and abstract concept (includes explanations)\n3. HUM, Human being (includes individuals, groups, etc.)\n4. LOC, Location (includes addresses, places, etc.)\n5. NUM, Numeric value (includes distances, dates, ages, etc.)\n\n- Focus on identifying the primary subject of the question and classifying based on what is being explicitly asked for.",
                role_desc="Task description",
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),
            "output_format_str": adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="Output format requirements",
                requires_opt=False,
                param_type=adal.ParameterType.PROMPT,
            ),
            # NOTE: 88.19%
            "few_shot_demos": adal.Parameter(
                data=None,
                requires_opt=True,
                role_desc="Few shot examples to help the model",
                param_type=adal.ParameterType.DEMOS,
            ),
        }
        # TODO:
        # mix, sequential (training)

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=self.parser,
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        input_data = self.data_class(question=question)
        input_str = self.parser.get_input_str(input_data)
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
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

    task = TRECClassifierStructuredOutput(**gpt_3_model)
    print(task)

    trainset, valset, testset = load_datasets()
    for data in trainset:
        response = task.call(data.question)
        print(response)
        print(data)

        break
