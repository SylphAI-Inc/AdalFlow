from typing import Dict, Union, Optional

import adalflow as adal
from adalflow.datasets.trec import _COARSE_LABELS_DESC, _COARSE_LABELS
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
task_desc_template = r"""You are a classifier. Given a question, you need to classify it into one of the following classes:
Format: class_index. class_name, class_description
{% if classes %}
{% for class in classes %}
{{loop.index-1}}. {{class.label}}, {{class.desc}}
{% endfor %}
{% endif %}
- Do not try to answer the question:
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

        self.parser = adal.DataClassParser(
            data_class=TRECExtendedData, return_data_class=True, format_type="yaml"
        )

        prompt_kwargs = {
            # NOTE: when the instruction is too long,
            # it is better to split it into two prompts it is more effective at training
            # 0.8056 val, 0.903 test
            "system_prompt": adal.Parameter(
                data=task_desc_str,
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
            # "few_shot_demos": adal.Parameter(
            #     data=None,
            #     requires_opt=True,
            #     role_desc="Few shot examples to help the model",
            #     param_type=adal.ParameterType.DEMOS,
            # ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=self.parser,
            use_cache=True,
        )

    def bicall(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm(prompt_kwargs={"input_str": question}, id=id)
        if isinstance(output, adal.Parameter):
            output.data_in_prompt = lambda x: x.data.raw_response
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
