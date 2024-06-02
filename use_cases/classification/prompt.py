# OUTPUT_FORMAT_STR = r"""You will output only class_index.
# - Do not output the class_name.
# """


import dataclasses
from typing import Dict, Any
from prompts.outputs import get_data_class_schema
import yaml

OUTPUT_FORMAT_YAML_STR = r"""
The output should be formatted as a standard JSON object with three keys:
```
{
{% if include_thought %}
"thought": "Your reasoning to classify the question to class_name",
{% endif %}
"class_name": "class_name",
"class_index": class_index(int)
}
- Quote the string values correctly!
```
"""
# {#"thought": "Your reasoning to classify the question to class_name",#}

# from core.data_classes import BaseDataClass
# from use_cases.classification.data import _COARSE_LABELS_DESC, _COARSE_LABELS


# @dataclasses.dataclass
# class InputFormat(BaseDataClass):
#     # add the "prompt_arg" to represent the prompt argument that it should get matched to
#     question: str = dataclasses.field(metadata={"desc": "The question to classify"})

#     @classmethod
#     def load_from_dict(cls, data: Dict[str, Any]):
#         # customize to convert data item from a dataset into input data object
#         # "text" -> "question"
#         data = {"question": data["text"]}
#         return super().load_from_dict(data)


# @dataclasses.dataclass
# class OutputFormat(BaseDataClass):
#     thought: str = dataclasses.field(
#         metadata={
#             "desc": "Your reasoning to classify the question to class_name",
#         }
#     )
#     class_name: str = dataclasses.field(metadata={"desc": "class_name"})

#     class_index: int = dataclasses.field(
#         metadata={"desc": "class_index in range[0, 5]"}
#     )

#     @classmethod
#     def load_from_dict(cls, data: Dict[str, Any]):
#         # customize to convert data item from a dataset into output data object
#         # "label" -> "class_index"
#         data = {
#             "thought": None,
#             "class_index": data["coarse_label"],
#             "class_name": _COARSE_LABELS_DESC[data["coarse_label"]],
#         }
#         return super().load_from_dict(data)


# output_example = OutputFormat(
#     thought="Grand Coulee Dam dam is a location",
#     class_index=4,
#     class_name="Location",
# )
# output = get_data_class_schema(OutputFormat)
# print(output)
