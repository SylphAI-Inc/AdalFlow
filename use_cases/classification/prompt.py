TEMPLATE = r"""{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}
{%if output_format_str %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
{% endif %}
{# example #}
{% if examples_str %}
<EXAMPLES>
{{examples_str}}
</EXAMPLES>
{% endif %}
{{input_label}}: {{input}}
Your output:
"""

CLASSIFICATION_TASK_DESC = r"""You are a classifier. Given a Question, you need to classify it into one of the following classes:
Format: class_index. class_name, class_description
{% for class in classes %}
{{loop.index-1}}. {{class.label}}, {{class.desc}}
{% endfor %}
"""
OUTPUT_FORMAT_STR = r"""You will output only class_index.
- Do not output the class_name.
"""

EXAMPLES_STR = r"""Question: {{input}}
class_name: {{output}} {%if description%}({{description}}){%endif%}, class_index: {{label}}
"""

import dataclasses
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


@dataclasses.dataclass
class OutputFormat:
    thought: str = dataclasses.field(
        metadata={
            "description": "Your reasoning to classify the question to class_name",
        }
    )
    class_index: int = dataclasses.field(
        metadata={"description": "class_index in range[0, 5]"}
    )
    class_name: str = dataclasses.field(metadata={"description": "class_name"})

    @classmethod
    def to_yaml_signature(self) -> str:
        """Generate a YAML signature based on field metadata descriptions."""
        # Create a dictionary to hold the descriptions
        metadata_dict = {}
        # Iterate over the fields of the dataclass
        for f in dataclasses.fields(self):
            # Each field's metadata 'description' is used as the value
            description = f.metadata.get("description", "No description provided")
            metadata_dict[f.name] = description

        # Convert the dictionary to a YAML string
        return yaml.dump(metadata_dict, default_flow_style=False)


output_example = OutputFormat(
    thought="Grand Coulee Dam dam is a location", class_index=4, class_name="Location"
)
# output = get_data_class_schema(OutputFormat)
# print(output)
