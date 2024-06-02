# OUTPUT_FORMAT_STR = r"""You will output only class_index.
# - Do not output the class_name.
# """


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


# output_example = OutputFormat(
#     thought="Grand Coulee Dam dam is a location",
#     class_index=4,
#     class_name="Location",
# )
# output = get_data_class_schema(OutputFormat)
# print(output)
