TEMPLATE = r"""{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}
{%if output_format_str %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
{endif}
{% endif %}
{# example #}
{% if examples_str %}
<EXAMPLES>
{{examples_str}}
</EXAMPLES>
{% endif %}
Text to classify: {{input}}
Your output(index of the class only, int):
"""

CLASSIFICATION_TASK_DESC = r"""You are a classifier. Given the user query, you need to classify it into one of the classes.
{% for class in classes %}
{{loop.index-1}}. {{class}}
{% endfor %}
"""
OUTPUT_FORMAT_STR = r"""You will output only the index of the class.
- Do not output the class name.
"""

EXAMPLES_STR = r"""input: {{input}}
output: {{output}} {%if description%}({{description}}){%endif%}
"""
