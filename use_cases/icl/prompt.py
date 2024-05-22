CLASSIFICATION_TASK_DESC = r"""You are a classifier. Given the text, you need to classify it into one of the classes.
{% for class in classes %}
{{loop.index-1}}. {{class}}
{% endfor %}
"""
OUTPUT_FORMAT_STR = r"""You will output only the index of the class.
- Do not output the class name.
"""
