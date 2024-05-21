from dataclasses import is_dataclass
from typing import Dict, Any, Type, Optional

from core.component import Component
from core.prompt_builder import Prompt
from core.functional import get_data_class_schema
from core.string_parser import YAMLParser

JSON_OUTPUT_FORMAT = r""""""
YAML_OUTPUT_FORMAT = r"""The output should be formatted as a standard YAML instance with the following JSON schema:
```
{{schema}}
```
-Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
-Follow the YAML formatting conventions with an indent of 2 spaces.
"""
LIST_OUTPUT_FORMAT = r""""""


YAML_OUTPUT_PARSER_OUTPUT_TYPE = Dict[str, Any]


class YAMLOutputParser(Component):
    __doc__ = r"""YAML output parser using dataclass for schema extraction.

    Examples:

    >>> from prompts.outputs import YAMLOutputParser
    >>> from dataclasses import dataclass, field
    >>> from typing import List
    >>>
    >>> @dataclass
    >>> class ThoughtAction:
    >>>     thought: str = field(metadata={"description": "Reasoning behind the answer"}) # required field
    >>>     answer: str = field(metadata={"description": "Your answer to the question"}, default=None) # optional field
    >>>
    >>> # If you want to parse it back to the dataclass, you can add a from_dict method to the dataclass
    >>> # def from_dict(self, d: Dict[str, Any]) -> "ThoughtAction":
    >>> #     return ThoughtAction(**d)
    >>>
    >>> yaml_parser = YAMLOutputParser(data_class_for_yaml=ThoughtAction)
    >>> yaml_format_instructions = yaml_parser.format_instructions()
    >>> print(yaml_format_instructions)
    >>> yaml_str = '''The output should be formatted as a standard YAML instance with the following JSON schema:
    >>> ```
    >>> 'thought': {'type': 'str', 'description': 'Reasoning behind the answer', 'required': True}, 'answer': {'type': 'str', 'description': '
    >>> Your answer to the question', 'required': False, 'default': None}
    >>> ```
    >>> -Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
    >>> -Follow the YAML formatting conventions with an indent of 2 spaces.
    >>> '''
    >>> # use it in the generator
    >>> task_desc_str = "You are a helpful assistant who answers user query. "+yaml_format_instructions 
    >>> generator = Generator(output_processors=yaml_parser, ..., preset_prompt_kwargs={"task_desc_str": task_desc_str})
    >>> generator("Should i be a doctor?")
    """

    def __init__(
        self,
        data_class_for_yaml: Type,
        yaml_output_format_template: Optional[str] = YAML_OUTPUT_FORMAT,
        output_processors: Optional[Component] = YAMLParser(),
    ):
        super().__init__()
        if not is_dataclass(data_class_for_yaml):
            raise ValueError(
                f"Provided class is not a dataclass: {data_class_for_yaml}"
            )
        self.data_class_for_yaml = data_class_for_yaml
        self.yaml_output_format_prompt = Prompt(template=yaml_output_format_template)
        self.output_processors = output_processors

    def format_instructions(self) -> str:
        r"""Return the formatted instructions to use in prompt for the YAML output format."""
        schema = get_data_class_schema(self.data_class_for_yaml)
        return self.yaml_output_format_prompt(schema=schema)

    def call(self, input: str) -> YAML_OUTPUT_PARSER_OUTPUT_TYPE:
        r"""Parse the YAML string to JSON object and return the JSON object."""
        return self.output_processors(input)

    def _extra_repr(self) -> str:
        s = f"data_class_for_yaml={self.data_class_for_yaml}"
        return s
