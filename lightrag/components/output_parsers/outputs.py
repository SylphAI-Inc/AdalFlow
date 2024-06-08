"""The most commonly used output parsers for the Generator.

Note: Even with OutputParser for output_format_str formatting and the response parsing, it is not 100% guaranteed 
as user query can impact the output. Test your code well!
"""

from dataclasses import is_dataclass
from typing import Dict, Any, Optional
import yaml
import logging

from lightrag.core.component import Component
from lightrag.core.prompt_builder import Prompt
from lightrag.core.string_parser import YAMLParser, ListParser, JsonParser
from lightrag.core.base_data_class import DataClass, DataclassFormatType

# TODO: might be worth to parse a list of yaml or json objects. For instance, a list of jokes.
# setup: Why couldn't the bicycle stand up by itself?
# punchline: Because it was two-tired.
#
# setup: What do you call a fake noodle?
# punchline: An impasta.
logger = logging.getLogger(__name__)

JSON_OUTPUT_FORMAT = r"""Your output should be formatted as a standard JSON instance with the following schema:
```
{{schema}}
```
{% if example %}
Here is an example:
```
{{example}}
```
{% endif %}
-Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
-Use double quotes for the keys and string values.
-Follow the JSON formatting conventions.
"""

YAML_OUTPUT_FORMAT = r"""Your output should be formatted as a standard YAML instance with the following schema:
```
{{schema}}
```
{% if example %}
Here is an example:
```
{{example}}
```
{% endif %}

-Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
-Follow the YAML formatting conventions with an indent of 2 spaces. 
-Quote the string values properly.
"""

LIST_OUTPUT_FORMAT = r"""Your output should be formatted as a standard Python list.
-Each element can be of any Python data type such as string, integer, float, list, dictionary, etc.
-You can also have nested lists and dictionaries.
-Please do not add anything other than valid Python list output!
"""


YAML_OUTPUT_PARSER_OUTPUT_TYPE = Dict[str, Any]


class OutputParser(Component):
    __doc__ = r"""The abstract class for all output parsers.

    This interface helps users customize output parsers with consistent interfaces for the Generator.
    Even though you don't always need to subclass it.

    LightRAG uses two core components: 
    1. the Prompt to format output instruction
    2. A string parser component from core.string_parser for response parsing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pass

    def format_instructions(self) -> str:
        r"""Return the formatted instructions to use in prompt for the output format."""
        raise NotImplementedError("This is an abstract method.")

    def call(self, input: str) -> Any:
        r"""Parse the output string to the desired format and return the parsed output."""
        raise NotImplementedError("This is an abstract method.")


class YAMLOutputParser(OutputParser):
    __doc__ = r"""YAML output parser using dataclass for schema extraction.

    Args:
        data_class (Type): The dataclass to extract the schema for the YAML output.
        example (Type, optional): The example dataclass object to show in the prompt. Defaults to None.
        yaml_output_format_template (str, optional): The template for the YAML output format. Defaults to YAML_OUTPUT_FORMAT.
        output_processors (Component, optional): The output processors to parse the YAML string to JSON object. Defaults to YAMLParser().

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
        data_class: DataClass,
        example: DataClass = None,
        template: Optional[str] = None,
        output_processors: Optional[Component] = None,
    ):

        super().__init__()
        if not is_dataclass(data_class):
            raise ValueError(f"Provided class is not a dataclass: {data_class}")

        # ensure example is instance of data class and initiated
        if example is not None and not isinstance(example, data_class):
            raise ValueError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        template = template or YAML_OUTPUT_FORMAT
        self.data_class_for_yaml = data_class
        self.yaml_output_format_prompt = Prompt(template=template)
        self.output_processors = output_processors or YAMLParser()
        self.example = example

    def format_instructions(
        self, format_type: Optional[DataclassFormatType] = None
    ) -> str:
        r"""Return the formatted instructions to use in prompt for the YAML output format.

        Args:
            format_type (DataclassFormatType, optional): The format type to show in the prompt.
                Defaults to DataclassFormatType.SIGNATURE_YAML for less token usage.
                Options: DataclassFormatType.SIGNATURE_YAML, DataclassFormatType.SIGNATURE_JSON, DataclassFormatType.SCHEMA.
        """
        format_type = format_type or DataclassFormatType.SIGNATURE_YAML
        schema = self.data_class_for_yaml.format_str(format_type=format_type)
        # convert example to string, convert data class to yaml string
        try:
            example_str = self.example.format_str(
                format_type=DataclassFormatType.EXAMPLE_YAML
            )
            logger.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception:
            example_str = None

        return self.yaml_output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> YAML_OUTPUT_PARSER_OUTPUT_TYPE:
        r"""Parse the YAML string to JSON object and return the JSON object."""
        return self.output_processors(input)

    def _extra_repr(self) -> str:
        s = f"data_class_for_yaml={self.data_class_for_yaml}"
        return s


class JsonOutputParser(OutputParser):
    def __init__(
        self,
        data_class: DataClass,
        example: DataClass = None,
        template: Optional[str] = None,
        output_processors: Optional[Component] = None,
    ):
        super().__init__()
        if not is_dataclass(data_class):
            raise ValueError(f"Provided class is not a dataclass: {data_class}")

        if example is not None and not isinstance(example, data_class):
            raise ValueError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        template = template or JSON_OUTPUT_FORMAT
        self.data_class_for_json = data_class
        self.json_output_format_prompt = Prompt(template=template)
        self.output_processors = output_processors or JsonParser()
        self.example = example

    def format_instructions(
        self, format_type: Optional[DataclassFormatType] = None
    ) -> str:
        r"""Return the formatted instructions to use in prompt for the JSON output format.

        Args:
            format_type (DataclassFormatType, optional): The format type to show in the prompt.
                Defaults to DataclassFormatType.SIGNATURE_JSON for less token usage compared with DataclassFormatType.SCHEMA.
                Options: DataclassFormatType.SIGNATURE_YAML, DataclassFormatType.SIGNATURE_JSON, DataclassFormatType.SCHEMA.
        """
        format_type = format_type or DataclassFormatType.SIGNATURE_JSON
        schema = self.data_class_for_json.format_str(format_type=format_type)
        try:
            example_str = self.example.format_str(
                format_type=DataclassFormatType.EXAMPLE_JSON
            )
            logger.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception:
            example_str = None
        return self.json_output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> Any:
        return self.output_processors(input)


class ListOutputParser(OutputParser):
    def __init__(self, list_output_format_template: str = LIST_OUTPUT_FORMAT):
        super().__init__()
        self.list_output_format_prompt = Prompt(template=list_output_format_template)
        self.output_processors = ListParser()

    def format_instructions(self) -> str:
        return self.list_output_format_prompt()

    def call(self, input: str) -> list:
        return self.output_processors(input)
