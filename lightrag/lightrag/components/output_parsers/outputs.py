"""The most commonly used output parsers for the Generator."""

from dataclasses import is_dataclass
from typing import Dict, Any, Optional, List
import logging

from lightrag.core.component import Component
from lightrag.core.prompt_builder import Prompt
from lightrag.core.string_parser import YamlParser, ListParser, JsonParser
from lightrag.core.base_data_class import DataClass, DataClassFormatType
from lightrag.core.base_data_class import ExcludeType


__all__ = [
    "OutputParser",
    "YamlOutputParser",
    "JsonOutputParser",
    "ListOutputParser",
    "BooleanOutputParser",
]

log = logging.getLogger(__name__)

JSON_OUTPUT_FORMAT = r"""Your output should be formatted as a standard JSON instance with the following schema:
```
{{schema}}
```
{% if example %}
Examples:
```
{{example}}
```
{% endif %}
-Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
-Use double quotes for the keys and string values.
-Follow the JSON formatting conventions."""

YAML_OUTPUT_FORMAT = r"""Your output should be formatted as a standard YAML instance with the following schema:
```
{{schema}}
```
{% if example %}
Examples:
```
{{example}}
```
{% endif %}

-Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
-Follow the YAML formatting conventions with an indent of 2 spaces.
-Quote the string values properly."""

LIST_OUTPUT_FORMAT = r"""Your output should be formatted as a standard Python list.
- Start the list with '[' and end with ']'"""


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


class YamlOutputParser(OutputParser):
    __doc__ = r"""YAML output parser using dataclass for schema extraction.

    .. note::
        Only use yaml for simple dataclass objects. For complex objects, use JSON.

    Args:
        data_class (Type): The dataclass to extract the schema for the YAML output.
        example (Type, optional): The example dataclass object to show in the prompt. Defaults to None.
        yaml_output_format_template (str, optional): The template for the YAML output format. Defaults to YAML_OUTPUT_FORMAT.
        output_processors (Component, optional): The output processors to parse the YAML string to JSON object. Defaults to YamlParser().

    Examples:

    >>> from prompts.outputs import YamlOutputParser
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
    >>> yaml_parser = YamlOutputParser(data_class_for_yaml=ThoughtAction)
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
        examples: List[DataClass] = None,
        exclude_fields: ExcludeType = None,
    ):

        super().__init__()
        if not is_dataclass(data_class):
            raise ValueError(f"Provided class is not a dataclass: {data_class}")

        if not issubclass(data_class, DataClass):
            raise ValueError(
                f"Provided class is not a subclass of DataClass: {data_class}"
            )

        # ensure example is instance of data class and initiated
        if examples is not None and not isinstance(examples[0], data_class):
            raise ValueError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        self._exclude_fields = exclude_fields
        self.data_class: DataClass = data_class
        self.yaml_output_format_prompt = Prompt(template=YAML_OUTPUT_FORMAT)
        self.output_processors = YamlParser()
        self.examples = examples

    def format_instructions(
        self,
        format_type: Optional[DataClassFormatType] = None,
    ) -> str:
        r"""Return the formatted instructions to use in prompt for the YAML output format.

        Args:
            format_type (DataClassFormatType, optional): The format type to show in the prompt.
                Defaults to DataClassFormatType.SIGNATURE_YAML for less token usage.
                Options: DataClassFormatType.SIGNATURE_YAML, DataClassFormatType.SIGNATURE_JSON, DataClassFormatType.SCHEMA.
            exclude (List[str], optional): The fields to exclude from the schema of the data class.
        """
        format_type = format_type or DataClassFormatType.SIGNATURE_YAML
        schema = self.data_class.format_class_str(
            format_type=format_type, exclude=self._exclude_fields
        )
        # convert example to string, convert data class to yaml string
        example_str = ""
        try:
            if self.examples and len(self.examples) > 0:
                for example in self.examples:
                    per_example_str = example.format_example_str(
                        format_type=DataClassFormatType.EXAMPLE_YAML,
                        exclude=self._exclude_fields,
                    )
                    example_str += f"{per_example_str}\n________\n"
                # remove the last new line
                example_str = example_str[:-1]
                log.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception as e:
            log.error(f"Error in formatting example for {__class__.__name__}, {e}")
            example_str = None

        return self.yaml_output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> YAML_OUTPUT_PARSER_OUTPUT_TYPE:
        r"""Parse the YAML string to JSON object and return the JSON object."""
        return self.output_processors(input)

    def _extra_repr(self) -> str:
        s = f"data_class={self.data_class.__name__}, examples={self.examples}, exclude_fields={self._exclude_fields}"
        return s


class JsonOutputParser(OutputParser):
    def __init__(
        self,
        data_class: DataClass,
        examples: List[DataClass] = None,
        exclude_fields: ExcludeType = None,
    ):
        super().__init__()
        if not is_dataclass(data_class):
            raise ValueError(f"Provided class is not a dataclass: {data_class}")

        if not issubclass(data_class, DataClass):
            raise ValueError(
                f"Provided class is not a subclass of DataClass: {data_class}"
            )

        if examples is not None and not isinstance(examples[0], data_class):
            raise ValueError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        self._exclude_fields = exclude_fields
        template = JSON_OUTPUT_FORMAT
        self.data_class: DataClass = data_class
        self.json_output_format_prompt = Prompt(template=template)
        self.output_processors = JsonParser()
        self.examples = examples

    # TODO: make exclude works with both
    def format_instructions(
        self,
        format_type: Optional[DataClassFormatType] = None,
    ) -> str:
        r"""Return the formatted instructions to use in prompt for the JSON output format.

        Args:
            format_type (DataClassFormatType, optional): The format type to show in the prompt.
                Defaults to DataClassFormatType.SIGNATURE_JSON for less token usage compared with DataClassFormatType.SCHEMA.
                Options: DataClassFormatType.SIGNATURE_YAML, DataClassFormatType.SIGNATURE_JSON, DataClassFormatType.SCHEMA.
        """
        format_type = format_type or DataClassFormatType.SIGNATURE_JSON
        schema = self.data_class.format_class_str(
            format_type=format_type, exclude=self._exclude_fields
        )
        example_str = ""
        try:
            if self.examples and len(self.examples) > 0:
                for example in self.examples:
                    per_example_str = example.format_example_str(
                        format_type=DataClassFormatType.EXAMPLE_JSON,
                        exclude=self._exclude_fields,
                    )
                    example_str += f"{per_example_str}\n________\n"
                # remove the last new line
                example_str = example_str[:-1]
                log.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception as e:
            log.error(f"Error in formatting example for {__class__.__name__}, {e}")
            example_str = None
        return self.json_output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> Any:
        return self.output_processors(input)

    def _extra_repr(self) -> str:
        s = f"""data_class={self.data_class.__name__}, examples={self.examples}, exclude_fields={self._exclude_fields}"""
        return s


class ListOutputParser(OutputParser):
    __doc__ = r"""List output parser to parse list of objects from the string."""

    def __init__(self, list_output_format_template: str = LIST_OUTPUT_FORMAT):
        super().__init__()
        self.list_output_format_prompt = Prompt(template=list_output_format_template)
        self.output_processors = ListParser()

    def format_instructions(self) -> str:
        return self.list_output_format_prompt()

    def call(self, input: str) -> list:
        return self.output_processors(input)


def _parse_boolean_from_str(input: str) -> Optional[bool]:
    input = input.strip()
    if "true" in input.lower():
        return True
    elif "false" in input.lower():
        return False
    else:
        return None


class BooleanOutputParser(OutputParser):
    __doc__ = r"""Boolean output parser to parse boolean values from the string."""

    def __init__(self):
        super().__init__()
        self.output_processors = None

    def format_instructions(self) -> str:
        return "The output should be a boolean value. True or False."

    def call(self, input: str) -> bool:

        input = input.strip()
        output = None
        # evaluate the expression to get the boolean value
        try:
            output = eval(input)
            if isinstance(output, bool):
                return output
            # go to string parsing
            output = _parse_boolean_from_str(input)
            if output is not None:
                return output
        except Exception as e:
            # try to do regex matching for boolean values
            log.info(f"Error: {e}")
            output = _parse_boolean_from_str(input)
            if output is not None:
                return output
        # when parsing is failed
        return None
