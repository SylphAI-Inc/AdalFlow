"""The most commonly used output parsers for the Generator.

Includes:
- YamlOutputParser: YAML output parser using dataclass for schema extraction.
- JsonOutputParser: JSON output parser using dataclass for schema extraction.
- ListOutputParser: List output parser to parse list of objects from the string.
- BooleanOutputParser: Boolean output parser to parse boolean values from the string.
"""

from dataclasses import is_dataclass
from typing import Dict, Any, Optional, List, Type, Union
import logging
import json

from adalflow.core.component import DataComponent
from adalflow.core.prompt_builder import Prompt
from adalflow.core.string_parser import YamlParser, ListParser, JsonParser
from adalflow.core.base_data_class import DataClass, DataClassFormatType
from adalflow.core.base_data_class import ExcludeType, IncludeType

from pydantic import BaseModel, ValidationError



__all__ = [
    "OutputParser",
    "YamlOutputParser",
    "JsonOutputParser",
    "JsonOutputParserPydanticModel",
    "ListOutputParser",
    "BooleanOutputParser",
]

log = logging.getLogger(__name__)

# TODO: delete examples here
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
**Schema Interpretation:**
   - The "properties" and "type" fields in the schema are NOT the actual JSON keys
   - Generate the correct nested JSON structure using the actual field names shown
   - Follow the exact field names and data types specified in the schema
   - **CRITICAL: Return actual data values, NOT the schema structure itself**
- Output ONLY valid JSON without any markdown formatting or backticks
- Use double quotes for all keys and string values
- Ensure proper JSON syntax with correct comma placement
- Do not include any text before or after the JSON object
- When including string values with newlines, use \\n instead of actual line breaks
- Properly escape special characters: use \\" for quotes, \\\\ for backslashes
- For multiline strings, keep them on a single line with \\n characters
**WARNING:** The JSON must be parseable by standard JSON parsers. Malformed JSON will cause parsing failures. When handling complex text with special characters, quotes, or formatting, prioritize proper escaping over readability.

"""

"""**CRITICAL JSON FORMATTING REQUIREMENTS:**

1. **Schema Interpretation:**
   - The "properties" and "type" fields in the schema are NOT the actual JSON keys
   - Generate the correct nested JSON structure using the actual field names shown
   - Follow the exact field names and data types specified in the schema

2. **Output Requirements:**
   - Output ONLY valid JSON - no markdown, backticks, explanations, or comments
   - Use double quotes for ALL keys and string values
   - Ensure proper JSON syntax with correct comma placement (no trailing commas)
   - Do not include any text before or after the JSON object

3. **String Escaping (CRITICAL for complex text):**
   - Escape double quotes inside strings: \\" 
   - Escape backslashes: \\\\
   - Escape forward slashes if needed: \\/
   - Use \\n for newlines, \\t for tabs, \\r for carriage returns
   - Keep multiline content on single lines with \\n characters

4. **Complex Content Handling:**
   - Unicode characters (émojis, accented letters, Chinese/Arabic): include directly
   - Nested quotes: escape properly (\\"inner quotes\\")
   - Special characters in addresses/names: escape appropriately
   - Preserve formatting information using escape sequences

5. **Data Type Handling:**
   - Strings: always in double quotes with proper escaping
   - Numbers: no quotes (unless schema specifies string type)
   - Booleans: use true/false (lowercase)
   - null values: use null (not undefined or empty string unless specified)
   - Arrays: proper bracket notation with comma separation
   - Objects: proper brace notation with escaped content

6. **Validation Checklist:**
   - All braces {} and brackets [] are properly matched
   - All strings are quoted and special characters escaped
   - No trailing commas after last elements
   - Structure matches schema requirements exactly
   - All required fields are present with correct types

**WARNING:** The JSON must be parseable by standard JSON parsers. Malformed JSON will cause parsing failures. When handling complex text with special characters, quotes, or formatting, prioritize proper escaping over readability.
"""
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
-DO NOT mistaken the "properties" and "type" in the schema as the actual fields in the YAML output.
-Quote the string values properly."""

LIST_OUTPUT_FORMAT = r"""Your output should be formatted as a standard Python list.
- Start the list with '[' and end with ']'
- DO NOT mistaken the "properties" and "type" in the schema as the actual fields in the list output.
"""


YAML_OUTPUT_PARSER_OUTPUT_TYPE = Dict[str, Any]


class OutputParser(DataComponent):
    __doc__ = r"""The abstract class for all output parsers.

    On top of the basic string Parser, it handles structured data interaction:
    1. format_instructions: Return the formatted instructions to use in prompt for the output format.
    2. call: Parse the output string to the desired format and return the parsed output via yaml or json.

    This interface helps users customize output parsers with consistent interfaces for the Generator.
    Even though you don't always need to subclass it.

    AdalFlow uses two core classes:
    1. the Prompt to format output instruction
    2. A string parser from core.string_parser for response parsing.
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
        include_fields: IncludeType = None,
        exclude_fields: ExcludeType = None,
        return_data_class: bool = False,
    ):

        super().__init__()
        if not is_dataclass(data_class):
            raise TypeError(f"Provided class is not a dataclass: {data_class}")

        if not issubclass(data_class, DataClass):
            raise TypeError(
                f"Provided class is not a subclass of DataClass: {data_class}"
            )

        # ensure example is instance of data class and initiated
        if examples is not None and not isinstance(examples[0], data_class):
            raise TypeError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        self._return_data_class = return_data_class
        self._exclude_fields = exclude_fields
        self._include_fields = include_fields
        self.data_class: DataClass = data_class
        self.output_format_prompt = Prompt(template=YAML_OUTPUT_FORMAT)
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
            format_type=format_type,
            exclude=self._exclude_fields,
            include=self._include_fields,
        )
        # convert example to string, convert data class to yaml string
        example_str = ""
        try:
            if self.examples and len(self.examples) > 0:
                for example in self.examples:
                    per_example_str = example.format_example_str(
                        format_type=DataClassFormatType.EXAMPLE_YAML,
                        exclude=self._exclude_fields,
                        include=self._include_fields,
                    )
                    example_str += f"{per_example_str}\n________\n"
                # remove the last new line
                example_str = example_str[:-1]
                log.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception as e:
            log.error(f"Error in formatting example for {__class__.__name__}, {e}")
            example_str = None

        return self.output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> YAML_OUTPUT_PARSER_OUTPUT_TYPE:
        r"""Parse the YAML string to JSON object and return the JSON object."""
        try:
            output_dict = self.output_processors(input)
            if self._return_data_class:
                return self.data_class.from_dict(output_dict)
            return output_dict
        except Exception as e:
            log.error(f"Error in parsing YAML to JSON: {e}")
            raise e

    def _extra_repr(self) -> str:
        s = f"data_class={self.data_class.__name__}, examples={self.examples}, exclude_fields={self._exclude_fields}, \
        include_fields={self._include_fields},\return_data_class={self._return_data_class}"
        return s


class JsonOutputParser(OutputParser):
    def __init__(
        self,
        data_class: DataClass,
        examples: List[DataClass] = None,
        include_fields: IncludeType = None,
        exclude_fields: ExcludeType = None,
        return_data_class: bool = False,
    ):
        super().__init__()
        if not is_dataclass(data_class):
            raise TypeError(f"Provided class is not a dataclass: {data_class}")

        if not issubclass(data_class, DataClass):
            raise TypeError(
                f"Provided class is not a subclass of DataClass: {data_class}"
            )

        if (
            examples is not None
            and len(examples) > 0
            and not isinstance(examples[0], data_class)
        ):
            raise TypeError(
                f"Provided example is not an instance of the data class: {data_class}"
            )
        self._return_data_class = return_data_class
        self._exclude_fields = exclude_fields
        self._include_fields = include_fields
        template = JSON_OUTPUT_FORMAT
        self.data_class: DataClass = data_class
        self.output_format_prompt = Prompt(template=template)
        self.output_processors = JsonParser()
        self.examples = examples

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
        format_type = format_type or DataClassFormatType.SCHEMA
        schema = self.data_class.format_class_str(
            format_type=format_type,
            exclude=self._exclude_fields,
            include=self._include_fields,
        )
        example_str = ""
        try:
            if self.examples and len(self.examples) > 0:
                for example in self.examples:
                    per_example_str = example.format_example_str(
                        format_type=DataClassFormatType.EXAMPLE_JSON,
                        exclude=self._exclude_fields,
                        include=self._include_fields,
                    )
                    example_str += f"{per_example_str}\n________\n"
                # remove the last new line
                example_str = example_str[:-1]
                log.debug(f"{__class__.__name__} example_str: {example_str}")

        except Exception as e:
            log.error(f"Error in formatting example for {__class__.__name__}, {e}")
            example_str = None
        return self.output_format_prompt(schema=schema, example=example_str)

    def call(self, input: str) -> Any:
        try:
            output_dict = self.output_processors(input)
            log.debug(f"{__class__.__name__} output_dict: {output_dict}")

        except Exception as e:
            log.error(f"Error in parsing JSON to JSON: {e}")
            raise e
        try:
            if self._return_data_class:
                return self.data_class.from_dict(output_dict)
            return output_dict
        except Exception as e:
            log.error(f"Error in converting dict to data class: {e}")
            raise e

    def _extra_repr(self) -> str:
        s = f"""data_class={self.data_class.__name__}, examples={self.examples}, exclude_fields={self._exclude_fields}, \
            include_fields={self._include_fields}, return_data_class={self._return_data_class}"""
        return s


class JsonOutputParserPydanticModel(OutputParser):
    """JSON output parser using Pydantic BaseModel for schema extraction and validation.
    
    This parser works with Pydantic BaseModel classes instead of AdalFlow's DataClass,
    providing better JSON schema generation and automatic validation.
    
    Args:
        pydantic_model (Type[BaseModel]): The Pydantic model class to use for schema and validation
        examples (List[BaseModel], optional): Example instances of the Pydantic model. Defaults to None.
        return_pydantic_object (bool, optional): If True, returns parsed Pydantic object. If False, returns dict. Defaults to True.
    
    Examples:
        >>> from pydantic import BaseModel
        >>> from typing import List
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...     emails: List[str]
        >>>
        >>> parser = JsonOutputParserPydanticModel(pydantic_model=User)
        >>> format_instructions = parser.format_instructions()
        >>> # Use in Generator with output_processors=parser
    """
    
    def __init__(
        self,
        pydantic_model: Type[BaseModel],
        examples: Optional[List[BaseModel]] = None,
        return_pydantic_object: bool = True,
    ):
        super().__init__()
        
        
        if not (isinstance(pydantic_model, type) and issubclass(pydantic_model, BaseModel)):
            raise TypeError(
                f"Provided model must be a Pydantic BaseModel class, got: {pydantic_model}"
            )
        
        if examples is not None and len(examples) > 0:
            if not isinstance(examples[0], pydantic_model):
                raise TypeError(
                    f"Provided examples must be instances of {pydantic_model.__name__}"
                )
        
        self.pydantic_model = pydantic_model
        self.examples = examples or []
        self._return_pydantic_object = return_pydantic_object
        
        # Use the same JSON_OUTPUT_FORMAT template as the regular JsonOutputParser
        self.output_format_prompt = Prompt(template=JSON_OUTPUT_FORMAT)
        self.output_processors = JsonParser()
    
    def format_instructions(self) -> str:
        """Return the formatted instructions with Pydantic model schema."""
        # Get JSON schema from Pydantic model
        schema = self._get_pydantic_schema()
        
        # Format examples if provided
        example_str = ""
        try:
            if self.examples and len(self.examples) > 0:
                example_jsons = []
                for example in self.examples:
                    example_json = example.model_dump_json(indent=2)
                    example_jsons.append(example_json)
                example_str = "\n________\n".join(example_jsons)
                log.debug(f"{__class__.__name__} example_str: {example_str}")
        except Exception as e:
            log.error(f"Error in formatting examples for {__class__.__name__}: {e}")
            example_str = None
        
        return self.output_format_prompt(schema=schema, example=example_str)
    
    def _get_pydantic_schema(self) -> str:
        """Generate a JSON schema description from Pydantic model using native functionality."""
        try:
            # Get the JSON schema from Pydantic and format it as JSON string
            json_schema = self.pydantic_model.model_json_schema()
            return json.dumps(json_schema, indent=2)
        except Exception as e:
            log.error(f"Error generating Pydantic schema: {e}")
            return f"Schema for {self.pydantic_model.__name__}"
    
    def call(self, input: str) -> Union[BaseModel, Dict[str, Any]]:
        """Parse JSON string to Pydantic object or dict."""
        try:
            # First parse the JSON string to dict
            output_dict = self.output_processors(input)
            log.debug(f"{__class__.__name__} parsed dict: {output_dict}")
            
        except Exception as e:
            log.error(f"Error parsing JSON string: {e}")
            raise e
        
        if self._return_pydantic_object:
            try:
                # Use Pydantic's validation to create the object
                return self.pydantic_model.model_validate(output_dict)
            except ValidationError as e:
                log.error(f"Pydantic validation error: {e}")
                raise e
            except Exception as e:
                log.error(f"Error creating Pydantic object: {e}")
                raise e
        else:
            return output_dict
    
    def _extra_repr(self) -> str:
        return f"pydantic_model={self.pydantic_model.__name__}, examples={len(self.examples)}, return_pydantic_object={self._return_pydantic_object}"


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
            return output
        except Exception as e:
            # try to do regex matching for boolean values
            log.info(f"Error: {e}")
            output = _parse_boolean_from_str(input)
            return output
