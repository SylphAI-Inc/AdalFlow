"""Extract and convert common string to Python objects.

From simple data types like boolean, integer, and float to more complex data types like JSON, YAML, and list strings."""

from typing import Dict, List, Union
import logging

from adalflow.core.component import Component
import adalflow.core.functional as F

log = logging.getLogger(__name__)

BOOLEAN_PARSER_OUTPUT_TYPE = bool


class Parser(Component):
    __doc__ = r"""Base class for all string parsers."""

    def __init__(self):
        super().__init__()

    def call(self, input: str) -> object:
        raise NotImplementedError(
            "Parser subclasses must implement the __call__ method"
        )


class BooleanParser(Parser):
    __doc__ = r"""Extracts boolean values from text.

    Examples:

    .. code-block:: python

        boolean_parser = BooleanParser()
        test_input_1 = "True" # or "true" or "...true..."
        print(boolean_parser(test_input_1))  # Expected to extract True
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, input: str) -> BOOLEAN_PARSER_OUTPUT_TYPE:
        input = input.strip()
        try:
            return F.extract_first_boolean(input)
        except Exception as e:
            raise ValueError(f"Error: {e}")


INT_PARSER_OUTPUT_TYPE = int


class IntParser(Parser):
    __doc__ = r"""Extracts integer values from text.

    Returns:
        int: Extracted integer value.

    Raises:
        ValueError: If the input text does not contain an integer

    Examples:

    .. code-block:: python

        int_parser = IntParser()
        test_input_2 = "123" # or "...123..."
        print(int_parser(test_input_2))  # Expected to extract 123
    """

    def __init__(self):
        super().__init__()

    def call(self, input: str) -> INT_PARSER_OUTPUT_TYPE:
        input = input.strip()
        try:
            return F.extract_first_int(input)
        except Exception as e:
            raise ValueError(f"Error: {e}")


FLOAT_PARSER_OUTPUT_TYPE = float


class FloatParser(Parser):
    __doc__ = r"""Extracts float values from text.

    Returns:
        float: Extracted float value.

    Raises:
        ValueError: If the input text does not contain a float

    Examples:

    .. code-block:: python

        float_parser = FloatParser()
        test_input_3 = "123.45" # or "...123.45..."
        print(float_parser(test_input_3))  # Expected to extract 123.45
    """

    def __init__(self):
        super().__init__()

    def call(self, input: str) -> FLOAT_PARSER_OUTPUT_TYPE:
        input = input.strip()
        try:
            return F.extract_first_float(input)
        except Exception as e:
            raise ValueError(f"Error: {e}")


LIST_PARSER_OUTPUT_TYPE = List[object]


class ListParser(Parser):
    __doc__ = r"""Extracts list `[...]` strings from text and parses them into a list object.

    Args:
        add_missing_right_bracket (bool, optional): Add a missing right bracket to the list string. Defaults to True.

    Returns:
        List[object]: Extracted list object.

    Raises:
        ValueError: If the input text does not contain a list

    Examples:

    .. code-block:: python

        list_parser = ListParser()
        test_input_4 = 'Some random text before ["item1", "item2"] and more after'
        print(list_parser(test_input_4))  # Expected to extract ["item1", "item2"]
    """

    def __init__(self, add_missing_right_bracket: bool = True):
        super().__init__()
        self.add_missing_right_bracket = add_missing_right_bracket

    def call(self, input: str) -> LIST_PARSER_OUTPUT_TYPE:
        input = input.strip()
        list_str = None
        # Extract list string
        try:
            list_str = F.extract_list_str(input, self.add_missing_right_bracket)

        except Exception as e:
            raise ValueError(f"Error at extracting list string: {e}")

        # Parse list string with json.loads and yaml.safe_load
        try:
            list_obj = F.parse_json_str_to_obj(list_str)
            return list_obj
        except Exception as e:
            log.error(f"Error at parsing list string with json.loads: {e}")
            raise ValueError(f"Error: {e}")


JSON_PARSER_OUTPUT_TYPE = Union[Dict[str, object], List[object]]


class JsonParser(Parser):
    __doc__ = r"""Extracts JSON strings `{...}` or `[...]` from text and parses them into a JSON object.

    It can output either a dictionary or a list as they are both valid JSON objects.

    Args:
        add_missing_right_brace (bool, optional): Add a missing right brace to the JSON string. Defaults to True.

    Returns:
        Union[Dict[str, object], List[object]]: Extracted JSON object.

    Raises:
        ValueError: If the input text does not contain a JSON object


    Examples:

    .. code-block:: python

        json_parser = JsonParser()
        json_str = "```json\n{\"key\": \"value\"}\n```"
        json_obj = json_parser(json_str)
        print(json_obj)  # Expected to extract {"key": "value"}
    """

    def __init__(self, add_missing_right_brace: bool = True):
        super().__init__()
        self.add_missing_right_brace = add_missing_right_brace

    def call(self, input: str) -> JSON_PARSER_OUTPUT_TYPE:
        input = input.strip()
        # Extract JSON string
        json_str = None
        try:
            json_str = F.extract_json_str(input, self.add_missing_right_brace)
            log.debug(f"json_str: {json_str}")

        except Exception as e:
            raise ValueError(f"Error: {e}")
        # Parse JSON string with json.loads and yaml.safe_load
        try:
            json_obj = F.parse_json_str_to_obj(json_str)
            log.debug(f"json_obj: {json_obj}")
            return json_obj
        except Exception as e:
            log.error(f"Error at parsing JSON string: {e}")
            raise ValueError(f"Error: {e}")


YAML_PARSER_OUTPUT_TYPE = JSON_PARSER_OUTPUT_TYPE


class YamlParser(Parser):
    __doc__ = r"""To extract YAML strings from text and parse them into a YAML object.

    Returns:
        JSON_PARSER_OUTPUT_TYPE: Extracted YAML object.

    Raises:
        ValueError: If the input text does not contain a YAML object

    Examples:

    .. code-block:: python

        yaml_parser = YamlParser()
        yaml_str = "```yaml\nkey: value\n```"
        yaml_obj = yaml_parser(yaml_str)
        print(yaml_obj)  # Expected to extract {"key": "value"}
    """

    def __init__(self):
        super().__init__()

    def call(self, input: str) -> YAML_PARSER_OUTPUT_TYPE:
        input = input.strip()
        # parse YAML string with yaml.safe_load
        try:
            yaml_str = F.extract_yaml_str(input)
            yaml_obj = F.parse_yaml_str_to_obj(yaml_str)
            return yaml_obj
        except Exception as e:
            raise ValueError(f"Error: {e}")
