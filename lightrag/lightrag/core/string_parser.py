"""Extract and convert JSON, YAML, and list strings to Python objects.

It can be used as output_processor for generator.

LLM applications requires lots of string processing. Such as the text output needed to be parsed into:
(1) JSON format or other formats
(2) SQL/Python valid format
(3) Tool(function) call format

We design this these string_parser modules to be generic to any input text without differentiating them as input text or output text.
"""

from typing import Dict, List, Union
import logging

from lightrag.core.component import Component
import lightrag.core.functional as F

log = logging.getLogger(__name__)

LIST_PARSER_OUTPUT_TYPE = List[object]


class ListParser(Component):
    __doc__ = r"""Extracts list `[...]` strings from text and parses them into a list object.

    Args:
        add_missing_right_bracket (bool, optional): Add a missing right bracket to the list string. Defaults to True.


    Examples:

    .. code-block:: python

        list_parser = ListParser()
        test_input_4 = 'Some random text before ["item1", "item2"] and more after'
        print(list_parser(test_input_4))  # Expected to extract ["item1", "item2"]
    """

    def __init__(self, add_missing_right_bracket: bool = True):
        super().__init__()
        self.add_missing_right_bracket = add_missing_right_bracket

    def __call__(self, input: str) -> LIST_PARSER_OUTPUT_TYPE:
        input = input.strip()
        try:
            list_str = F.extract_list_str(input, self.add_missing_right_bracket)
            list_obj = F.parse_json_str_to_obj(list_str)
            return list_obj
        except Exception as e:
            raise ValueError(f"Error: {e}")


JSON_PARSER_OUTPUT_TYPE = Union[Dict[str, object], List[object]]


class JsonParser(Component):
    __doc__ = r"""Extracts JSON strings `{...}` or `[...]` from text and parses them into a JSON object.

    It can output either a dictionary or a list as they are both valid JSON objects.


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
        try:
            json_str = F.extract_json_str(input, self.add_missing_right_brace)
            log.debug(f"json_str: {json_str}")
            json_obj = F.parse_json_str_to_obj(json_str)
            return json_obj
        except Exception as e:
            raise ValueError(f"Error: {e}")


YAML_PARSER_OUTPUT_TYPE = JSON_PARSER_OUTPUT_TYPE


class YamlParser(Component):
    __doc__ = r"""To extract YAML strings from text and parse them into a YAML object.

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
        try:
            yaml_str = F.extract_yaml_str(input)
            yaml_obj = F.parse_yaml_str_to_obj(yaml_str)
            return yaml_obj
        except Exception as e:
            raise ValueError(f"Error: {e}")
