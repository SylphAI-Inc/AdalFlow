# test_output_parsers.py
import unittest
from dataclasses import dataclass, field
from adalflow.components.output_parsers.outputs import (
    JsonOutputParser,
    YamlOutputParser,
)
from adalflow.core.base_data_class import DataClass


@dataclass
class User(DataClass):
    id: int = field(default=1, metadata={"description": "User ID"})
    name: str = field(default="John", metadata={"description": "User name"})


class TestOutputParsers(unittest.TestCase):

    def setUp(self):
        self.user_example = User(id=1, name="John")
        self.json_user_to_parse = '{"id": 2, "name": "Jane"}'
        self.yaml_user_to_parse = "id: 2\nname: Jane"

    def test_json_output_parser_without_dataclass(self):
        parser = JsonOutputParser(data_class=User, examples=[self.user_example])
        parsed_user = parser(self.json_user_to_parse)
        expected_output = {"id": 2, "name": "Jane"}
        self.assertEqual(parsed_user, expected_output)

    def test_json_output_parser_with_dataclass(self):
        parser = JsonOutputParser(
            data_class=User, examples=[self.user_example], return_data_class=True
        )
        parsed_user = parser(self.json_user_to_parse)
        self.assertIsInstance(parsed_user, User)
        self.assertEqual(parsed_user.id, 2)
        self.assertEqual(parsed_user.name, "Jane")

    def test_yaml_output_parser_without_dataclass(self):
        parser = YamlOutputParser(data_class=User, examples=[self.user_example])
        parsed_user = parser(self.yaml_user_to_parse)
        expected_output = {"id": 2, "name": "Jane"}
        self.assertEqual(parsed_user, expected_output)

    def test_yaml_output_parser_with_dataclass(self):
        parser = YamlOutputParser(
            data_class=User, examples=[self.user_example], return_data_class=True
        )
        parsed_user = parser(self.yaml_user_to_parse)
        self.assertIsInstance(parsed_user, User)
        self.assertEqual(parsed_user.id, 2)
        self.assertEqual(parsed_user.name, "Jane")

    # Exception test cases
    def test_json_output_parser_invalid_data(self):
        parser = JsonOutputParser(data_class=User, examples=[self.user_example])
        invalid_json_data = "invalid json"
        with self.assertRaises(Exception):
            parser(invalid_json_data)

    def test_yaml_output_parser_invalid_data(self):
        parser = YamlOutputParser(data_class=User, examples=[self.user_example])
        invalid_yaml_data = "invalid: yaml: data"
        with self.assertRaises(Exception):
            parser(invalid_yaml_data)


if __name__ == "__main__":
    unittest.main()
