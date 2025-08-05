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

    __input_fields__ = ["id", "name"]


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
    
    # Test JSON wrapped in markdown code blocks
    def test_json_output_parser_with_markdown_blocks(self):
        parser = JsonOutputParser(data_class=User, examples=[self.user_example])
        
        # Test with ```json
        json_with_markdown = '''```json
{"id": 3, "name": "Bob"}
```'''
        parsed_user = parser(json_with_markdown)
        expected_output = {"id": 3, "name": "Bob"}
        self.assertEqual(parsed_user, expected_output)
        
        # Test with just ```
        json_with_simple_markdown = '''```
{"id": 4, "name": "Alice"}
```'''
        parsed_user = parser(json_with_simple_markdown)
        expected_output = {"id": 4, "name": "Alice"}
        self.assertEqual(parsed_user, expected_output)
    
    def test_json_output_parser_with_markdown_blocks_and_dataclass(self):
        parser = JsonOutputParser(
            data_class=User, examples=[self.user_example], return_data_class=True
        )
        json_with_markdown = '''```json
{"id": 5, "name": "Charlie"}
```'''
        parsed_user = parser(json_with_markdown)
        self.assertIsInstance(parsed_user, User)
        self.assertEqual(parsed_user.id, 5)
        self.assertEqual(parsed_user.name, "Charlie")
    
    def test_json_output_parser_complex_case_from_error(self):
        """Test the exact case that was causing the error in the user's example"""
        
        # Create a more complex dataclass for testing
        @dataclass
        class AgentOutput(DataClass):
            thought: str = field(default="", metadata={"description": "Reasoning"})
            name: str = field(default="", metadata={"description": "Function name"})
            kwargs: dict = field(default_factory=dict, metadata={"description": "Function kwargs"})
        
        parser = JsonOutputParser(data_class=AgentOutput)
        
        # This is similar to the error case from the user
        complex_json = '''```
{
    "thought": "I need to add the import for Link and update the Careers page. Let me add the import statement at the top of the file.",
    "name": "EditFileTool_acall",
    "kwargs": {
        "file_path": "/Users/test/src/pages/Careers.tsx",
        "old_string": "const Careers = () => {",
        "new_string": "import { Link } from 'react-router-dom';\\n\\nconst Careers = () => {"
    }
}
```'''
        
        parsed_output = parser(complex_json)
        self.assertEqual(parsed_output["thought"], "I need to add the import for Link and update the Careers page. Let me add the import statement at the top of the file.")
        self.assertEqual(parsed_output["name"], "EditFileTool_acall")
        self.assertEqual(parsed_output["kwargs"]["file_path"], "/Users/test/src/pages/Careers.tsx")
    
    def test_json_output_parser_exact_error_case(self):
        """Test the EXACT error case from the user's logs"""
        
        @dataclass
        class WebThinkerOutput(DataClass):
            thought: str = field(default="", metadata={"description": "Reasoning"})
            name: str = field(default="", metadata={"description": "Function name"})
            kwargs: dict = field(default_factory=dict, metadata={"description": "Function kwargs"})
        
        parser = JsonOutputParser(data_class=WebThinkerOutput)
        
        # This is the exact JSON string from the error message (note the escaped newlines)
        exact_error_json = r'''{
    "thought": "I need to add the import for Link and update the Careers page. Let me add the import statement at the top of the file.",
    "name": "EditFileTool_acall",
    "kwargs": {
        "file_path": "/Users/liyin/Documents/test/ai-girl-glow/src/pages/Careers.tsx",
        "old_string": "const Careers = () => {",
        "new_string": "import { Link } from 'react-router-dom';\n\nconst Careers = () => {"
    }
}
```}'''
        
        # The parser should handle this even with the trailing ```}
        parsed_output = parser(exact_error_json)
        self.assertEqual(parsed_output["thought"], "I need to add the import for Link and update the Careers page. Let me add the import statement at the top of the file.")
        self.assertEqual(parsed_output["name"], "EditFileTool_acall")


if __name__ == "__main__":
    unittest.main()
