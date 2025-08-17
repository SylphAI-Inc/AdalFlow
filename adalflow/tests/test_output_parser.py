# test_output_parsers.py
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from adalflow.components.output_parsers.outputs import (
    JsonOutputParser,
    YamlOutputParser,
    JsonOutputParserPydanticModel,
)
from adalflow.core.base_data_class import DataClass


@dataclass
class User(DataClass):
    id: int = field(default=1, metadata={"description": "User ID"})
    name: str = field(default="John", metadata={"description": "User name"})

    __input_fields__ = ["id", "name"]


# Pydantic models for testing JsonOutputParserPydanticModel
class UserPydantic(BaseModel):
    id: int = Field(description="User ID")
    name: str = Field(description="User name")
    age: Optional[int] = Field(default=None, description="User age")
    emails: List[str] = Field(default_factory=list, description="User email addresses")


class ThoughtAction(BaseModel):
    thought: str = Field(description="Reasoning behind the action")
    action: str = Field(description="The action to take")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level")


class ComplexOutput(BaseModel):
    """A complex output model for testing nested structures"""
    thought: str = Field(description="Reasoning")
    name: str = Field(description="Function name")
    kwargs: dict = Field(default_factory=dict, description="Function arguments")
    metadata: Optional[UserPydantic] = Field(default=None, description="Optional metadata")


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
    
    def test_json_output_parser_action_with_embedded_bash(self):
        """Test the exact failing case with Action dataclass containing bash code blocks"""
        
        @dataclass
        class Action(DataClass):
            thought: str = field(metadata={"description": "Reasoning behind the action"})
            _is_answer_final: bool = field(default=False, metadata={"description": "Whether this is the final answer"})
            _answer: str = field(default="", metadata={"description": "The final answer"})
        
        parser = JsonOutputParser(data_class=Action)
        
        # This is the exact JSON string that's failing to parse
        action_json = r'''{"thought": "The user is showing me the error output from running the setup script where Ollama is installed but the server daemon isn't running, causing 'ollama server not responding' errors. I already updated their setup script in step 6 to handle this automatically. Since I'm at step 25/30, I should provide a final answer with the immediate solution.", "_is_answer_final": true, "_answer": "The error shows Ollama is installed but the server daemon isn't running. I've already updated your setup script to handle this automatically. Here's the immediate solution:\n\n**Quick Fix:**\n```bash\n# Start Ollama server in background\nollama serve &\nsleep 5\n# Now run the updated setup script\n./setup_gpt_oss.sh\n```\n\n**Alternative - Use separate terminals:**\n```bash\n# Terminal 1: Start server\nollama serve\n\n# Terminal 2: Run setup\n./setup_gpt_oss.sh\n```\n\nThe updated setup script now automatically checks if the Ollama server is running and starts it if needed. The core issue is that while the Ollama binary was installed, the daemon service wasn't running, which is required for all model operations."}'''
        
        # Parse the JSON string
        parsed_output = parser(action_json)
        
        # Verify the structure is correct
        self.assertIn("thought", parsed_output)
        self.assertIn("_is_answer_final", parsed_output)
        self.assertIn("_answer", parsed_output)
        self.assertEqual(parsed_output["_is_answer_final"], True)
        self.assertIn("bash", parsed_output["_answer"])  # Verify the bash code block is preserved

    # Tests for JsonOutputParserPydanticModel
    def test_pydantic_parser_basic_functionality(self):
        """Test basic parsing with Pydantic model"""
        parser = JsonOutputParserPydanticModel(pydantic_model=UserPydantic)
        json_string = '{"id": 1, "name": "Alice", "age": 30, "emails": ["alice@test.com"]}'
        
        result = parser(json_string)
        self.assertIsInstance(result, UserPydantic)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.age, 30)
        self.assertEqual(result.emails, ["alice@test.com"])

    def test_pydantic_parser_return_dict(self):
        """Test parsing with return_pydantic_object=False"""
        parser = JsonOutputParserPydanticModel(pydantic_model=UserPydantic, return_pydantic_object=False)
        json_string = '{"id": 2, "name": "Bob", "emails": []}'
        
        result = parser(json_string)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], 2)
        self.assertEqual(result["name"], "Bob")
        self.assertEqual(result["emails"], [])

    def test_pydantic_parser_with_examples(self):
        """Test parser with example instances"""
        example = UserPydantic(id=999, name="Example", age=25, emails=["example@test.com"])
        parser = JsonOutputParserPydanticModel(pydantic_model=UserPydantic, examples=[example])
        
        # Test format_instructions includes examples
        instructions = parser.format_instructions()
        self.assertIn("example@test.com", instructions)
        self.assertIn("Example", instructions)

    def test_pydantic_parser_validation_error(self):
        """Test that Pydantic validation errors are properly handled"""
        parser = JsonOutputParserPydanticModel(pydantic_model=ThoughtAction)
        invalid_json = '{"thought": "test", "action": "test", "confidence": 2.0}'  # confidence > 1.0
        
        with self.assertRaises(ValidationError):
            parser(invalid_json)

    def test_pydantic_parser_complex_nested_model(self):
        """Test parsing with nested Pydantic models"""
        parser = JsonOutputParserPydanticModel(pydantic_model=ComplexOutput)
        complex_json = '''{
            "thought": "Processing complex data",
            "name": "complex_function",
            "kwargs": {"param1": "value1", "param2": 42},
            "metadata": {"id": 1, "name": "John", "age": 30, "emails": ["john@test.com"]}
        }'''
        
        result = parser(complex_json)
        self.assertIsInstance(result, ComplexOutput)
        self.assertEqual(result.thought, "Processing complex data")
        self.assertEqual(result.name, "complex_function")
        self.assertEqual(result.kwargs["param1"], "value1")
        self.assertIsInstance(result.metadata, UserPydantic)
        self.assertEqual(result.metadata.name, "John")

    def test_pydantic_parser_schema_generation(self):
        """Test that schema generation works with native Pydantic functionality"""
        parser = JsonOutputParserPydanticModel(pydantic_model=ThoughtAction)
        schema = parser._get_pydantic_schema()
        
        # Should be valid JSON string
        import json
        schema_dict = json.loads(schema)
        
        # Verify schema structure
        self.assertIn("properties", schema_dict)
        self.assertIn("thought", schema_dict["properties"])
        self.assertIn("action", schema_dict["properties"])
        self.assertIn("confidence", schema_dict["properties"])
        
        # Check that descriptions are preserved
        self.assertEqual(schema_dict["properties"]["thought"]["description"], "Reasoning behind the action")

    def test_pydantic_parser_format_instructions(self):
        """Test format_instructions method uses Pydantic schema"""
        parser = JsonOutputParserPydanticModel(pydantic_model=UserPydantic)
        instructions = parser.format_instructions()
        
        # Should contain schema information
        self.assertIn("schema", instructions.lower())
        self.assertIn("json", instructions.lower())
        
        # Should contain field descriptions from Pydantic model
        self.assertIn("User ID", instructions)
        self.assertIn("User name", instructions)

    def test_pydantic_parser_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown blocks"""
        parser = JsonOutputParserPydanticModel(pydantic_model=UserPydantic)
        json_with_markdown = '''```json
{"id": 3, "name": "Charlie", "age": 25}
```'''
        
        result = parser(json_with_markdown)
        self.assertIsInstance(result, UserPydantic)
        self.assertEqual(result.id, 3)
        self.assertEqual(result.name, "Charlie")
        self.assertEqual(result.age, 25)

    def test_pydantic_parser_invalid_model_type(self):
        """Test that non-BaseModel classes raise TypeError"""
        with self.assertRaises(TypeError):
            JsonOutputParserPydanticModel(pydantic_model=dict)  # dict is not a BaseModel
        
        with self.assertRaises(TypeError):
            JsonOutputParserPydanticModel(pydantic_model=User)  # DataClass is not BaseModel

    def test_pydantic_parser_invalid_example_type(self):
        """Test that invalid example types raise TypeError"""
        with self.assertRaises(TypeError):
            JsonOutputParserPydanticModel(
                pydantic_model=UserPydantic, 
                examples=[{"id": 1, "name": "test"}]  # dict instead of UserPydantic instance
            )


if __name__ == "__main__":
    unittest.main()
