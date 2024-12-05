import unittest
from dataclasses import dataclass, field
from typing import List
from adalflow.core.base_data_class import DataClass
from adalflow.components.output_parsers.dataclass_parser import DataClassParser


# Define a basic DataClass for testing
@dataclass
class BasicOutput(DataClass):
    explanation: str = field(
        metadata={"desc": "A brief explanation of the concept in one sentence."}
    )
    example: str = field(metadata={"desc": "An example of the concept in a sentence."})
    __output_fields__ = ["explanation", "example"]


# Define a nested DataClass for testing
@dataclass
class NestedOutput(DataClass):
    title: str
    description: str
    items: List[str]
    __output_fields__ = ["title", "description", "items"]


class TestDataClassParser(unittest.TestCase):

    def setUp(self):
        self.basic_data_class = BasicOutput
        self.nested_data_class = NestedOutput
        self.basic_parser = DataClassParser(
            data_class=self.basic_data_class, return_data_class=True, format_type="json"
        )
        self.nested_parser = DataClassParser(
            data_class=self.nested_data_class,
            return_data_class=True,
            format_type="yaml",
        )

    def test_basic_data_class_json(self):
        input_instance = BasicOutput(
            explanation="This is a test.", example="Example sentence."
        )
        input_str = self.basic_parser.get_input_str(input_instance)
        self.assertIn("This is a test.", input_str)
        self.assertIn("Example sentence.", input_str)

        output_format_str = self.basic_parser.get_output_format_str()
        self.assertIn("explanation", output_format_str)
        self.assertIn("example", output_format_str)

        output = self.basic_parser.call(
            '{"explanation": "Test explanation", "example": "Test example."}'
        )
        self.assertIsInstance(output, BasicOutput)

    def test_basic_data_class_yaml(self):
        self.yaml_parser = DataClassParser(
            data_class=self.basic_data_class, return_data_class=True, format_type="yaml"
        )
        input_instance = BasicOutput(
            explanation="This is a test.", example="Example sentence."
        )
        input_str = self.yaml_parser.get_input_str(input_instance)
        self.assertIn("This is a test.", input_str)

        self.assertIn("Example sentence.", input_str)

        output_format_str = self.yaml_parser.get_output_format_str()
        self.assertIn("explanation", output_format_str)
        self.assertIn("example", output_format_str)

        output = self.yaml_parser.call(
            """explanation: Test explanation
example: Test example."""
        )
        print(f"output: {output}")
        self.assertIsInstance(output, BasicOutput)

    def test_nested_data_class_json(self):
        input_instance = NestedOutput(
            title="Title", description="Description", items=["Item 1", "Item 2"]
        )
        input_str = self.nested_parser.get_input_str(input_instance)
        self.assertIn("Title", input_str)
        self.assertIn("Description", input_str)
        self.assertIn("Item 1", input_str)
        self.assertIn("Item 2", input_str)

        output_format_str = self.nested_parser.get_output_format_str()
        self.assertIn("title", output_format_str)
        self.assertIn("description", output_format_str)
        self.assertIn("items", output_format_str)

        output = self.nested_parser.call(
            """title: Nested Title
description: Nested description
items:
  - Item 1
  - Item 2"""
        )
        self.assertIsInstance(output, NestedOutput)

    def test_nested_data_class_yaml(self):
        self.nested_parser._format_type = "yaml"
        input_instance = NestedOutput(
            title="Title", description="Description", items=["Item 1", "Item 2"]
        )
        input_str = self.nested_parser.get_input_str(input_instance)
        self.assertIn("Title", input_str)
        self.assertIn("Description", input_str)
        self.assertIn("Item 1", input_str)
        self.assertIn("Item 2", input_str)

        output_format_str = self.nested_parser.get_output_format_str()
        self.assertIn("title", output_format_str)
        self.assertIn("description", output_format_str)
        self.assertIn("items", output_format_str)

        output = self.nested_parser.call(
            """title: Nested Title
description: Nested description
items:
  - Item 1
  - Item 2"""
        )
        self.assertIsInstance(output, NestedOutput)

    def test_invalid_data_class(self):
        with self.assertRaises(ValueError):
            DataClassParser(data_class=dict)  # dict is not a dataclass

    def test_invalid_format_type(self):
        with self.assertRaises(ValueError):
            DataClassParser(
                data_class=self.basic_data_class, format_type="xml"
            )  # Invalid format type


if __name__ == "__main__":
    unittest.main()
