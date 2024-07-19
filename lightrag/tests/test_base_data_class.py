import unittest
from lightrag.core import DataClass, required_field
from dataclasses import field, MISSING, dataclass, asdict

from lightrag.core.functional import get_type_schema


from typing import List, Dict, Optional, Set


# simple class, both fields are required
@dataclass
class MyOutputs(DataClass):
    age: int = field(
        default=MISSING, metadata={"desc": "The age of the person", "prefix": "Age:"}
    )
    name: str = field(
        metadata={
            "desc": "The name of the person",
            "prefix": "Name:",
        },  # will make it a required field
    )


@dataclass
class Address:
    street: str
    city: str
    zipcode: str


import enum


# @dataclass
class Label(str, enum.Enum):
    """Enumeration for single-label text classification."""

    SPAM = "spam"
    NOT_SPAM = "not_spam"


@dataclass
class ClassificationOutput(DataClass):
    """
    Class for a single class label prediction.
    """

    label: Label = field(metadata={"desc": "Label of the category."})


# Example instance of the nested dataclasses and complex data types as list, dict
@dataclass
class Person(DataClass):
    name: Optional[str] = field(
        metadata={"desc": "The name of the person"}, default=None
    )
    age: int = field(
        metadata={"desc": "The age of the person"},
        default_factory=required_field(),  # customized behavior to allow required fields after optional fields
    )
    addresses: List[Address] = field(
        default_factory=list, metadata={"desc": "The list of addresses"}
    )
    single_address: Address = field(
        default=None, metadata={"desc": "The single address"}
    )
    dict_addresses: Dict[str, Address] = field(default_factory=dict)
    set_hobbies: Set[int] = field(
        metadata={"desc": "The set of hobbies"}, default_factory=required_field()
    )


class TestBaseDataClass(unittest.TestCase):
    # setup
    def setUp(self):
        self.person_instance = Person(
            name="John Doe",
            age=30,
            addresses=[
                Address(street="123 Main St", city="Anytown", zipcode="12345"),
                Address(street="456 Elm St", city="Othertown", zipcode="67890"),
            ],
            single_address=Address(
                street="123 Main St", city="Anytown", zipcode="12345"
            ),
            dict_addresses={
                "home": Address(street="123 Main St", city="Anytown", zipcode="12345"),
                "work": Address(street="456 Elm St", city="Othertown", zipcode="67890"),
            },
            set_hobbies={1, 2, 3},
        )
        self.output_instance = MyOutputs(age=25, name="John Doe")

    def test_to_dict_instance(self):
        """Test the to_dict method on an instance of the dataclass."""
        expected_result = {"age": 25, "name": "John Doe"}
        instance_dict = self.output_instance.to_dict()
        print(f"instance_dict: {instance_dict}")
        print(f"instance dict from asdict: {asdict(self.output_instance)}")
        self.assertEqual(instance_dict, expected_result)

        # test from_dict
        reconstructed_instance = MyOutputs.from_dict(instance_dict)
        self.assertEqual(reconstructed_instance, self.output_instance)

        # test with nested dataclass
        expected_result = {
            "name": "John Doe",
            "age": 30,
            "addresses": [
                {"street": "123 Main St", "city": "Anytown", "zipcode": "12345"},
                {"street": "456 Elm St", "city": "Othertown", "zipcode": "67890"},
            ],
            "single_address": {
                "street": "123 Main St",
                "city": "Anytown",
                "zipcode": "12345",
            },
            "dict_addresses": {
                "home": {
                    "street": "123 Main St",
                    "city": "Anytown",
                    "zipcode": "12345",
                },
                "work": {
                    "street": "456 Elm St",
                    "city": "Othertown",
                    "zipcode": "67890",
                },
            },
            "set_hobbies": {1, 2, 3},
        }
        instance_dict = self.person_instance.to_dict()
        print(f"instance_dict: {instance_dict}")
        print(f"instance dict from asdict: {asdict(self.person_instance)}")
        self.assertEqual(instance_dict, expected_result)
        self.assertEqual(asdict(self.person_instance), expected_result)

        # test from_dict
        reconstructed_instance = Person.from_dict(instance_dict)
        print(f"original_instance: {self.person_instance}")
        print(f"reconstructed_instance: {reconstructed_instance}")
        self.assertEqual(reconstructed_instance, self.person_instance)

    def test_to_dict_class_nested(self):
        """Test the to_dict method on an instance of the dataclass with nested"""
        expected_result = {
            "type": "Person",
            "properties": {
                "name": {"type": "Optional[str]", "desc": "The name of the person"},
                "age": {"type": "int", "desc": "The age of the person"},
                "addresses": {
                    "type": "List[{'type': 'Address', 'properties': {'street': {'type': 'str'}, 'city': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'city', 'zipcode']}]",
                    "desc": "The list of addresses",
                },
                "single_address": {
                    "type": "{'type': 'Address', 'properties': {'street': {'type': 'str'}, 'city': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'city', 'zipcode']}",
                    "desc": "The single address",
                },
                "dict_addresses": {
                    "type": "Dict[str, {'type': 'Address', 'properties': {'street': {'type': 'str'}, 'city': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'city', 'zipcode']}]"
                },
                "set_hobbies": {
                    "type": "Set[int]",
                    "desc": "The set of hobbies",
                },
            },
            "required": ["age", "set_hobbies"],
        }

        person_dict_class = Person.to_dict_class()
        print(f"person_dict_class: {person_dict_class}")
        self.assertEqual(person_dict_class, expected_result)

    def test_to_dict_instance_with_exclusion(self):
        """Test the to_dict method with field exclusion on an instance."""
        output = self.output_instance.to_dict(exclude=["age"])
        print(f"output: {output}")
        expected_result = {"name": "John Doe"}
        self.assertEqual(output, expected_result)

    def test_to_dict_class(self):
        """Test the to_dict method on the class itself."""
        expected_result = {
            "type": "MyOutputs",
            "properties": {
                "age": {
                    "type": "int",
                    "desc": "The age of the person",
                    "prefix": "Age:",
                },
                "name": {
                    "type": "str",
                    "desc": "The name of the person",
                    "prefix": "Name:",
                },
            },
            "required": ["age", "name"],
        }
        output = MyOutputs.to_dict_class()
        self.assertEqual(output, expected_result)

    def test_to_dict_class_with_exclusion(self):
        """Test the to_dict method with field exclusion on the class."""
        exclude = ["age"]
        expected_result = {
            "type": "MyOutputs",
            "properties": {
                "name": {
                    "type": "str",
                    "desc": "The name of the person",
                    "prefix": "Name:",
                },
            },
            "required": ["name"],
        }
        output = MyOutputs.to_dict_class(exclude=exclude)
        self.assertEqual(output, expected_result)

        # on Person class
        exclude = {"Person": ["addresses", "set_hobbies"], "Address": ["city"]}
        expected_result = {
            "type": "Person",
            "properties": {
                "name": {"type": "Optional[str]", "desc": "The name of the person"},
                "age": {"type": "int", "desc": "The age of the person"},
                "single_address": {
                    "type": "{'type': 'Address', 'properties': {'street': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'zipcode']}",
                    "desc": "The single address",
                },
                "dict_addresses": {
                    "type": "Dict[str, {'type': 'Address', 'properties': {'street': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'zipcode']}]"
                },
            },
            "required": ["age"],
        }
        output = Person.to_dict_class(exclude=exclude)
        print(f"output 1: {output}")
        # self.assertEqual(output, expected_result)

    def test_error_non_dataclass(self):
        """Test error handling when to_dict is called on a non-dataclass."""
        with self.assertRaises(AttributeError):
            non_dataclass = "Not a dataclass"
            non_dataclass.to_dict()


class TestGetTypeSchema(unittest.TestCase):

    def test_enum_schema(self):
        result = get_type_schema(Label)
        expected = "Enum[Label(SPAM=spam, NOT_SPAM=not_spam)]"
        self.assertEqual(result, expected)

    def test_enum_field_in_dataclass(self):
        result = get_type_schema(ClassificationOutput)
        expected = "{'type': 'ClassificationOutput', 'properties': {'label': {'type': 'Enum[Label(SPAM=spam, NOT_SPAM=not_spam)]', 'desc': 'Label of the category.'}}, 'required': ['label']}"
        self.assertEqual(result, expected)

    def test_optional_enum_field(self):
        @dataclass
        class OptionalClassificationOutput:
            label: Optional[Label] = field(
                default=None, metadata={"desc": "Label of the category."}
            )

        result = get_type_schema(OptionalClassificationOutput)
        expected = "{'type': 'OptionalClassificationOutput', 'properties': {'label': {'type': 'Optional[Enum[Label(SPAM=spam, NOT_SPAM=not_spam)]]', 'desc': 'Label of the category.'}}, 'required': []}"
        self.assertEqual(result, expected)

    def test_enum_as_list_element(self):
        @dataclass
        class EnumListClassificationOutput:
            labels: List[Label] = field(
                default_factory=list, metadata={"desc": "List of labels."}
            )

        result = get_type_schema(EnumListClassificationOutput)
        print(f"result: {result}")
        expected = "{'type': 'EnumListClassificationOutput', 'properties': {'labels': {'type': 'List[Enum[Label(SPAM=spam, NOT_SPAM=not_spam)]]', 'desc': 'List of labels.'}}, 'required': []}"
        self.assertEqual(result, expected)

        # test instance with DataClass
        @dataclass
        class EnumListClassificationOutput(DataClass):
            labels: List[Label] = field(
                default_factory=list, metadata={"desc": "List of labels."}
            )

        instance = EnumListClassificationOutput(labels=[Label.SPAM, Label.NOT_SPAM])
        result = instance.to_dict()
        print(f"EnumListClassificationOutput instance: {result}")
        expected = "{'labels': [<Label.SPAM: 'spam'>, <Label.NOT_SPAM: 'not_spam'>]}"
        self.assertEqual(str(result), expected)
        restored_instance = EnumListClassificationOutput.from_dict(result)
        print(f"restored_instance: {restored_instance}")
        print(f"instance: {instance}")
        self.assertEqual(restored_instance, instance)

        from lightrag.core import Generator
        from lightrag.components.model_client.openai_client import OpenAIClient
        from lightrag.components.output_parsers import JsonOutputParser
        from lightrag.utils import setup_env

        setup_env()
        parser = JsonOutputParser(
            data_class=EnumListClassificationOutput, return_data_class=True
        )
        generator = Generator(
            model_client=OpenAIClient(),
            prompt_kwargs={
                "task_desc_str": "Classify the following text as spam or not spam.",
                "output_format_str": parser.format_instructions(),
            },
            model_kwargs={"model": "gpt-3.5-turbo"},
            output_processors=parser,
        )
        prompt_kwargs = {"input_str": "This is a spam message."}
        result = generator(prompt_kwargs)
        print(f"result: {result}")


if __name__ == "__main__":
    # class_format = ClassificationOutput.to_schema()
    # class_instance = ClassificationOutput(label=Label.NOT_SPAM)
    # print(class_format)
    # print(class_instance.to_dict())
    unittest.main()
