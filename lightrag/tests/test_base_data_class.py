import unittest
from lightrag.core import DataClass, required_field
from dataclasses import field, MISSING, dataclass


from typing import List, Dict, Optional


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
    single_address: Optional[Address] = field(
        default=None, metadata={"desc": "The single address"}
    )
    dict_addresses: Dict[str, Address] = field(default_factory=dict)


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
        )
        self.output_instance = MyOutputs(age=25, name="John Doe")

    def test_to_dict_instance(self):
        """Test the to_dict method on an instance of the dataclass."""
        expected_result = {"age": 25, "name": "John Doe"}
        instance_dict = self.output_instance.to_dict()
        print(f"instance_dict: {instance_dict}")
        self.assertEqual(instance_dict, expected_result)

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
        }
        instance_dict = self.person_instance.to_dict()
        self.assertEqual(instance_dict, expected_result)

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
                    "type": "Optional[{'type': 'Address', 'properties': {'street': {'type': 'str'}, 'city': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'city', 'zipcode']}]",
                    "desc": "The single address",
                },
                "dict_addresses": {
                    "type": "Dict[str, {'type': 'Address', 'properties': {'street': {'type': 'str'}, 'city': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'city', 'zipcode']}]"
                },
            },
            "required": ["age"],
        }

        person_dict_class = Person.to_dict_class()
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
        exclude = {"Person": ["addresses"], "Address": ["city"]}
        expected_result = {
            "type": "Person",
            "properties": {
                "name": {"type": "Optional[str]", "desc": "The name of the person"},
                "age": {"type": "int", "desc": "The age of the person"},
                "single_address": {
                    "type": "Optional[{'type': 'Address', 'properties': {'street': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'zipcode']}]",
                    "desc": "The single address",
                },
                "dict_addresses": {
                    "type": "Dict[str, {'type': 'Address', 'properties': {'street': {'type': 'str'}, 'zipcode': {'type': 'str'}}, 'required': ['street', 'zipcode']}]"
                },
            },
            "required": ["age"],
        }
        output = Person.to_dict_class(exclude=exclude)
        self.assertEqual(output, expected_result)

    def test_error_non_dataclass(self):
        """Test error handling when to_dict is called on a non-dataclass."""
        with self.assertRaises(AttributeError):
            non_dataclass = "Not a dataclass"
            non_dataclass.to_dict()
