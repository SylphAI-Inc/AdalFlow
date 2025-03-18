import unittest
from dataclasses import field, MISSING, dataclass
from typing import List, Dict, Optional, Set
import enum

# Assume these imports come from the adalflow package
from adalflow.core.base_data_class import DataClass, required_field, check_adal_dataclass
from adalflow.core.functional import get_type_schema

import json
import yaml

# Simple dataclass for testing
@dataclass
class MyOutputs(DataClass):
    age: int = field(
        default=MISSING, metadata={"desc": "The age of the person", "prefix": "Age:"}
    )
    name: str = field(
        metadata={"desc": "The name of the person", "prefix": "Name:"},
    )

@dataclass
class Address:
    street: str
    city: str
    zipcode: str

class Label(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"

@dataclass
class ClassificationOutput(DataClass):
    label: Label = field(metadata={"desc": "Label of the category."})

@dataclass
class Person(DataClass):
    name: Optional[str] = field(
        metadata={"desc": "The name of the person"}, default=None
    )
    age: int = field(
        metadata={"desc": "The age of the person"},
        default_factory=required_field(),
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


# A simple dataclass for testing formatting and field management
@dataclass
class TestData(DataClass):
    field1: int = field(metadata={"desc": "Field one"}, default=1)
    field2: str = field(metadata={"desc": "Field two"}, default="default")
    field3: Optional[float] = field(metadata={"desc": "Optional field three"}, default=None)


class TestPydanticConversionExtended(unittest.TestCase):
    def test_missing_required_field(self):
        """
        Test that missing a required field in the dictionary (simulated by overriding to_dict)
        raises a validation error when converting to a Pydantic model.
        """
        # Create a valid instance.
        instance = MyOutputs(age=40, name="Alice")
        # Override to_dict to simulate missing 'age'
        original_to_dict = instance.to_dict
        instance.to_dict = lambda exclude=None, include=None: {"name": "Alice"}
        
        with self.assertRaises(Exception):
            MyOutputs.to_pydantic(instance)
        
        # Restore the original to_dict
        instance.to_dict = original_to_dict
    
    def test_invalid_type_conversion(self):
        """
        Test that providing an invalid type raises a validation error.
        """
        # Create an instance with an invalid type for 'age'
        instance = MyOutputs(age="not_an_int", name="Alice")
        with self.assertRaises(Exception):
            MyOutputs.to_pydantic(instance)
    
    def test_default_value_usage(self):
        """
        Test that fields with default values are correctly used when not provided.
        """
        @dataclass
        class WithDefault(DataClass):
            value: int = field(default=100, metadata={"desc": "A default value"})
        # Here, we pass an instance without modifying the default.
        instance = WithDefault()
        p_instance = WithDefault.to_pydantic(instance)
        self.assertEqual(p_instance.value, 100)
    
    def test_extra_fields_behavior(self):
        """
        Test how extra fields are handled. Extra fields in the input dict are ignored.
        """
        # Create a proper instance first.
        instance = MyOutputs(age=30, name="Bob")
        p_instance = MyOutputs.to_pydantic(instance)
        # Although we cannot directly pass extra fields via to_pydantic (since it builds from to_dict()),
        # we simulate the behavior by creating a Pydantic model instance manually.
        ModelClass = type(p_instance)
        p_manual = ModelClass(**{**instance.to_dict(), "extra_field": "should_be_ignored"})
        # Check that the extra field is not set.
        self.assertEqual(p_manual.age, 30)
        self.assertEqual(p_manual.name, "Bob")
        self.assertFalse(hasattr(p_manual, "extra_field"))
    
    def test_union_optional_handling(self):
        """
        Test a dataclass field with an Optional type to ensure that None is accepted.
        """
        @dataclass
        class WithOptional(DataClass):
            optional_value: Optional[int] = field(metadata={"desc": "An optional integer"}, default=None)
        # Create an instance without providing a value.
        instance = WithOptional()
        p_instance = WithOptional.to_pydantic(instance)
        self.assertIsNone(p_instance.optional_value)
        # Now provide a value.
        instance2 = WithOptional(optional_value=42)
        p_instance2 = WithOptional.to_pydantic(instance2)
        self.assertEqual(p_instance2.optional_value, 42)
    
    def test_nested_model_conversion_errors(self):
        """
        Test nested dataclass conversion where nested dict has an invalid type.
        """
        # Create a Person instance with an invalid type for addresses.
        instance = Person(
            name="Test",
            age=25,
            addresses=[123],  # invalid: should be a dict for Address
            single_address={"street": "X", "city": "Y", "zipcode": "Z"},
            dict_addresses={"home": {"street": "X", "city": "Y", "zipcode": "Z"}},
            set_hobbies={1, 2}
        )
        with self.assertRaises(Exception):
            Person.to_pydantic(instance)
    
    def test_pydantic_model_repr(self):
        """
        Test that the __repr__ of the Pydantic model includes the expected field values.
        """
        instance = MyOutputs(age=55, name="Charlie")
        p_instance = MyOutputs.to_pydantic(instance)
        repr_str = repr(p_instance)
        self.assertIn("age=55", repr_str)
        self.assertIn("name='Charlie'", repr_str)
    
    def test_round_trip_conversion(self):
        """
        Full round-trip test: convert DataClass instance -> Pydantic model instance -> back to DataClass.
        """
        # For MyOutputs:
        original = MyOutputs(age=40, name="Alice")
        p_instance = MyOutputs.to_pydantic(original)
        base_instance = MyOutputs.pydantic_to_dataclass(p_instance)
        self.assertEqual(base_instance, original)
        
        # For Person with nested data:
        original_person = Person(
            name="Dana",
            age=45,
            addresses=[Address(street="100 Main St", city="Cityville", zipcode="00000")],
            single_address=Address(street="100 Main St", city="Cityville", zipcode="00000"),
            dict_addresses={"home": Address(street="100 Main St", city="Cityville", zipcode="00000")},
            set_hobbies={9, 10}
        )
        p_person = Person.to_pydantic(original_person)
        base_person = Person.pydantic_to_dataclass(p_person)
        self.assertEqual(base_person, original_person)

if __name__ == "__main__":
    unittest.main()



