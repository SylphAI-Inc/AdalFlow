import unittest
from dataclasses import field, MISSING, dataclass
from typing import List, Dict, Optional, Set
import enum
from dataclasses import asdict

# Assume these imports come from the adalflow package
from adalflow.core.base_data_class import (
    DataClass,
    required_field,
)
from adalflow.core.functional import get_type_schema


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
        self.assertEqual(output, expected_result)

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

    def test_enum_as_data_class(self):
        @dataclass
        class LabelDataClass(DataClass, str, enum.Enum):
            """Enumeration for single-label text classification."""

            SPAM = "spam"
            NOT_SPAM = "not_spam"

        schema = LabelDataClass.to_schema()
        print(f"schema: {schema}")
        type_schema = get_type_schema(LabelDataClass)
        print(f"type_schema: {type_schema}")
        self.assertEqual(
            schema,
            {
                "type": "Enum[LabelDataClass(SPAM=spam, NOT_SPAM=not_spam)]",
                "properties": {},
                "required": [],
            },
        )
        self.assertEqual(
            type_schema,
            "Enum[LabelDataClass(SPAM=spam, NOT_SPAM=not_spam)]",
        )


@dataclass
class ListDataclass(DataClass):
    answer: str = field(metadata={"desc": "The answer to the user question."})
    pmids: list[int] = field(
        metadata={"desc": "The PMIDs of the relevant articles used to answer."}
    )
    __input_fields__ = ["pmids"]
    __output_fields__ = ["answer"]


class TestUnnestedDataclass(unittest.TestCase):
    def test_list_dataclass(self):
        instance = ListDataclass(answer="answer", pmids=[1, 2, 3])
        result = instance.to_dict()
        print(f"result: {result}")
        expected = "{'pmids': [1, 2, 3], 'answer': 'answer'}"

        self.assertEqual(str(result), expected)
        restored_instance = ListDataclass.from_dict(result)
        self.assertEqual(restored_instance, instance)

        # rearrange the order using include
        result_reordered = instance.to_dict(include=["pmids", "answer"])
        print(str(result_reordered))

    def test_dict_dataclass(self):
        @dataclass
        class DictDataclass(DataClass):
            answer: str = field(metadata={"desc": "The answer to the user question."})
            pmids: Dict[str, int] = field(
                metadata={"desc": "The PMIDs of the relevant articles used to answer."}
            )

        instance = DictDataclass(answer="answer", pmids={"a": 1, "b": 2, "c": 3})
        result = instance.to_dict()
        print(f"result: {result}")
        # expected = "{'answer': 'answer', 'pmids': {'a': 1, 'b': 2, 'c': 3}}"
        # self.assertEqual(str(result), expected)
        restored_instance = DictDataclass.from_dict(result)
        self.assertEqual(restored_instance, instance)

    def test_set_dataclass(self):
        @dataclass
        class SetDataclass(DataClass):
            answer: str = field(metadata={"desc": "The answer to the user question."})
            pmids: Set[int] = field(
                metadata={"desc": "The PMIDs of the relevant articles used to answer."}
            )

        instance = SetDataclass(answer="answer", pmids={1, 2, 3})
        result = instance.to_dict()
        print(f"result: {result}")
        expected = "{'answer': 'answer', 'pmids': {1, 2, 3}}"
        self.assertEqual(str(result), expected)
        restored_instance = SetDataclass.from_dict(result)
        self.assertEqual(restored_instance, instance)


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
        p_manual = ModelClass(
            **{**instance.to_dict(), "extra_field": "should_be_ignored"}
        )
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
            optional_value: Optional[int] = field(
                metadata={"desc": "An optional integer"}, default=None
            )

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
            set_hobbies={1, 2},
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
            addresses=[
                Address(street="100 Main St", city="Cityville", zipcode="00000")
            ],
            single_address=Address(
                street="100 Main St", city="Cityville", zipcode="00000"
            ),
            dict_addresses={
                "home": Address(street="100 Main St", city="Cityville", zipcode="00000")
            },
            set_hobbies={9, 10},
        )
        p_person = Person.to_pydantic(original_person)
        base_person = Person.pydantic_to_dataclass(p_person)
        self.assertEqual(base_person, original_person)


if __name__ == "__main__":
    unittest.main()
