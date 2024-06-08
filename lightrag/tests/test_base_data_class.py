import unittest
from lightrag.core import DataClass
from dataclasses import field, MISSING, dataclass


# Assuming DataClass is in your_module and correctly imported
@dataclass
class MyOutputs(DataClass):
    age: int = field(
        default=MISSING, metadata={"desc": "The age of the person", "prefix": "Age:"}
    )
    name: str = field(
        default=MISSING, metadata={"desc": "The name of the person", "prefix": "Name:"}
    )


class TestBaseDataClass(unittest.TestCase):

    def test_to_dict_instance(self):
        """Test the to_dict method on an instance of the dataclass."""
        instance = MyOutputs(age=25, name="John Doe")
        expected_result = {"age": 25, "name": "John Doe"}
        self.assertEqual(instance.to_dict(), expected_result)

    def test_to_dict_instance_with_exclusion(self):
        """Test the to_dict method with field exclusion on an instance."""
        instance = MyOutputs(age=25, name="John Doe")
        expected_result = {"name": "John Doe"}
        self.assertEqual(instance.to_dict(exclude=["age"]), expected_result)

    def test_to_dict_class(self):
        """Test the to_dict method on the class itself."""
        expected_result = {
            "age": {"type": "int", "desc": "The age of the person", "required": True},
            "name": {"type": "str", "desc": "The name of the person", "required": True},
        }
        self.assertEqual(MyOutputs.to_dict_class(), expected_result)

    def test_to_dict_class_with_exclusion(self):
        """Test the to_dict method with field exclusion on the class."""
        exclude = ["age"]
        expected_result = {
            "name": {"type": "str", "desc": "The name of the person", "required": True}
        }
        self.assertEqual(MyOutputs.to_dict_class(exclude=exclude), expected_result)

    def test_error_non_dataclass(self):
        """Test error handling when to_dict is called on a non-dataclass."""
        with self.assertRaises(AttributeError):
            non_dataclass = "Not a dataclass"
            non_dataclass.to_dict()
