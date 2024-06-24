import unittest
from dataclasses import dataclass
from typing import List, Dict
from collections import OrderedDict

from lightrag.core.functional import dataclass_obj_to_dict, dataclass_obj_from_dict
from lightrag.core.base_data_class import DataClass


# Define test dataclasses
@dataclass
class SimpleData:
    name: str
    age: int
    score: float


@dataclass
class NestedData:
    simple: SimpleData
    description: str


@dataclass
class ListData:
    items: List[SimpleData]
    total: int


@dataclass
class DictData(DataClass):
    mappings: Dict[str, SimpleData]
    count: int


@dataclass
class OrderedDictData:
    ordered_mappings: OrderedDict[str, SimpleData]
    count: int


@dataclass
class ComplexData(DataClass):
    nested: NestedData
    list_data: ListData
    dict_data: DictData
    ordered_dict_data: OrderedDictData


# Define the test class
class TestDataclassFuncConversion(unittest.TestCase):

    def test_simple_data(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        simple_dict = dataclass_obj_to_dict(simple)
        expected_dict = {"name": "John", "age": 30, "score": 95.5}
        self.assertEqual(simple_dict, expected_dict)

        reconstructed_simple = dataclass_obj_from_dict(SimpleData, simple_dict)
        self.assertEqual(reconstructed_simple, simple)

    def test_nested_data(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        nested = NestedData(simple=simple, description="Test description")
        nested_dict = dataclass_obj_to_dict(nested)
        expected_dict = {
            "simple": {"name": "John", "age": 30, "score": 95.5},
            "description": "Test description",
        }
        self.assertEqual(nested_dict, expected_dict)

        reconstructed_nested = dataclass_obj_from_dict(NestedData, nested_dict)
        self.assertEqual(reconstructed_nested, nested)

    def test_list_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        list_data = ListData(items=[simple1, simple2], total=2)
        list_data_dict = dataclass_obj_to_dict(list_data)
        expected_dict = {
            "items": [
                {"name": "John", "age": 30, "score": 95.5},
                {"name": "Jane", "age": 25, "score": 88.0},
            ],
            "total": 2,
        }
        self.assertEqual(list_data_dict, expected_dict)

        reconstructed_list_data = dataclass_obj_from_dict(ListData, list_data_dict)
        self.assertEqual(reconstructed_list_data, list_data)

    def test_dict_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        dict_data = DictData(mappings={"first": simple1, "second": simple2}, count=2)
        dict_data_dict = dataclass_obj_to_dict(dict_data)
        expected_dict = {
            "mappings": {
                "first": {"name": "John", "age": 30, "score": 95.5},
                "second": {"name": "Jane", "age": 25, "score": 88.0},
            },
            "count": 2,
        }
        self.assertEqual(dict_data_dict, expected_dict)

        reconstructed_dict_data = dataclass_obj_from_dict(DictData, dict_data_dict)
        self.assertEqual(reconstructed_dict_data, dict_data)

    def test_ordered_dict_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        ordered_dict_data = OrderedDictData(
            ordered_mappings=OrderedDict([("first", simple1), ("second", simple2)]),
            count=2,
        )
        ordered_dict_data_dict = dataclass_obj_to_dict(ordered_dict_data)
        expected_dict = {
            "ordered_mappings": OrderedDict(
                [
                    ("first", {"name": "John", "age": 30, "score": 95.5}),
                    ("second", {"name": "Jane", "age": 25, "score": 88.0}),
                ]
            ),
            "count": 2,
        }
        self.assertEqual(ordered_dict_data_dict, expected_dict)

        reconstructed_ordered_dict_data = dataclass_obj_from_dict(
            OrderedDictData, ordered_dict_data_dict
        )
        self.assertEqual(reconstructed_ordered_dict_data, ordered_dict_data)

    def test_complex_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        nested = NestedData(simple=simple1, description="Test description")
        list_data = ListData(items=[simple1, simple2], total=2)
        dict_data = DictData(mappings={"first": simple1, "second": simple2}, count=2)
        ordered_dict_data = OrderedDictData(
            ordered_mappings=OrderedDict([("first", simple1), ("second", simple2)]),
            count=2,
        )
        complex_data = ComplexData(
            nested=nested,
            list_data=list_data,
            dict_data=dict_data,
            ordered_dict_data=ordered_dict_data,
        )
        complex_data_dict = dataclass_obj_to_dict(complex_data)
        expected_dict = {
            "nested": {
                "simple": {"name": "John", "age": 30, "score": 95.5},
                "description": "Test description",
            },
            "list_data": {
                "items": [
                    {"name": "John", "age": 30, "score": 95.5},
                    {"name": "Jane", "age": 25, "score": 88.0},
                ],
                "total": 2,
            },
            "dict_data": {
                "mappings": {
                    "first": {"name": "John", "age": 30, "score": 95.5},
                    "second": {"name": "Jane", "age": 25, "score": 88.0},
                },
                "count": 2,
            },
            "ordered_dict_data": {
                "ordered_mappings": OrderedDict(
                    [
                        ("first", {"name": "John", "age": 30, "score": 95.5}),
                        ("second", {"name": "Jane", "age": 25, "score": 88.0}),
                    ]
                ),
                "count": 2,
            },
        }
        self.assertEqual(complex_data_dict, expected_dict)

        reconstructed_complex_data = dataclass_obj_from_dict(
            ComplexData, complex_data_dict
        )
        self.assertEqual(reconstructed_complex_data, complex_data)

    def test_exclude(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        simple_dict = dataclass_obj_to_dict(simple, exclude={"SimpleData": ["age"]})
        expected_dict = {"name": "John", "score": 95.5}
        self.assertEqual(simple_dict, expected_dict)


class TestDataClassBaseClassConversion(unittest.TestCase):

    def test_dict_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        dict_data = DictData(mappings={"first": simple1, "second": simple2}, count=2)
        dict_data_dict = dict_data.to_dict()
        expected_dict = {
            "mappings": {
                "first": {"name": "John", "age": 30, "score": 95.5},
                "second": {"name": "Jane", "age": 25, "score": 88.0},
            },
            "count": 2,
        }
        self.assertEqual(dict_data_dict, expected_dict)

        reconstructed_dict_data = DictData.from_dict(dict_data_dict)
        self.assertEqual(reconstructed_dict_data, dict_data)

    def test_complex_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        nested = NestedData(simple=simple1, description="Test description")
        list_data = ListData(items=[simple1, simple2], total=2)
        dict_data = DictData(mappings={"first": simple1, "second": simple2}, count=2)
        ordered_dict_data = OrderedDictData(
            ordered_mappings=OrderedDict([("first", simple1), ("second", simple2)]),
            count=2,
        )
        complex_data = ComplexData(
            nested=nested,
            list_data=list_data,
            dict_data=dict_data,
            ordered_dict_data=ordered_dict_data,
        )
        complex_data_dict = complex_data.to_dict()
        expected_dict = {
            "nested": {
                "simple": {"name": "John", "age": 30, "score": 95.5},
                "description": "Test description",
            },
            "list_data": {
                "items": [
                    {"name": "John", "age": 30, "score": 95.5},
                    {"name": "Jane", "age": 25, "score": 88.0},
                ],
                "total": 2,
            },
            "dict_data": {
                "mappings": {
                    "first": {"name": "John", "age": 30, "score": 95.5},
                    "second": {"name": "Jane", "age": 25, "score": 88.0},
                },
                "count": 2,
            },
            "ordered_dict_data": {
                "ordered_mappings": OrderedDict(
                    [
                        ("first", {"name": "John", "age": 30, "score": 95.5}),
                        ("second", {"name": "Jane", "age": 25, "score": 88.0}),
                    ]
                ),
                "count": 2,
            },
        }
        self.assertEqual(complex_data_dict, expected_dict)

        reconstructed_complex_data = ComplexData.from_dict(complex_data_dict)
        self.assertEqual(reconstructed_complex_data, complex_data)

    def test_exclude(self):
        simple = DictData(
            mappings={"first": SimpleData(name="John", age=30, score=95.5)},
            count=1,
        )
        simple_dict = simple.to_dict(exclude={"DictData": ["count"]})
        expected_dict = {
            "mappings": {"first": {"name": "John", "age": 30, "score": 95.5}}
        }
        self.assertEqual(simple_dict, expected_dict)

        complex = ComplexData(
            nested=NestedData(
                simple=SimpleData(name="John", age=30, score=95.5),
                description="Test description",
            ),
            list_data=ListData(
                items=[
                    SimpleData(name="John", age=30, score=95.5),
                    SimpleData(name="Jane", age=25, score=88.0),
                ],
                total=2,
            ),
            dict_data=simple,
            ordered_dict_data=OrderedDictData(
                ordered_mappings=OrderedDict(
                    [
                        ("first", SimpleData(name="John", age=30, score=95.5)),
                        ("second", SimpleData(name="Jane", age=25, score=88.0)),
                    ]
                ),
                count=2,
            ),
        )
        complex_dict = complex.to_dict(
            exclude={"ListData": ["items"], "DictData": ["count"]}
        )
        expected_dict = {
            "nested": {
                "simple": {"name": "John", "age": 30, "score": 95.5},
                "description": "Test description",
            },
            "list_data": {"total": 2},
            "dict_data": {
                "mappings": {"first": {"name": "John", "age": 30, "score": 95.5}}
            },
            "ordered_dict_data": {
                "ordered_mappings": {
                    "first": {"name": "John", "age": 30, "score": 95.5},
                    "second": {"name": "Jane", "age": 25, "score": 88.0},
                },
                "count": 2,
            },
        }
        self.assertEqual(complex_dict, expected_dict)


if __name__ == "__main__":
    unittest.main()
