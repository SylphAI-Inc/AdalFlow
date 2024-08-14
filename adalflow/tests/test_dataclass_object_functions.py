import unittest
from dataclasses import dataclass
from typing import List, Dict
from collections import OrderedDict

from adalflow.core.functional import custom_asdict, dataclass_obj_from_dict
from adalflow.core.base_data_class import DataClass


# Define test dataclasses
@dataclass
class SimpleData(DataClass):
    name: str
    age: int
    score: float


@dataclass
class NestedData(DataClass):
    simple: SimpleData
    description: str


@dataclass
class ListData(DataClass):
    items: List[SimpleData]
    total: int


@dataclass
class DictData(DataClass):
    mappings: Dict[str, SimpleData]
    count: int


@dataclass
class OrderedDictData(DataClass):
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
        simple_dict = custom_asdict(simple)
        expected_dict = {"name": "John", "age": 30, "score": 95.5}
        self.assertEqual(simple_dict, expected_dict)

        reconstructed_simple = dataclass_obj_from_dict(SimpleData, simple_dict)
        self.assertEqual(reconstructed_simple, simple)

    def test_nested_data(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        nested = NestedData(simple=simple, description="Test description")
        nested_dict = custom_asdict(nested)
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
        list_data_dict = custom_asdict(list_data)
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
        dict_data_dict = custom_asdict(dict_data)
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
        ordered_dict_data_dict = custom_asdict(ordered_dict_data)
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
        complex_data_dict = custom_asdict(complex_data)
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
        simple_dict = custom_asdict(simple, exclude={"SimpleData": ["age"]})
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


@dataclass
class ComplexData2(DataClass):
    field1: str
    field2: int
    field3: float
    nested: NestedData
    list_data: ListData
    dict_data: DictData
    ordered_dict_data: OrderedDictData


# Define the test class
class TestDataClassYamlJsonConversion(unittest.TestCase):

    def test_simple_data(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        simple_dict = simple.to_dict()
        expected_dict = {"name": "John", "age": 30, "score": 95.5}
        self.assertEqual(simple_dict, expected_dict)

        simple_json = simple.to_json()
        expected_json = """{
    "name": "John",
    "age": 30,
    "score": 95.5
}"""

        self.assertEqual(simple_json, expected_json)

        simple_yaml = simple.to_yaml()
        expected_yaml = "name: John\nage: 30\nscore: 95.5\n"
        self.assertEqual(simple_yaml, expected_yaml)

        reconstructed_simple = SimpleData.from_dict(simple_dict)
        self.assertEqual(reconstructed_simple, simple)

        reconstructed_simple_json = SimpleData.from_json(simple_json)
        self.assertEqual(reconstructed_simple_json, simple)

        reconstructed_simple_yaml = SimpleData.from_yaml(expected_yaml)
        self.assertEqual(reconstructed_simple_yaml, simple)

    def test_nested_data(self):
        simple = SimpleData(name="John", age=30, score=95.5)
        nested = NestedData(simple=simple, description="Test description")
        nested_dict = nested.to_dict()
        expected_dict = {
            "simple": {"name": "John", "age": 30, "score": 95.5},
            "description": "Test description",
        }
        self.assertEqual(nested_dict, expected_dict)

        nested_json = nested.to_json()
        expected_json = """{
    "simple": {
        "name": "John",
        "age": 30,
        "score": 95.5
    },
    "description": "Test description"
}"""
        self.assertEqual(nested_json, expected_json)

        nested_yaml = nested.to_yaml()
        expected_yaml = """simple:
  name: John
  age: 30
  score: 95.5
description: Test description
"""

        self.assertEqual(nested_yaml, expected_yaml)

        reconstructed_nested = NestedData.from_dict(nested_dict)
        self.assertEqual(reconstructed_nested, nested)

        reconstructed_nested_json = NestedData.from_json(nested_json)
        self.assertEqual(reconstructed_nested_json, nested)

        reconstructed_nested_yaml = NestedData.from_yaml(expected_yaml)
        self.assertEqual(reconstructed_nested_yaml, nested)

    def test_list_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        list_data = ListData(items=[simple1, simple2], total=2)
        list_data_dict = list_data.to_dict()
        expected_dict = {
            "items": [
                {"name": "John", "age": 30, "score": 95.5},
                {"name": "Jane", "age": 25, "score": 88.0},
            ],
            "total": 2,
        }
        self.assertEqual(list_data_dict, expected_dict)

        list_data_json = list_data.to_json()
        expected_json = """{
    "items": [
        {
            "name": "John",
            "age": 30,
            "score": 95.5
        },
        {
            "name": "Jane",
            "age": 25,
            "score": 88.0
        }
    ],
    "total": 2
}"""
        self.assertEqual(list_data_json, expected_json)

        list_data_yaml = list_data.to_yaml()
        expected_yaml = """items:
- name: John
  age: 30
  score: 95.5
- name: Jane
  age: 25
  score: 88.0
total: 2
"""
        self.assertEqual(list_data_yaml, expected_yaml)

        reconstructed_list_data = ListData.from_dict(list_data_dict)
        self.assertEqual(reconstructed_list_data, list_data)

        reconstructed_list_data_json = ListData.from_json(list_data_json)
        self.assertEqual(reconstructed_list_data_json, list_data)

        reconstructed_list_data_yaml = ListData.from_yaml(expected_yaml)
        self.assertEqual(reconstructed_list_data_yaml, list_data)

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

        dict_data_json = dict_data.to_json()
        expected_json = """{
    "mappings": {
        "first": {
            "name": "John",
            "age": 30,
            "score": 95.5
        },
        "second": {
            "name": "Jane",
            "age": 25,
            "score": 88.0
        }
    },
    "count": 2
}"""
        self.assertEqual(dict_data_json, expected_json)

        dict_data_yaml = dict_data.to_yaml()
        expected_yaml = """mappings:
  first:
    name: John
    age: 30
    score: 95.5
  second:
    name: Jane
    age: 25
    score: 88.0
count: 2
"""
        self.assertEqual(dict_data_yaml, expected_yaml)

        reconstructed_dict_data = DictData.from_dict(dict_data_dict)
        self.assertEqual(reconstructed_dict_data, dict_data)

        reconstructed_dict_data_json = DictData.from_json(dict_data_json)
        self.assertEqual(reconstructed_dict_data_json, dict_data)

        reconstructed_dict_data_yaml = DictData.from_yaml(dict_data_yaml)
        self.assertEqual(reconstructed_dict_data_yaml, dict_data)

    def test_ordered_dict_data(self):
        simple1 = SimpleData(name="John", age=30, score=95.5)
        simple2 = SimpleData(name="Jane", age=25, score=88.0)
        ordered_dict_data = OrderedDictData(
            ordered_mappings=OrderedDict([("first", simple1), ("second", simple2)]),
            count=2,
        )
        ordered_dict_data_dict = ordered_dict_data.to_dict()
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

        ordered_dict_data_json = ordered_dict_data.to_json()
        expected_json = """{
    "ordered_mappings": {
        "first": {
            "name": "John",
            "age": 30,
            "score": 95.5
        },
        "second": {
            "name": "Jane",
            "age": 25,
            "score": 88.0
        }
    },
    "count": 2
}"""
        self.assertEqual(ordered_dict_data_json, expected_json)

        ordered_dict_data_yaml = ordered_dict_data.to_yaml()
        expected_yaml = """ordered_mappings:
  first:
    name: John
    age: 30
    score: 95.5
  second:
    name: Jane
    age: 25
    score: 88.0
count: 2
"""
        print(ordered_dict_data_yaml)
        self.assertEqual(ordered_dict_data_yaml, expected_yaml)

        reconstructed_ordered_dict_data = OrderedDictData.from_dict(
            ordered_dict_data_dict
        )
        self.assertEqual(reconstructed_ordered_dict_data, ordered_dict_data)

        reconstructed_ordered_dict_data_json = OrderedDictData.from_json(
            ordered_dict_data_json
        )
        self.assertEqual(reconstructed_ordered_dict_data_json, ordered_dict_data)

        reconstructed_ordered_dict_data_yaml = OrderedDictData.from_yaml(
            ordered_dict_data_yaml
        )
        self.assertEqual(reconstructed_ordered_dict_data_yaml, ordered_dict_data)

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
        complex_data = ComplexData2(
            field1="field1_value",
            field2=123,
            field3=456.78,
            nested=nested,
            list_data=list_data,
            dict_data=dict_data,
            ordered_dict_data=ordered_dict_data,
        )
        complex_data_dict = complex_data.to_dict()
        expected_dict = {
            "field1": "field1_value",
            "field2": 123,
            "field3": 456.78,
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

        complex_data_json = complex_data.to_json()
        expected_json = """{
    "field1": "field1_value",
    "field2": 123,
    "field3": 456.78,
    "nested": {
        "simple": {
            "name": "John",
            "age": 30,
            "score": 95.5
        },
        "description": "Test description"
    },
    "list_data": {
        "items": [
            {
                "name": "John",
                "age": 30,
                "score": 95.5
            },
            {
                "name": "Jane",
                "age": 25,
                "score": 88.0
            }
        ],
        "total": 2
    },
    "dict_data": {
        "mappings": {
            "first": {
                "name": "John",
                "age": 30,
                "score": 95.5
            },
            "second": {
                "name": "Jane",
                "age": 25,
                "score": 88.0
            }
        },
        "count": 2
    },
    "ordered_dict_data": {
        "ordered_mappings": {
            "first": {
                "name": "John",
                "age": 30,
                "score": 95.5
            },
            "second": {
                "name": "Jane",
                "age": 25,
                "score": 88.0
            }
        },
        "count": 2
    }
}"""
        self.assertEqual(complex_data_json, expected_json)

        complex_data_yaml = complex_data.to_yaml()
        expected_yaml = """field1: field1_value
field2: 123
field3: 456.78
nested:
  simple:
    name: John
    age: 30
    score: 95.5
  description: Test description
list_data:
  items:
  - name: John
    age: 30
    score: 95.5
  - name: Jane
    age: 25
    score: 88.0
  total: 2
dict_data:
  mappings:
    first:
      name: John
      age: 30
      score: 95.5
    second:
      name: Jane
      age: 25
      score: 88.0
  count: 2
ordered_dict_data:
  ordered_mappings:
    first:
      name: John
      age: 30
      score: 95.5
    second:
      name: Jane
      age: 25
      score: 88.0
  count: 2
"""
        self.assertEqual(complex_data_yaml, expected_yaml)

        reconstructed_complex_data = ComplexData2.from_dict(complex_data_dict)
        self.assertEqual(reconstructed_complex_data, complex_data)

        reconstructed_complex_data_json = ComplexData2.from_json(complex_data_json)
        self.assertEqual(reconstructed_complex_data_json, complex_data)

        reconstructed_complex_data_yaml = ComplexData2.from_yaml(complex_data_yaml)
        self.assertEqual(reconstructed_complex_data_yaml, complex_data)


if __name__ == "__main__":
    unittest.main()
