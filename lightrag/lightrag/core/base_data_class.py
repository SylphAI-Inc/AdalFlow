"""
The role of the base data class in LightRAG for LLM applications is like `Tensor` for `PyTorch`.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import collections

import enum
from copy import deepcopy
from dataclasses import (
    dataclass,
    field,
    fields,
    make_dataclass,
    is_dataclass,
)

import json
import yaml
import warnings
import logging

from lightrag.core.functional import (
    # dataclass_obj_to_dict,
    custom_asdict,
    dataclass_obj_from_dict,
    get_dataclass_schema,
    convert_schema_to_signature,
    represent_ordereddict,
)


logger = logging.getLogger(__name__)


class DataClassFormatType(enum.Enum):
    r"""The format type for the DataClass schema."""

    # for class
    SCHEMA = "schema"
    SIGNATURE_YAML = "signature_yaml"
    SIGNATURE_JSON = "signature_json"
    # for instance
    EXAMPLE_YAML = "example_yaml"
    EXAMPLE_JSON = "example_json"


# Register the custom representer
yaml.add_representer(collections.OrderedDict, represent_ordereddict)


def required_field() -> Callable[[], Any]:
    """
    A factory function to create a required field in a dataclass.
    The returned callable raises a TypeError when invoked, indicating a required field was not provided.

    Args:
        name (Optional[str], optional): The name of the required field. Defaults to None

    Returns:
        Callable[[], Any]: A callable that raises TypeError when called, indicating a missing required field.

    Example:

    .. code-block:: python

        from dataclasses import dataclass
        from lightrag.core.base_data_class import required_field, DataClass

        @dataclass
        class Person(DataClass):
            name: str = field(default=None)
            age: int = field(default_factory=required_field())# allow required field after optional field
    """

    def required_field_error():
        """This function is returned by required_field and raises an error indicating the field is required."""
        raise TypeError("This field is required and was not provided.")

    required_field_error.__name__ = (
        "required_field"  # Set the function's name explicitly
    )
    return required_field_error


# Dict is for the nested dataclasses, e.g. {"Person": ["name", "age"], "Address": ["city"]}
ExcludeType = Optional[Union[List[str], Dict[str, List[str]]]]


class DataClass:
    __doc__ = r"""The base data class for all data types that interact with LLMs.

    Please only exclude optional fields in the exclude dictionary.

    Designed to streamline the handling, serialization, and description of data within our applications, especially to LLM prompt.
    We explicitly handle this instead of relying on 3rd party libraries such as pydantic or marshmallow to have better
    transparency and to keep the order of the fields when get serialized.

    How to create your own dataclass?

    1. Subclass DataClass and define the fields with the `field` decorator.
    2. Use the `medata` argument and a `desc` key to describe the field.
    3. Keep the order of the fields as how you want them to be serialized and described to LLMs.
    4. field with default value is considered optional. Field without default value and field with default_factory=required_field is considered required.

    How to use it?

    Describing:

    We defined :ref:`DataClassFormatType <core-base_data_class_format_type>` to categorize DataClass description formats
    as input or output in LLM prompt.


    (1) For describing the class (data structure):

    `Signature` is more token effcient than schema, and schema as it is always a json string, when you want LLMs to output yaml, it can be misleading if you describe the data structure in json.

    - DataClassFormatType.SCHEMA: a more standard way to describe the data structure in Json string, :meth:`to_schema` as string and :meth:`to_schema` as dict.
    - DataClassFormatType.SIGNATURE_JSON: emitating a json object with field name as key and description as value, :meth:`to_json_signature` as string.
    - DataClassFormatType.SIGNATURE_YAML: emitating a yaml object with field name as key and description as value, :meth:`to_yaml_signature` as string.

    (2) For describing the class instance: this is helpful to do few-shot examples in LLM prompt.
    - DataClassFormatType.EXAMPLE_JSON: the json representation of the instance, :meth:`to_json` as string.
    - DataClassFormatType.EXAMPLE_YAML: the yaml representation of the instance, :meth:`to_yaml` as string.

    Overall, we have a unified class method :meth:`format_str` to generate formatted output based on the type of operation and class/instance context.

    note::
        You do not need to use our format, overwrite any method in the subclass to fit in your needs.

    Loading data:

    - :meth:`from_dict` is used to create a dataclass instance from a dictionary.


    Refer :ref:`DataClass<core-base_data_class_note>` for more detailed instructions.

    Examples:

    .. code-block:: python

        # Define a dataclass
        from lightrag.core import DataClass
        from dataclasses import dataclass, field

        @dataclass
        class MyOutputs(DataClass):
            age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
            name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})

        # Create json signature
        print(MyOutputs.to_json_signature())
        # Output:
        # {
        #     "age": "The age of the person",
        #     "name": "The name of the person"
        # }
        # Create yaml signature
        print(MyOutputs.to_yaml_signature())
        # Output:
        # age: The age of the person
        # name: The name of the person

        # Create a dataclass instance
        my_instance = MyOutputs(age=25, name="John Doe")
        # Create json example
        print(my_instance.to_json_example())
        # Output:
        # {
        #     "age": 25,
        #     "name": "John Doe"
        # }
        # Create yaml signature
        print(my_instance.to_yaml_example())
        # Output:
        # age: 25
        # name: John Doe

    """

    def __post_init__(self):

        for f in fields(self):
            if "desc" not in f.metadata and "description" not in f.metadata:
                warnings.warn(
                    f"Class {  self.__class__.__name__} Field {f.name} is missing 'desc' in metadata",
                    UserWarning,
                )

    def to_dict(self, exclude: ExcludeType = None) -> Dict[str, Any]:
        """Convert a dataclass object to a dictionary.

        Supports nested dataclasses, lists, and dictionaries.
        Allow exclude keys for each dataclass object.

        Use cases:
        - Decide what information will be included to be serialized to JSON or YAML that can be used in LLM prompt.
        - Exclude sensitive information from the serialized output.
        - Serialize the dataclass instance to a dictionary for saving states.

        Args:
            exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.


        Example:

        .. code-block:: python

            from dataclasses import dataclass
            from typing import List

            @dataclass
            class TrecData:
                question: str
                label: int

            @dataclass
            class TrecDataList(DataClass):

                data: List[TrecData]
                name: str

            trec_data = TrecData(question="What is the capital of France?", label=0)
            trec_data_list = TrecDataList(data=[trec_data], name="trec_data_list")

            trec_data_list.to_dict(exclude={"TrecData": ["label"], "TrecDataList": ["name"]})

            # Output:
            # {'data': [{'question': 'What is the capital of France?'}]}
        """
        if not is_dataclass(self):
            raise ValueError("to_dict() called on a class type, not an instance.")
        excluded: Optional[Dict[str, List[str]]] = None
        if exclude and isinstance(exclude, List):
            excluded = {self.__class__.__name__: exclude}
        elif exclude and isinstance(exclude, Dict):
            excluded = deepcopy(exclude)
        else:
            excluded = None
        return custom_asdict(self, exclude=excluded)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataClass":
        """Create a dataclass instance from a dictionary.

        Supports nested dataclasses, lists, and dictionaries.

        Example from the :meth:`to_dict` method.

        ..code-block:: python

            data_dict = trec_data_list.to_dict()
            restored_data = TreDataList.from_dict(data_dict)

            assert str(restored_data.__dict__) == str(trec_data_list.__dict__)

        .. note::
        If any required field is missing, it will raise an error.
        Do not use the dict that has excluded required fields.

        Use cases:
        - Convert the json/yaml output from LLM prediction to a dataclass instance.
        - Restore the dataclass instance from the serialized output used for states saving.
        """
        return dataclass_obj_from_dict(cls, data)

    @classmethod
    def from_json(cls, json_str: str) -> "DataClass":
        """Create a dataclass instance from a JSON string.

        Args:
            json_str (str): The JSON string to convert to a dataclass instance.

        Example:

        .. code-block:: python

            json_str = '{"question": "What is the capital of France?", "label": 0}'
            trec_data = TrecData.from_json(json_str)
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to load JSON string: {e}")

    def to_json_obj(self, exclude: ExcludeType = None) -> Any:
        r"""Convert the dataclass instance to a JSON object.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.
        This can be important to llm prompt.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return json.loads(self.to_json(exclude))

    def to_json(self, exclude: ExcludeType = None) -> str:
        r"""Convert the dataclass instance to a JSON string.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.
        This can be important to llm prompt.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return json.dumps(self.to_dict(exclude), indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataClass":
        """Create a dataclass instance from a YAML string.

        Args:
            yaml_str (str): The YAML string to convert to a dataclass instance.

        Example:

        .. code-block:: python

            yaml_str = 'question: What is the capital of France?\nlabel: 0'
            trec_data = TrecData.from_yaml(yaml_str)
        """
        try:
            data = yaml.safe_load(yaml_str)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to load YAML string: {e}")

    def to_yaml_obj(self, exclude: ExcludeType = None) -> Any:
        r"""Convert the dataclass instance to a YAML object.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return yaml.safe_load(self.to_yaml(exclude))

    def to_yaml(self, exclude: ExcludeType = None) -> str:
        r"""Convert the dataclass instance to a YAML string.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return yaml.dump(
            self.to_dict(exclude), default_flow_style=False, sort_keys=False
        )

    @classmethod
    def to_schema(cls, exclude: ExcludeType = None) -> Dict[str, Dict[str, Any]]:
        """Generate a Json schema which is more detailed than the signature."""
        # convert exclude to dict if it is a list
        excluded: Optional[Dict[str, List[str]]] = None
        if exclude and isinstance(exclude, List):
            excluded = {cls.__name__: exclude}
        elif exclude and isinstance(exclude, Dict):
            excluded = deepcopy(exclude)
        else:
            excluded = None
        return get_dataclass_schema(cls, excluded)

    @classmethod
    def to_schema_str(cls, exclude: ExcludeType = None) -> str:
        """Generate a Json schema which is more detailed than the signature."""
        schema = cls.to_schema(exclude)
        return json.dumps(schema, indent=4)

    @classmethod
    def to_yaml_signature(cls, exclude: ExcludeType = None) -> str:
        r"""Generate a YAML signature for the class from desc in metadata.

        Used mostly as LLM prompt to describe the output data format.
        """
        # NOTE: we manually format the yaml string as the yaml.dump will always sort the keys
        # Which can impact the final model output
        schema = cls.to_schema(exclude)
        signature_dict = convert_schema_to_signature(schema)
        yaml_content = []
        for key, value in signature_dict.items():
            yaml_content.append(f"{key}: {value}")

        yaml_output = "\n".join(yaml_content)
        return yaml_output

    @classmethod
    def to_json_signature(cls, exclude: ExcludeType = None) -> str:
        """Generate a JSON `signature`(json string) for the class from desc in metadata.

        Used mostly as LLM prompt to describe the output data format.

        Example:

        >>> @dataclass
        >>> class MyOutputs(DataClass):
        >>>    age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
        >>>    name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})

        >>> print(MyOutputs.to_json_signature())
        >>> # Output is a JSON string:
        >>> # '{
        >>> #    "age": "The age of the person (int) (required)",
        >>> #    "name": "The name of the person (str) (required)"
        >>> #}'
        """
        schema = cls.to_schema(exclude)
        signature_dict = convert_schema_to_signature(schema)

        return json.dumps(signature_dict, indent=4)

    @classmethod
    def to_dict_class(cls, exclude: ExcludeType = None) -> Dict[str, Any]:
        """More of an internal used class method for serialization.

        Converts the dataclass to a dictionary, optionally excluding specified fields.
        Use this to save states of the class in serialization, not advised to use in LLM prompt.
        """
        return cls.to_schema(exclude)

    @classmethod
    def format_class_str(
        cls,
        format_type: DataClassFormatType,
        exclude: ExcludeType = None,
    ) -> str:
        """Generate formatted output based on the type of operation and class/instance context.

        Args:
            format_type (DataClassFormatType): Specifies the format and type (schema, signature, example).

        Returns:
            str: A string representing the formatted output.

        Examples:

        .. code-block:: python

            # Define a dataclass
            from lightrag.core import DataClass

        """

        if format_type == DataClassFormatType.SIGNATURE_JSON:
            return cls.to_json_signature(exclude)
        elif format_type == DataClassFormatType.SIGNATURE_YAML:
            return cls.to_yaml_signature(exclude)

        elif format_type == DataClassFormatType.SCHEMA:
            return cls.to_schema_str(exclude)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def format_example_str(
        self, format_type: DataClassFormatType, exclude: ExcludeType = None
    ) -> str:
        """Generate formatted output based on the type of operation and class/instance context.

        Args:
            format_type (DataClassFormatType): Specifies the format and type (schema, signature, example).

        Returns:
            str: A string representing the formatted output.

        """

        # Check the type of format required and whether it's called on an instance or class
        if format_type == DataClassFormatType.EXAMPLE_JSON:
            return self.to_json(exclude)
        elif format_type == DataClassFormatType.EXAMPLE_YAML:
            return self.to_yaml(exclude)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


"""Reserved for Agent to automatically create a dataclass and to manipulate the code"""


@dataclass
class DynamicDataClassFactory:
    __doc__ = r"""
    This class is used to create a dynamic dataclass called `DynamicOutputs` from a dictionary.
    The dictionary should have the following structure:
    {
        "field_name": {
            "value": field_value,
            "desc": "Field description",
        },

    }

    Examples:

    .. code-block:: python

        data = {
            "age": {"value": 30, "desc": "The age of the person"},
            "name": {"value": "John Doe", "desc": "The name of the person"},
        }

        DynamicOutputs = DynamicDataClassFactory.create_from_dict(data)
        class_instance = DynamicOutputs()
        print(class_instance)

        # Output:
        # DataClass(age=30, name='John Doe')
    """

    @staticmethod
    def create_from_dict(data: dict, base_class=DataClass, class_name="DynamicOutputs"):
        fields_spec = []
        for key, value_dict in data.items():
            field_type = type(value_dict["value"])
            default_value = value_dict["value"]
            metadata = {
                "desc": value_dict.get("desc", "No description provided"),
            }
            fields_spec.append(
                (key, field_type, field(default=default_value, metadata=metadata))
            )

        dynamic_class = make_dataclass(class_name, fields_spec, bases=(base_class,))

        return dynamic_class
