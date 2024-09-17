"""A base class that provides an easy way for data to interact with LLMs."""

from typing import List, Dict, Any, Optional, Union, Callable, Type
import collections
from collections import OrderedDict


import enum
from copy import deepcopy
from dataclasses import (
    field,
    fields,
    make_dataclass,
    is_dataclass,
)

import json
import yaml
import logging

from adalflow.core.functional import (
    # dataclass_obj_to_dict,
    custom_asdict,
    dataclass_obj_from_dict,
    get_dataclass_schema,
    convert_schema_to_signature,
    represent_ordereddict,
)

__all__ = [
    "DataClass",
    "DataClassFormatType",
    "required_field",
    "ExcludeType",
    "IncludeType",
    "check_adal_dataclass",
    "DynamicDataClassFactory",
]
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
        from adalflow.core.base_data_class import required_field, DataClass

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
IncludeType = Optional[Union[List[str], Dict[str, List[str]]]]


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

    We defined :class:`DataClassFormatType<core.types.DataClassFormatType>` to categorize DataClass description formats
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
        1. Avoid using Optional[Type] for the type of fields, as dataclass already distingushes between optional and required fields using default value.
        2. If you need to customize, you can subclass and overwrite any method to fit your needs.

    Loading data:

    - :meth:`from_dict` is used to create a dataclass instance from a dictionary.


    Refer :ref:`DataClass<core-base_data_class_note>` for more detailed instructions.

    Examples:

    .. code-block:: python

        # Define a dataclass
        from adalflow.core import DataClass
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
    __input_fields__: List[str] = []
    __output_fields__: List[str] = []

    def __post_init__(self):

        for f in fields(self):
            if "desc" not in f.metadata and "description" not in f.metadata:

                logger.debug(
                    f"Class {  self.__class__.__name__} Field {f.name} is missing 'desc' in metadata"
                )

    @classmethod
    def get_task_desc(cls) -> str:
        """Get the task description for the dataclass.

        Returns:
            str: The task description for the dataclass.
        """
        return cls.__doc__

    @classmethod
    def set_task_desc(cls, task_desc: str) -> None:
        """Set the task description for the dataclass.

        Args:
            task_desc (str): The task description to set.
        """
        cls.__doc__ = task_desc

    @classmethod
    def get_input_fields(cls):
        """Return a list of all input fields."""
        return cls.__input_fields__

    @classmethod
    def set_input_fields(cls, input_fields: List[str]):
        """Set the input fields for the dataclass.
          When creating schema or instance, it will follow the input field and output field order

        Args:
            input_fields (List[str]): The input fields to set.
        """
        cls.__input_fields__ = input_fields

    @classmethod
    def get_output_fields(cls):
        """Return a list of all output fields."""
        return cls.__output_fields__

    @classmethod
    def set_output_fields(cls, output_fields: List[str]):
        """Set the output fields for the dataclass.
          When creating schema or instance, it will follow the input field and output field order

        Args:
            output_fields (List[str]): The output fields to set.
        """
        cls.__output_fields__ = output_fields

    def to_dict(
        self,
        *,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> Dict[str, Any]:
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
        # convert all fields to its data if its parameter
        fields = self.__dataclass_fields__
        from adalflow.optim.parameter import Parameter

        for f in fields.values():
            # print(f"field: {f}")
            field_value = getattr(self, f.name)
            # if its a parameter, convert to its data
            if isinstance(field_value, Parameter):
                setattr(self, f.name, field_value.data)
        # print(f"adapted self: {self}")

        # ensure only either include or exclude is used not both
        if include and exclude:
            raise ValueError("Either include or exclude can be used, not both.")

        # convert include to excluded

        excluded: Optional[Dict[str, List[str]]] = None
        if include:  # only support unnested fields
            # fild all fields of the class
            fields = self.__dataclass_fields__
            # generate the excluded dict
            excluded = {
                self.__class__.__name__: [
                    f.name for f in fields.values() if f.name not in include
                ]
            }
        elif exclude:
            if exclude and isinstance(exclude, List):
                excluded = {self.__class__.__name__: exclude}
            elif exclude and isinstance(exclude, Dict):
                excluded = deepcopy(exclude)
            else:
                excluded = None
        # return custom_asdict(self, exclude=excluded)
        # Convert the dataclass to a dictionary

        raw_dict = custom_asdict(self, exclude=excluded)

        # Reorder the dictionary based on input_field and output_field
        input_fields = self.get_input_fields()
        output_fields = self.get_output_fields()

        ordered_dict = OrderedDict()

        # First, add input fields in order
        for field_name in input_fields:
            if field_name in raw_dict:
                ordered_dict[field_name] = raw_dict[field_name]

        # Then, add output fields in order
        for field_name in output_fields:
            if field_name in raw_dict:
                ordered_dict[field_name] = raw_dict[field_name]

        # Finally, add any remaining fields (if there are any)
        for field_name, value in raw_dict.items():
            if field_name not in ordered_dict:
                ordered_dict[field_name] = value

        return dict(ordered_dict)

        return ordered_dict

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
        try:
            dclass = dataclass_obj_from_dict(cls, data)
            logger.debug(f"Dataclass instance created from dict: {dclass}")
            return dclass
        except TypeError as e:
            raise ValueError(f"Failed to load data: {e}")

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

    def to_json_obj(
        self,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> Any:
        r"""Convert the dataclass instance to a JSON object.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.
        This can be important to llm prompt.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return json.loads(self.to_json(exclude=exclude, include=include))

    def to_json(
        self,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        r"""Convert the dataclass instance to a JSON string.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.
        This can be important to llm prompt.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return json.dumps(
            self.to_dict(exclude=exclude, include=include), indent=4, sort_keys=False
        )

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

    def to_yaml_obj(
        self,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> Any:
        r"""Convert the dataclass instance to a YAML object.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return yaml.safe_load(self.to_yaml(exclude=exclude, include=include))

    def to_yaml(
        self,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        r"""Convert the dataclass instance to a YAML string.

        :meth:`to_dict` along with the use of sort_keys=False to ensure the order of the fields is maintained.

        Args:

        exclude (Optional[Dict[str, List[str]]], optional): A dictionary of fields to exclude for each dataclass object. Defaults to None.
        """
        return yaml.dump(
            self.to_dict(exclude=exclude, include=include),
            default_flow_style=False,
            sort_keys=False,
        ).strip()

    def dict_to_yaml(self, data: Dict[str, Any]) -> str:
        """Convert a dictionary to a YAML string.

        Args:
            data (Dict[str, Any]): The dictionary to convert to a YAML string.

        Returns:
            str: The YAML string representation of the dictionary.
        """
        return yaml.dump(data, default_flow_style=False, sort_keys=False).strip()

    @classmethod
    def to_schema(
        cls,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate a Json schema which is more detailed than the signature."""
        # convert exclude to dict if it is a list
        if include and exclude:
            raise ValueError("Either include or exclude can be used, not both.")
        excluded: Optional[Dict[str, List[str]]] = None
        if include:  # only support unnested fields
            # fild all fields of the class
            fields = cls.__dataclass_fields__
            # generate the excluded dict
            excluded = {
                cls.__name__: [f.name for f in fields.values() if f.name not in include]
            }
        elif exclude:
            if exclude and isinstance(exclude, List):
                excluded = {cls.__name__: exclude}
            elif exclude and isinstance(exclude, Dict):
                excluded = deepcopy(exclude)
            else:
                excluded = None

        raw_dict = get_dataclass_schema(
            cls, excluded, getattr(cls, "__type_var_map__", None)
        )
        # Reorder the dictionary based on input_field and output_field

        properties = raw_dict.get("properties", {})
        # reorder the properties fields
        input_fields = cls.get_input_fields()
        output_fields = cls.get_output_fields()

        ordered_dict = OrderedDict()

        # First, add input fields in order
        for field_name in input_fields:
            if field_name in properties:
                ordered_dict[field_name] = properties[field_name]

        # Then, add output fields in order
        for field_name in output_fields:
            if field_name in properties:
                ordered_dict[field_name] = properties[field_name]

        # Finally, add any remaining fields (if there are any)
        for field_name, value in properties.items():
            if field_name not in ordered_dict:
                ordered_dict[field_name] = value

        # Update the properties field
        raw_dict["properties"] = ordered_dict

        return raw_dict

    @classmethod
    def to_schema_str(
        cls,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        """Generate a Json schema which is more detailed than the signature."""
        schema = cls.to_schema(exclude=exclude, include=include)
        return json.dumps(schema, indent=4).strip()

    @classmethod
    def to_yaml_signature(
        cls,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        r"""Generate a YAML signature for the class from desc in metadata.

        Used mostly as LLM prompt to describe the output data format.
        """
        # NOTE: we manually format the yaml string as the yaml.dump will always sort the keys
        # Which can impact the final model output
        schema = cls.to_schema(exclude=exclude, include=include)
        signature_dict = convert_schema_to_signature(schema)
        yaml_content = []
        for key, value in signature_dict.items():
            yaml_content.append(f"{key}: {value}")

        yaml_output = "\n".join(yaml_content)
        return yaml_output

    @classmethod
    def to_json_signature(
        cls,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
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
        schema = cls.to_schema(exclude=exclude, include=include)
        signature_dict = convert_schema_to_signature(schema)

        return json.dumps(signature_dict, indent=4)

    @classmethod
    def to_dict_class(
        cls,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> Dict[str, Any]:
        """More of an internal used class method for serialization.

        Converts the dataclass to a dictionary, optionally excluding specified fields.
        Use this to save states of the class in serialization, not advised to use in LLM prompt.
        """
        return cls.to_schema(exclude=exclude, include=include)

    @classmethod
    def format_class_str(
        cls,
        format_type: DataClassFormatType,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        """Generate formatted output based on the type of operation and class/instance context.

        Args:
            format_type (DataClassFormatType): Specifies the format and type (schema, signature, example).

        Returns:
            str: A string representing the formatted output.

        Examples:

        .. code-block:: python

            # Define a dataclass
            from adalflow.core import DataClass

        """

        if format_type == DataClassFormatType.SIGNATURE_JSON:
            return cls.to_json_signature(exclude=exclude, include=include)
        elif format_type == DataClassFormatType.SIGNATURE_YAML:
            return cls.to_yaml_signature(exclude=exclude, include=include)

        elif format_type == DataClassFormatType.SCHEMA:
            return cls.to_schema_str(exclude=exclude, include=include)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def format_example_str(
        self,
        format_type: DataClassFormatType,
        exclude: ExcludeType = None,
        include: IncludeType = None,
    ) -> str:
        """Generate formatted output based on the type of operation and class/instance context.

        Args:
            format_type (DataClassFormatType): Specifies the format and type (schema, signature, example).

        Returns:
            str: A string representing the formatted output.

        """

        # Check the type of format required and whether it's called on an instance or class
        if format_type == DataClassFormatType.EXAMPLE_JSON:
            return self.to_json(exclude=exclude, include=include)
        elif format_type == DataClassFormatType.EXAMPLE_YAML:
            return self.to_yaml(exclude=exclude, include=include)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


def check_adal_dataclass(data_class: Type) -> None:
    """Check if the provided class is a valid dataclass for the AdalFlow framework.

    Args:
        data_class (Type): The class to check.
    """

    if not is_dataclass(data_class):
        raise TypeError(f"Provided class is not a dataclass: {data_class}")

    if not issubclass(data_class, DataClass):
        raise TypeError(f"Provided class is not a subclass of DataClass: {data_class}")


class DynamicDataClassFactory:
    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        base_class: Type = DataClass,
        class_name: str = "DynamicDataClass",
    ) -> DataClass:
        """
        Create an instance of a dataclass from a dictionary. The dictionary should have the following structure:
        {
            "field_name": field_value,
            ...
        }

        Args:
            data (dict): The dictionary with field names and values.
            base_class (type): The base class to inherit from (default: BaseDataClass).
            class_name (str): The name of the generated dataclass (default: DynamicDataClass).

        Returns:
            BaseDataClass: An instance of the generated dataclass.
        """
        # Create field specifications for the dataclass
        # fields_spec = [
        #     (key, type(value), field(default=value)) for key, value in data.items()
        # ]
        fields_spec = []
        for key, value in data.items():
            field_type = type(value)
            if isinstance(value, (list, dict, set)):
                fields_spec.append(
                    (key, field_type, field(default_factory=lambda v=value: v))
                )
            else:
                fields_spec.append((key, field_type, field(default=value)))

        # Create the dataclass
        dynamic_class = make_dataclass(class_name, fields_spec, bases=(base_class,))

        # Create an instance of the dataclass with the provided values
        instance = dynamic_class.from_dict(data)

        return instance
