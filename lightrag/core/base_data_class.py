"""
The data classes used to support core components.
"""

from typing import List, Dict, Any, Optional, TypeVar, Type
from dataclasses import (
    dataclass,
    field,
    fields,
    make_dataclass,
    MISSING,
    is_dataclass,
)

import json
import yaml
import warnings
import logging


logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


def required_field(name):
    r"""
    A patch for `TypeError: non-default argument follows default argument`

    Use default_factory=required_field to make a field required if field before has used default
    or default_factory before it.

    With this patch, our dataclass schema will make this a required field in string description.
    """
    raise TypeError(f"The '{name}' field is required and was not provided.")


def _get_data_class_schema(
    data_class: Type, exclude: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    r"""Helper function to get the schema of a BaseDataClass in type of Dict."""
    from dataclasses import fields, is_dataclass, MISSING

    if not is_dataclass(data_class):
        raise ValueError("Provided class is not a dataclass")
    schema: Dict[str, Dict] = {}
    if exclude is None:
        exclude = []
    for f in fields(data_class):
        field_name = f.name
        if field_name in exclude:
            continue

        field_info = {
            "type": f.type.__name__,
        }
        # add description if available
        if "desc" in f.metadata or "description" in f.metadata:
            field_info["desc"] = f.metadata.get("desc", f.metadata.get("description"))

        # Determine if the field is required or optional
        # Using __name__ to check for function identity
        if f.default is MISSING and (
            f.default_factory is MISSING
            or (
                hasattr(f.default_factory, "__name__")
                and f.default_factory.__name__ == "required_field"
            )
        ):
            field_info["required"] = True
        else:
            field_info["required"] = False
            # if f.default is not MISSING:
            #     field_info["default"] = f.default
            # elif f.default_factory is not MISSING:
            #     field_info["default"] = f.default_factory()

        schema[field_name] = field_info

    return schema


def convert_schema_to_signature(schema: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    r"""Convert the value from _get_data_class_schema to a string description."""

    signature = {}
    for field_name, field_info in schema.items():
        field_signature = field_info.get("desc", "")
        if not field_signature:
            field_signature = f"{field_name} ({field_info['type']})"
        if field_info["required"]:
            field_signature += " (required)"
        else:
            field_signature += " (optional)"
        signature[field_name] = field_signature
    return signature


@dataclass
class BaseDataClass:
    __doc__ = r"""Base class designed to streamline the handling, serialization, and description of data within our applications.

    Especially to LLM prompt.

    We explicitly handle this instead of relying on 3rd party libraries such as pydantic or marshmallow to have better
    transparency and to keep the order of the fields when get serialized.

    It creates string `signature` or `schema` for both the class type and the class instance.

    Schema is a standard way to describe the data structure in Json string.

    Signature is more token effcient than schema, and schema can mislead the model if it is not used properly.

    Better use schema with example signature (either yaml or json) depending on the use case.

    Example usage:
    ```
    # Define a dataclass
    @dataclass
    class MyOutputs(BaseDataClass):
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

    ```
    """

    def __post_init__(self):
        # TODO: use desription in the field
        for f in fields(self):
            if "desc" not in f.metadata:
                warnings.warn(
                    f"Field {f.name} is missing 'desc' in metadata", UserWarning
                )

    def set_field_value(self, field_name: str, value: Any):
        r"""Set the value of a field in the dataclass instance."""
        if field_name not in self.__dict__:  # check if the field exists
            logging.warning(f"Field {field_name} does not exist in the dataclass")
        setattr(self, field_name, value)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]):
        r"""
        Create a dataclass instance from a dictionary.
        """
        valid_data: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name in data:
                valid_data[f.name] = data[f.name]
        return cls(**valid_data)

    @classmethod
    def get_data_class_schema(
        cls, exclude: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Generate a Json schema which is more detailed than the signature."""
        return _get_data_class_schema(cls, exclude)

    @classmethod
    def to_yaml_signature(cls, exclude: Optional[List[str]] = None) -> str:
        r"""Generate a YAML signature for the class from desc in metadata.

        Used mostly as LLM prompt to describe the output data format.
        """
        # NOTE: we manually format the yaml string as the yaml.dump will always sort the keys
        # Which can impact the final model output
        schema = cls.get_data_class_schema(exclude)
        signature_dict = convert_schema_to_signature(schema)
        yaml_content = []
        for key, value in signature_dict.items():
            yaml_content.append(f"{key}: {value}")

        yaml_output = "\n".join(yaml_content)
        return yaml_output
        # return yaml.dump(signature_dict, default_flow_style=False)

    @classmethod
    def to_json_signature(cls, exclude: Optional[List[str]] = None) -> str:
        r"""Generate a JSON `signature`(json string) for the class from desc in metadata.

        Used mostly as LLM prompt to describe the output data format.

        Example:

        >>> @dataclass
        >>> class MyOutputs(BaseDataClass):
        >>>    age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
        >>>    name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})

        >>> print(MyOutputs.to_json_signature())
        >>> # Output is a JSON string:
        >>> # '{
        >>> #    "age": "The age of the person (int) (required)",
        >>> #    "name": "The name of the person (str) (required)"
        >>> #}'
        """
        schema = cls.get_data_class_schema(exclude)
        signature_dict = convert_schema_to_signature(schema)
        # manually format the json string as the json.dump will always sort the keys
        # Which can impact the final model output
        json_content = []
        for key, value in signature_dict.items():
            json_content.append(f'"{key}": "{value}"')

        # Join all parts with commas to form the complete JSON string
        json_output = ",\n".join(json_content)
        return "{\n" + json_output + "\n}"
        # return json.dumps(metadata_dict, indent=4)

    def to_yaml(self, exclude: Optional[List[str]] = None) -> str:
        """
        Convert the dataclass instance to a YAML string.
        Manually formats each field to ensure proper YAML output without unwanted characters.

        You can load it back to yaml object with:
        >>> yaml.safe_load(yaml_string)
        """
        exclude = exclude or []
        yaml_content = []
        for f in fields(self):
            if f.name and exclude and f.name in exclude:
                continue
            value = getattr(self, f.name)
            # Serialize value to a more controlled YAML format string
            if isinstance(value, str):
                # Directly format strings to ensure quotes are correctly placed
                value_formatted = f'"{value}"'
            elif isinstance(value, (list, dict)):
                # For complex types, use yaml.dump but strip unwanted newlines and marks
                value_formatted = (
                    yaml.dump(value, default_flow_style=False).strip().rstrip("\n...")
                )
            else:
                # Use yaml.dump for other types but ensure the output is clean
                value_formatted = (
                    yaml.dump(value, default_flow_style=False).strip().rstrip("\n...")
                )

            yaml_content.append(f"{f.name}: {value_formatted}")

        yaml_output = "\n".join(yaml_content)
        return yaml_output

    def to_json(self, exclude: Optional[List[str]] = None) -> str:
        """
        Convert the dataclass instance to a JSON string.
        Manually formats each field to ensure proper JSON output without unwanted characters.

        You can load it back to json object with:
        >>> json.loads(json_string)
        """
        exclude = exclude or []
        json_content = {}
        for f in fields(self):
            if f.name and exclude and f.name in exclude:
                continue
            value = getattr(self, f.name)
            # Serialize each field according to its type
            # For strings, integers, floats, booleans, directly assign
            # For lists and dicts, use json.dumps to ensure proper formatting
            if isinstance(value, (str, int, float, bool)):
                json_content[f.name] = value
            elif isinstance(value, (list, dict)):
                # Convert lists and dictionaries to a string and then parse it back to ensure correct format
                json_content[f.name] = json.loads(json.dumps(value))
            else:
                # Fallback for other types if necessary, can be customized further based on needs
                json_content[f.name] = str(value)

        # Convert the entire content dictionary to a JSON string
        json_output = json.dumps(json_content, indent=4)
        return json_output

    @classmethod
    def to_dict_class(cls, exclude: Optional[List[str]] = None) -> dict:
        """Converts the dataclass to a dictionary, optionally excluding specified fields.

        Use this to save states of the instance in serialization, not advised to use in LLM prompt.
        """
        return cls.get_data_class_schema(exclude)

    # TODO: maybe worth to support recursive to_dict for nested dataclasses
    # Can consider when we find the nested dataclass needs
    def to_dict(self, exclude: Optional[List[str]] = None) -> dict:
        """Converts the dataclass to a dictionary, optionally excluding specified fields.

        Use this to save states of the instance in serialization, not advised to use in LLM prompt.
        """
        # valid that it is called on a class instance not a class
        if not is_dataclass(self):
            raise ValueError("to_dict() called on a class type, not an instance.")
        if exclude is None:
            exclude = []
        exclude_set = set(exclude)

        data = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in exclude_set
        }

        return data


# Reserved for Agent to automatically create a dataclass and to manipulate the code
@dataclass
class DynamicDataClassFactory:
    __doc__ = r"""
    This class is used to create a dynamic dataclass called `DynamicOutputs` from a dictionary.
    The dictionary should have the following structure:
    {
        "field_name": {
            "value": field_value,
            "desc": "Field description",
            "prefix": "Field prefix",
        },
        ...
    }
    Examples:

    data = {
        "age": {"value": 30, "desc": "The age of the person", "prefix": "Age:"},
        "name": {"value": "John Doe", "desc": "The name of the person", "prefix": "Name:"},
    }

    DynamicOutputs = DynamicDataClassFactory.create_from_dict(data)
    class_instance = DynamicOutputs()
    print(class_instance)

    # Output:
    # BaseDataClass(age=30, name='John Doe')
    """

    @staticmethod
    def create_from_dict(data: dict, base_class=BaseDataClass):
        fields_spec = []
        for key, value_dict in data.items():
            field_type = type(value_dict["value"])
            default_value = value_dict["value"]
            metadata = {
                "desc": value_dict.get("desc", "No description provided"),
                "prefix": value_dict.get("prefix", ""),
            }
            fields_spec.append(
                (key, field_type, field(default=default_value, metadata=metadata))
            )

        dynamic_class = make_dataclass(
            "DynamicOutputs", fields_spec, bases=(base_class,)
        )

        return dynamic_class
