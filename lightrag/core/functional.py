"""Functional interface. 
Core functions we use to build across the components.
Users can leverage these functions to customize their own components."""

from typing import Dict, Any, Callable, Type
import re
import json


# TODO: test to convert all functions to component


def compose_model_kwargs(default_model_kwargs: Dict, model_kwargs: Dict) -> Dict:
    r"""
    The model configuration exclude the input itself.
    Combine the default model, model_kwargs with the passed model_kwargs.
    Example:
    model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
    self.model_kwargs = {"model": "gpt-3.5"}
    combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

    """
    pass_model_kwargs = default_model_kwargs.copy()

    if model_kwargs:
        pass_model_kwargs.update(model_kwargs)
    return pass_model_kwargs


# import hashlib
# import json


# def generate_component_key(component: Component) -> str:
#     """
#     Generates a unique key for a Component instance based on its type,
#     version, configuration, and nested components.
#     """
#     # Start with the basic information: class name and version
#     key_parts = {
#         "class_name": component._get_name(),
#         "version": getattr(component, "_version", 0),
#     }

#     # If the component stores configuration directly, serialize this configuration
#     if hasattr(component, "get_config"):
#         config = (
#             component.get_config()
#         )  # Ensure this method returns a serializable dictionary
#         key_parts["config"] = json.dumps(config, sort_keys=True)

#     # If the component contains other components, include their keys
#     if hasattr(component, "_components") and component._components:
#         nested_keys = {}
#         for name, subcomponent in component._components.items():
#             if subcomponent:
#                 nested_keys[name] = generate_component_key(subcomponent)
#         key_parts["nested"] = nested_keys

#     # Serialize key_parts to a string and hash it
#     key_str = json.dumps(key_parts, sort_keys=True)
#     return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def generate_readable_key_for_function(fn: Callable) -> str:

    module_name = fn.__module__
    function_name = fn.__name__
    return f"{module_name}.{function_name}"


def extract_json_str(text: str, add_missing_right_brace: bool = True) -> str:
    """
    Extract JSON string from text.
    NOTE: Only handles the first JSON object found in the text. And it expects at least one JSON object in the text.
    If right brace is not found, we add one to the end of the string.
    """
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    text = text.strip().replace("{{", "{").replace("}}", "}")
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in the text: {text}")

    # Attempt to find the matching closing brace
    brace_count = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            end = i
            break

    if end == -1 and add_missing_right_brace:
        # If no closing brace is found, but we are allowed to add one
        text += "}"
        end = len(text) - 1
    elif end == -1:
        raise ValueError(
            "Incomplete JSON object found and add_missing_right_brace is False."
        )

    return text[start : end + 1]


def extract_list_str(text: str, add_missing_right_bracket: bool = True) -> str:
    """
    Extract the first complete list string from the provided text. If the list string is incomplete
    (missing the closing bracket), an option allows adding a closing bracket at the end.

    Args:
        text (str): The text containing potential list data.
        add_missing_right_bracket (bool): Whether to add a closing bracket if it is missing.

    Returns:
        str: The extracted list string.

    Raises:
        ValueError: If no list is found or if the list extraction is incomplete
                    without the option to add a missing bracket.
    """
    text = text.strip()
    start = text.find("[")
    if start == -1:
        raise ValueError("No list found in the text.")

    # Attempt to find the matching closing bracket
    bracket_count = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "[":
            bracket_count += 1
        elif text[i] == "]":
            bracket_count -= 1

        if bracket_count == 0:
            end = i
            break

    if end == -1 and add_missing_right_bracket:
        # If no closing bracket is found, but we are allowed to add one
        text += "]"
        end = len(text) - 1
    elif end == -1:
        raise ValueError(
            "Incomplete list found and add_missing_right_bracket is False."
        )

    return text[start : end + 1]


def extract_yaml_str(text: str) -> str:
    r"""Extract YAML string from text.

    In default, we use regex pattern to match yaml code blocks within triple backticks with optional yaml or yml prefix.
    """
    try:
        yaml_re_pattern: re.Pattern = re.compile(
            r"^```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL
        )
        match = yaml_re_pattern.search(text.strip())

        yaml_str = ""
        if match:
            yaml_str = match.group("yaml")
        else:
            yaml_str = text
        return yaml_str
    except Exception as e:
        raise ValueError(f"Failed to extract YAML from text: {e}")


def fix_json_missing_commas(json_str: str) -> str:
    # Example: adding missing commas, only after double quotes
    # Regular expression to find missing commas
    regex = r'(?<=[}\]"\'\d])(\s+)(?=[\{"\[])'

    # Add commas where missing
    fixed_json_str = re.sub(regex, r",\1", json_str)

    return fixed_json_str


def fix_json_escaped_single_quotes(json_str: str) -> str:
    # First, replace improperly escaped single quotes inside strings
    # json_str = re.sub(r"(?<!\\)\'", '"', json_str)
    # Fix escaped single quotes
    json_str = json_str.replace(r"\'", "'")
    return json_str


def parse_yaml_str_to_obj(yaml_str: str) -> Dict[str, Any]:
    r"""
    Parse a YAML string to a Python object.
    yaml_str: has to be a valid YAML string.
    """
    try:
        import yaml

        yaml_obj = yaml.safe_load(yaml_str)
        return yaml_obj
    except yaml.YAMLError as e:
        raise ValueError(
            f"Got invalid YAML object. Error: {e}. Got YAML string: {yaml_str}"
        )
    except NameError as exc:
        raise ImportError("Please pip install PyYAML.") from exc


def parse_json_str_to_obj(json_str: str) -> Dict[str, Any]:
    r"""
    Parse a JSON string to a Python object.
    json_str: has to be a valid JSON string. Either {} or [].
    """
    json_str = json_str.strip()
    try:
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        # 2nd attemp after fixing the json string
        try:
            print("Trying to fix potential missing commas...")
            json_str = fix_json_missing_commas(json_str)
            print("Trying to fix scaped single quotes...")
            json_str = fix_json_escaped_single_quotes(json_str)
            print(f"Fixed JSON string: {json_str}")
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            # 3rd attemp using yaml
            try:
                import yaml

                # NOTE: parsing again with pyyaml
                #       pyyaml is less strict, and allows for trailing commas
                #       right now we rely on this since guidance program generates
                #       trailing commas
                print("Parsing JSON string with PyYAML...")
                json_obj = yaml.safe_load(json_str)
                return json_obj
            except yaml.YAMLError as e_yaml:
                raise ValueError(
                    f"Got invalid JSON object. Error: {e}. Got JSON string: {json_str}"
                )
            except NameError as exc:
                raise ImportError("Please pip install PyYAML.") from exc
