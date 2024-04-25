"""
LLM applications requires lots of string processing. Such as the text output needed to be parsed into:
(1) JSON format or other formats
(2) SQL/Python valid format
(3) Tool call format

We design this the string_parser module to be generic to any input text without differentiate them as input text or output text.
TODO: function and arguments parser
"""

import re

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import json
import ast


############################################################################################################
# String as other output format such as JSON, List, etc.
############################################################################################################
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


def fix_json_formatting(json_str: str) -> str:
    # Example: adding missing commas, only after double quotes
    # Regular expression to find missing commas
    regex = r'(?<=[}\]"\'\d])(\s+)(?=[\{"\[])'

    # Add commas where missing
    fixed_json_str = re.sub(regex, r",\1", json_str)
    # print("adding commas when missing, fixed_json_str: ", fixed_json_str)

    return fixed_json_str


############################################################################################################
# Often used as the output parser for LLM text output
############################################################################################################
class BaseTextParser(ABC):
    """
    A base class for text parsers. Good for type hinting and inheritance.
    """

    @abstractmethod
    def __call__(self, text: str) -> Any:
        """
        Parse the provided text and return the parsed data.
        """
        pass


class JsonParser(BaseTextParser):
    """
    A text parser for extracting JSON strings from text to json object.
    NOTE: ensure only pass one json string in the text.
    """

    def __init__(
        self, add_missing_right_brace: bool = True, fix_missing_commas: bool = True
    ):
        self.add_missing_right_brace = add_missing_right_brace
        self.fix_missing_commas = fix_missing_commas

    def __call__(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            json_str = extract_json_str(text, self.add_missing_right_brace)
            # 1st attempt to load the json string
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            # 2nd attemp after fixing the json string
            if self.fix_missing_commas:
                try:
                    print("Fixing JSON formatting issues...")
                    json_str = fix_json_formatting(json_str)
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
                        json_obj = yaml.safe_load(json_str)
                        return json_obj
                    except yaml.YAMLError as e_yaml:
                        raise ValueError(
                            f"Got invalid JSON object. Error: {e}. Got JSON string: {json_str}"
                        )
                    except NameError as exc:
                        raise ImportError("Please pip install PyYAML.") from exc


############################################################################################################
# String as function call
############################################################################################################
def evaluate_ast_node(node: ast.AST, context_map: Dict[str, Any] = None):
    """
    Recursively evaluates an AST node and returns the corresponding Python object.

    Args:
        node (ast.AST): The AST node to evaluate. This node can represent various parts of Python expressions,
                        such as literals, identifiers, lists, dictionaries, and function calls.
        context_map (Dict[str, Any]): A dictionary that maps variable names to their respective values and functions.
                                      This context is used to resolve names and execute functions.

    Returns:
        Any: The result of evaluating the node. The type of the returned object depends on the nature of the node:
             - Constants return their literal value.
             - Names are looked up in the context_map.
             - Lists and tuples return their contained values as a list or tuple.
             - Dictionaries return a dictionary with keys and values evaluated.
             - Function calls invoke the function with evaluated arguments and return its result.

    Raises:
        ValueError: If the node type is unsupported, a ValueError is raised indicating the inability to evaluate the node.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Dict):
        return {
            evaluate_ast_node(k): evaluate_ast_node(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.List):
        return [evaluate_ast_node(elem) for elem in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(evaluate_ast_node(elem) for elem in node.elts)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate_ast_node(node.operand, context_map)  # unary minus
    elif isinstance(node, ast.Name):  # variable name
        return context_map[node.id]
    elif isinstance(node, ast.Call):  # another fun or class as argument and value
        func = evaluate_ast_node(node.func, context_map)
        args = [evaluate_ast_node(arg, context_map) for arg in node.args]
        kwargs = {
            kw.arg: evaluate_ast_node(kw.value, context_map) for kw in node.keywords
        }
        print(f"another fun or class as argument and value: {func}, {args}, {kwargs}")
        output = func(*args, **kwargs)
        print(f"output: {output}")
        return output
    else:
        raise ValueError(f"Unsupported AST node type: {type(node)}")


def parse_function_call(
    call_string: str, context_map: Dict[str, Any] = None
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """
    Parse a string representing a function call into its components.
    Args:
        call_string (str): The string representing the function call.
        context_map (Dict[str, Any]): A dictionary that maps variable names to their respective values and functions.
                                      This context is used to resolve names and execute functions.
    """
    call_string = call_string.strip()
    # Parse the string into an AST
    tree = ast.parse(call_string, mode="eval")

    if isinstance(tree.body, ast.Call):
        # Extract the function name
        func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None

        # Prepare the list of arguments and keyword arguments
        args = [evaluate_ast_node(arg, context_map) for arg in tree.body.args]
        keywords = {
            kw.arg: evaluate_ast_node(kw.value, context_map)
            for kw in tree.body.keywords
        }

        return func_name, args, keywords
    else:
        raise ValueError("Provided string is not a function call.")


if __name__ == "__main__":
    test_input = (
        '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}'
    )
    print(
        extract_json_str(test_input, add_missing_right_brace=True)
    )  # Expected to complete the JSON properly

    test_input_2 = 'Some random text before {"key1": "value1"} and more after'
    print(extract_json_str(test_input_2))  # Expected to extract {"key1": "value1"}

    test_input_3 = "No JSON here"
    try:
        print(extract_json_str(test_input_3))
    except ValueError as e:
        print(e)  # Expected to raise an error about no JSON object found
