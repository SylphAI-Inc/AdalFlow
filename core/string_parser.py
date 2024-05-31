"""
LLM applications requires lots of string processing. Such as the text output needed to be parsed into:
(1) JSON format or other formats
(2) SQL/Python valid format
(3) Tool(function) call format

We design this these string_parser modules to be generic to any input text without differentiating them as input text or output text.
"""

from typing import Any, Dict, List, Tuple
import ast
from core.tool_helper import ToolOutput
from core.component import Component
import core.functional as F
from core.data_classes import Output


class ListParser(Component):
    """
    A text parser for extracting list strings from text to list object.
    NOTE: ensure only pass one list string in the text.
    You can use `extract_list_str` to extract the first list string from the text.
    """

    def __init__(self, add_missing_right_bracket: bool = True):
        super().__init__()
        self.add_missing_right_bracket = add_missing_right_bracket

    def __call__(self, input: str) -> List[Any]:
        list_str = F.extract_list_str(input, self.add_missing_right_bracket)
        list_obj = F.parse_json_str_to_obj(list_str)
        return list_obj


JASON_PARSER_OUTPUT_TYPE = Dict[str, Any]


class JsonParser(Component):
    """
    A text parser for extracting JSON strings from text to json object.
    NOTE: ensure only pass one json string in the text.
    You can use `extract_json_str` to extract the first json string from the text.
    """

    def __init__(self, add_missing_right_brace: bool = True):
        super().__init__()
        self.add_missing_right_brace = add_missing_right_brace

    def call(self, input: str) -> JASON_PARSER_OUTPUT_TYPE:
        input = input.strip()
        json_str = F.extract_json_str(input, self.add_missing_right_brace)
        json_obj = F.parse_json_str_to_obj(json_str)
        return json_obj


YAML_PARSER_OUTPUT_TYPE = Output[Dict[str, Any]]


class YAMLParser(Component):
    __doc__ = r"""A text parser for extracting YAML strings and parsing them into a JSON object.

    Examples:
        >>> yaml_parser = YAMLParser()
        >>> yaml_str = "```yaml\nkey: value\n```"
        >>> yaml_obj = yaml_parser(yaml_str)
        >>> print(yaml_obj)
        {'key': 'value'}
    """

    def __init__(self):
        super().__init__()

    def call(self, input: str) -> YAML_PARSER_OUTPUT_TYPE:
        input = input.strip()
        try:
            yaml_str = F.extract_yaml_str(input)
            yaml_obj = F.parse_yaml_str_to_obj(yaml_str)
            output = Output(data=yaml_obj)
            return output
        except Exception as e:
            return Output(error=str(e))
            # track the error message

        # yaml_str = F.extract_yaml_str(input)
        # json_object = F.parse_yaml_str_to_obj(yaml_str)
        # return json_object


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
    elif isinstance(
        node, ast.BinOp
    ):  # support "multiply(2024-2017, 12)", the "2024-2017" is a "BinOp" node
        left = evaluate_ast_node(node.left, context_map)
        right = evaluate_ast_node(node.right, context_map)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Mod):
            return left % right
        elif isinstance(node.op, ast.Pow):
            return left**right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
    elif isinstance(node, ast.Name):  # variable name
        try:
            output_fun = context_map[node.id]
            return output_fun
        # TODO: raise the error back to the caller so that the llm can get the error message
        except KeyError as e:
            raise ValueError(
                f"Error: {e}, {node.id} does not exist in the context_map."
            )

    elif isinstance(
        node, ast.Call
    ):  # another fun or class as argument and value, e.g. add( multiply(4,5), 3)
        func = evaluate_ast_node(node.func, context_map)
        args = [evaluate_ast_node(arg, context_map) for arg in node.args]
        kwargs = {
            kw.arg: evaluate_ast_node(kw.value, context_map) for kw in node.keywords
        }
        print(f"another fun or class as argument and value: {func}, {args}, {kwargs}")
        output = func(*args, **kwargs)
        if isinstance(output, ToolOutput):
            return output.raw_output
        print(f"output: {output}")
        return output
    else:
        raise ValueError(f"Unsupported AST node type: {type(node)}")


def parse_function_call(
    call_string: str, context_map: Dict[str, Any] = None
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """
    Parse a string representing a function call into its components and ensure safe execution by only allowing function calls from a predefined context map.
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
    # test_input = (
    #     '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}'
    # )
    # print(
    #     extract_json_str(test_input, add_missing_right_brace=True)
    # )  # Expected to complete the JSON properly

    # test_input_2 = 'Some random text before {"key1": "value1"} and more after'
    # print(extract_json_str(test_input_2))  # Expected to extract {"key1": "value1"}

    # test_input_3 = "No JSON here"
    # try:
    #     print(extract_json_str(test_input_3))
    # except ValueError as e:
    #     print(e)  # Expected to raise an error about no JSON object found

    # test list parser
    list_parser = ListParser()
    test_input_4 = 'Some random text before ["item1", "item2"] and more after'
    print(list_parser(test_input_4))  # Expected to extract ["item1", "item2"]
