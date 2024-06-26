"""Functional interface.
Core functions we use to build across the components.
Users can leverage these functions to customize their own components."""

from typing import (
    Dict,
    Any,
    Callable,
    Union,
    List,
    Tuple,
    Optional,
    Type,
    get_type_hints,
)
import logging
import numpy as np
import re
import json
import yaml
import ast
from inspect import signature, Parameter
from dataclasses import fields, is_dataclass

log = logging.getLogger(__name__)


########################################################################################
# For FunctionTool component
########################################################################################
def get_fun_schema(name: str, func: Callable[..., object]) -> Dict[str, object]:
    r"""Get the schema of a function.
    Support dataclass, Union and normal data types such as int, str, float, etc, list, dict, set.

    Examples:
    def example_function(x: int, y: str = "default") -> int:
        return x
    schema = get_fun_schema("example_function", example_function)
    print(json.dumps(schema, indent=4))
    # Output:
    {
        "type": "object",
        "properties": {
            "x": {
                "type": "int"
            },
            "y": {
                "type": "str",
                "default": "default"
            }
        },
        "required": [
            "x"
        ]
    }
    """
    sig = signature(func)
    schema = {"type": "object", "properties": {}, "required": []}
    type_hints = get_type_hints(func)

    for param_name, parameter in sig.parameters.items():
        param_type = type_hints.get(param_name, "Any")
        if parameter.default == Parameter.empty:
            schema["required"].append(param_name)
            schema["properties"][param_name] = {**get_type_schema(param_type)}
        else:
            schema["properties"][param_name] = {
                **get_type_schema(param_type),
                "default": parameter.default,
            }

    return schema


def get_type_schema(param_type: object) -> Dict[str, Any]:
    if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
        return {
            "type": "Union",
            "choices": [get_type_schema(arg) for arg in param_type.__args__],
        }
    elif is_dataclass(param_type):
        return get_dataclass_schema(param_type)
    elif hasattr(param_type, "__name__"):
        return {"type": param_type.__name__}
    else:
        return {"type": "Any"}


def get_dataclass_schema(cls):
    """Generate schema for a dataclass."""
    data_class_type = cls.__name__
    schema = {"type": data_class_type, "properties": {}, "required": []}
    for field_ in fields(cls):
        field_schema = {"type": field_.type.__name__}
        if field_.default != field_.default_factory:
            field_schema["default"] = field_.default
        if field_.metadata:
            field_schema.update(field_.metadata)
        schema["properties"][field_.name] = field_schema
        if field_.default == field_.default_factory:
            schema["required"].append(field_.name)

    return schema


# For parse function call for FunctionTool component
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
    elif isinstance(node, ast.Attribute):  # e.g. math.pi
        value = evaluate_ast_node(node.value, context_map)
        return getattr(value, node.attr)

    elif isinstance(
        node, ast.Call
    ):  # another fun or class as argument and value, e.g. add( multiply(4,5), 3)
        func = evaluate_ast_node(node.func, context_map)
        args = [evaluate_ast_node(arg, context_map) for arg in node.args]
        kwargs = {
            kw.arg: evaluate_ast_node(kw.value, context_map) for kw in node.keywords
        }
        output = func(*args, **kwargs)
        if hasattr(output, "raw_output"):
            return output.raw_output
        return output
    else:
        # directly evaluate the node
        # print(f"Unsupported AST node type: {type(node)}")
        # return eval(compile(ast.Expression(node), filename="<ast>", mode="eval"))
        raise ValueError(f"Unsupported AST node type: {type(node)}")


def parse_function_call_expr(
    function_expr: str, context_map: Dict[str, Any] = None
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """
    Parse a string representing a function call into its components and ensure safe execution by only allowing function calls from a predefined context map.
    Args:
        function_expr (str): The string representing the function
        context_map (Dict[str, Any]): A dictionary that maps variable names to their respective values and functions.
                                      This context is used to resolve names and execute functions.
    """
    function_expr = function_expr.strip()
    # Parse the string into an AST
    tree = ast.parse(function_expr, mode="eval")

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


def generate_function_call_expression_from_callable(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> str:
    """
    Generate a function call expression string from a callable function and its arguments.

    Args:
        func (Callable[..., Any]): The callable function.
        *args (Any): Positional arguments to be passed to the function.
        **kwargs (Any): Keyword arguments to be passed to the function.

    Returns:
        str: The function call expression string.
    """
    func_name = func.__name__
    args_str = ", ".join(repr(arg) for arg in args)
    kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())

    if args_str and kwargs_str:
        full_args_str = f"{args_str}, {kwargs_str}"
    else:
        full_args_str = args_str or kwargs_str

    return f"{func_name}({full_args_str})"


import threading

# Define a list of safe built-ins
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def sandbox_exec(code: str, context=SAFE_BUILTINS, timeout: int = 5) -> Dict:
    r"""Execute code in a sandboxed environment with a timeout.

    Works similar to eval(), but with timeout and context similar to parse_function_call_expr.

    Args:
        code (str): The code to execute. Has to be output=... or similar so that the result can be captured.
        context (Dict[str, Any]): The context to use for the execution.
        timeout (int): The execution timeout in seconds.
    """
    result = {"output": None, "error": None}
    try:
        compiled_code = compile(code, "<string>", "exec")

        # Result dictionary to store execution results

        # Define a target function for the thread
        def target():
            try:
                # Execute the code
                exec(compiled_code, context, result)
            except Exception as e:
                result["error"] = e

        # Create a thread to execute the code
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        # Check if the thread is still alive (timed out)
        if thread.is_alive():
            result["error"] = TimeoutError("Execution timed out")
            raise TimeoutError("Execution timed out")
    except Exception as e:
        print(f"Errpr at sandbox_exec: {e}")
        raise e

    return result


########################################################################################
# For Dataclass base class
########################################################################################
def dataclass_obj_to_dict(
    obj: Any, exclude: Optional[Dict[str, List[str]]] = None, parent_key: str = ""
) -> Dict[str, Any]:
    r"""Convert a dataclass object to a dictionary.

    Supports nested dataclasses, lists, and dictionaries.
    Allow exclude keys for each dataclass object.
    Example:

    .. code-block:: python

       from dataclasses import dataclass
       from typing import List

       @dataclass
       class TrecData:
           question: str
           label: int

       @dataclass
       class TrecDataList:

           data: List[TrecData]
           name: str

       trec_data = TrecData(question="What is the capital of France?", label=0)
       trec_data_list = TrecDataList(data=[trec_data], name="trec_data_list")

       dataclass_obj_to_dict(trec_data_list, exclude={"TrecData": ["label"], "TrecDataList": ["name"]})

       # Output:
       # {'data': [{'question': 'What is the capital of France?'}]}

    """
    if exclude is None:
        exclude = {}

    obj_class_name = obj.__class__.__name__
    current_exclude = exclude.get(obj_class_name, [])

    if hasattr(obj, "__dataclass_fields__"):
        return {
            key: dataclass_obj_to_dict(value, exclude, parent_key=key)
            for key, value in obj.__dict__.items()
            if key not in current_exclude
        }
    elif isinstance(obj, list):
        return [dataclass_obj_to_dict(item, exclude, parent_key) for item in obj]
    elif isinstance(obj, dict):

        return {
            key: dataclass_obj_to_dict(value, exclude, parent_key)
            for key, value in obj.items()
        }
    else:

        return obj


def dataclass_obj_from_dict(cls: Type[Any], data: Dict[str, Any]) -> Any:
    r"""Convert a dictionary to a dataclass object.

    Supports nested dataclasses, lists, and dictionaries.

    .. note::
        If any required field is missing, it will raise an error.
        Do not use the dict that has excluded required fields.

    Example:

    .. code-block:: python

       from dataclasses import dataclass
       from typing import List

       @dataclass
       class TrecData:
           question: str
           label: int

       @dataclass
       class TrecDataList:

           data: List[TrecData]
           name: str

       trec_data_dict = {"data": [{"question": "What is the capital of France?", "label": 0}], "name": "trec_data_list"}

       dataclass_obj_from_dict(TrecDataList, trec_data_dict)

       # Output:
       # TrecDataList(data=[TrecData(question='What is the capital of France?', label=0)], name='trec_data_list')

    """
    if hasattr(cls, "__dataclass_fields__"):
        log.debug(f"{cls} is a dataclass.")
        fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        return cls(
            **{
                key: dataclass_obj_from_dict(fieldtypes[key], value)
                for key, value in data.items()
            }
        )
    elif isinstance(data, list):
        restored_data = []
        for item in data:
            if cls.__args__[0] and hasattr(cls.__args__[0], "__dataclass_fields__"):
                # restore the value to its dataclass type
                restored_data.append(dataclass_obj_from_dict(cls.__args__[0], item))
            else:
                # Use the original data [Any]
                restored_data.append(item)

        return restored_data

    elif isinstance(data, dict):
        for key, value in data.items():
            if cls.__args__[1] and hasattr(cls.__args__[1], "__dataclass_fields__"):
                # restore the value to its dataclass type
                data[key] = dataclass_obj_from_dict(cls.__args__[1], value)
            else:
                # Use the original data [Any]
                data[key] = value
        return data

    else:
        log.debug(f"Not datclass, or list, or dict: {cls}, use the original data.")
        return data


def from_dict_to_json(data: Dict[str, Any], sort_keys: bool = False) -> str:
    r"""Convert a dictionary to a JSON string."""
    try:
        return json.dumps(data, indent=4, sort_keys=sort_keys)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to convert dict to JSON: {e}")


def from_dict_to_yaml(data: Dict[str, Any], sort_keys: bool = False) -> str:
    r"""Convert a dictionary to a YAML string."""
    try:
        return yaml.dump(data, default_flow_style=False, sort_keys=sort_keys)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to convert dict to YAML: {e}")


def from_json_to_dict(json_str: str) -> Dict[str, Any]:
    r"""Convert a JSON string to a dictionary."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to convert JSON to dict: {e}")


def from_yaml_to_dict(yaml_str: str) -> Dict[str, Any]:
    r"""Convert a YAML string to a dictionary."""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to convert YAML to dict: {e}")


########################################################################################
# For ** component
########################################################################################
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


########################################################################################
# For Tokenizer component
########################################################################################
VECTOR_TYPE = Union[List[float], np.ndarray]


def is_normalized(v: VECTOR_TYPE, tol=1e-4) -> bool:
    if isinstance(v, list):
        v = np.array(v)
    # Compute the norm of the vector (assuming v is 1D)
    norm = np.linalg.norm(v)
    # Check if the norm is approximately 1
    return np.abs(norm - 1) < tol


def normalize_np_array(v: np.ndarray) -> np.ndarray:
    # Compute the norm of the vector (assuming v is 1D)
    norm = np.linalg.norm(v)
    # Normalize the vector
    normalized_v = v / norm
    # Return the normalized vector
    return normalized_v


def normalize_vector(v: VECTOR_TYPE) -> List[float]:
    if isinstance(v, list):
        v = np.array(v)
    # Compute the norm of the vector (assuming v is 1D)
    norm = np.linalg.norm(v)
    # Normalize the vector
    normalized_v = v / norm
    # Return the normalized vector as a list
    return normalized_v.tolist()


def get_top_k_indices_scores(
    scores: Union[List[float], np.ndarray], top_k: int
) -> Tuple[List[int], List[float]]:
    if isinstance(scores, list):
        scores_np = np.array(scores)
    else:
        scores_np = scores
    top_k_indices = np.argsort(scores_np)[-top_k:][::-1]
    top_k_scores = scores_np[top_k_indices]
    return top_k_indices.tolist(), top_k_scores.tolist()


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
            yaml_str = text.strip()
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
    except json.JSONDecodeError:
        # 2nd attemp after fixing the json string
        try:
            print("Trying to fix potential missing commas...")
            json_str = fix_json_missing_commas(json_str)
            print("Trying to fix scaped single quotes...")
            json_str = fix_json_escaped_single_quotes(json_str)
            print(f"Fixed JSON string: {json_str}")
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
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
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Got invalid JSON object. Error: {e}. Got JSON string: {json_str}"
                )
            except NameError as exc:
                raise ImportError("Please pip install PyYAML.") from exc
