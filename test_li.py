# import llama_index.core.output_parsers
# from llama_index.core.agent import ReActAgent
from langchain.agents import AgentExecutor, create_react_agent

import ast
from typing import Dict, Any, List, Callable, Awaitable, Union, Optional


class ChildClass:
    def __init__(self, tools: List[str] = []):
        self.tools = tools

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        output = self.tools.__str__()
        print(f"output: {output}")
        return output


def function_name(arg1: int, arg2, arg3=None, arg4=None):
    assert isinstance(arg1, int)
    print(f"arg1: {arg1}, arg2: {arg2}, arg3: {arg3}, arg4: {arg4}")
    return f"arg1: {arg1}, arg2: {arg2}, arg3: {arg3}, arg4: {arg4}"
    return


def fun_with_list_dict(arg1: Dict[str, Any], arg2: List[int]):
    print(f"arg1: {arg1}, arg2: {arg2}")


def fun_with_class_instance(arg1: ChildClass):
    tmp = arg1()
    print(f"tmp: {tmp}")
    return tmp


def fun_with_function_instance(arg1: Any):

    print(f"fun_with_function_instance: {arg1}")
    return arg1


def evaluate_ast_node(node, context_map: Dict[str, Any] = None):
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


def parse_function_call(call_string: str, context_map: Dict[str, Any] = None):
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


context_map = {
    "function_name": function_name,
    "fun_with_list_dict": fun_with_list_dict,
    "fun_with_class_instance": fun_with_class_instance,
    "ChildClass": ChildClass,
    "fun_with_function_instance": fun_with_function_instance,
}
# Example Usage
call_string = "function_name(1, 42, arg3={'key': 'value'}, arg4=[1, 2, 3])"
parsed_call = parse_function_call(call_string)
print(parsed_call)
# print(f"""parsed_call["function_name"]: {parsed_call["function_name"]}""")
# print(f"""parsed_call["args"]: {parsed_call["args"]}""")
# print(f"""parsed_call["keywords"]: {parsed_call["keywords"]}""")
func_name, args, keywords = parse_function_call(call_string)

# Dynamically call the function with parsed arguments
print(f"func_name: {func_name}, args: {args}, keywords: {keywords}")
# function = globals().get(func_name)
function = context_map.get(func_name)


if function:
    function(*args, **keywords)
else:
    print("Function not found.")

# test with class
call_string = "fun_with_class_instance(ChildClass(['tool1', 'tool2']))"
parsed_call = parse_function_call(call_string, context_map)
print(parsed_call)

func_name, args, keywords = parse_function_call(call_string, context_map)
if func_name:
    function = context_map.get(func_name)
    if function:
        function(*args, **keywords)
    else:
        print("Function not found.")

# test with function instance
call_string = "fun_with_function_instance(function_name(1, 42, arg3={'key': 'value'}, arg4=[1, 2, 3]))"
parsed_call = parse_function_call(call_string, context_map)
print(parsed_call)
func_name, args, keywords = parse_function_call(call_string, context_map)
print(f"func_name: {func_name}, args: {args}, keywords: {keywords}")
if func_name:
    function = context_map.get(func_name)
    if function:
        function(*args, **keywords)
    else:

        print("Function not found.")


# call fun with only positional arguments
call_string = "function_name(1, 42)"
parsed_call = parse_function_call(call_string, context_map)
print(parsed_call)
func_name, args, keywords = parse_function_call(call_string, context_map)
if func_name:
    function = context_map.get(func_name)
    if function:
        function(*args, **keywords)
    else:
        print("Function not found.")
