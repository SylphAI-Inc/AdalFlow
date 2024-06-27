import ast
import pytest

from lightrag.core.functional import evaluate_ast_node, parse_function_call_expr

from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: int
    y: int


def add_points(p1: Point, p2: Point) -> Point:
    return Point(p1.x + p2.x, p1.y + p2.y)


def numpy_sum(arr: np.ndarray) -> int:
    return np.sum(arr)


class TestAstEvaluation:
    def setup_method(self):
        def add(a, b: int) -> int:
            return a + b

        self.context_map = {
            "x": 10,
            "y": 5,
            "add": add,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b,
            "Point": Point,
            "add_points": add_points,
            "np": np,  # Adding numpy to the context map
            "array": np.array,
            "sum": np.sum,
            "mean": np.mean,
            "numpy_sum": numpy_sum,
        }

    def test_evaluate_constant(self):
        node = ast.parse("42", mode="eval").body
        assert evaluate_ast_node(node) == 42

    def test_evaluate_dict(self):
        node = ast.parse("{'a': 1, 'b': 2}", mode="eval").body
        assert evaluate_ast_node(node) == {"a": 1, "b": 2}

    def test_evaluate_list(self):
        node = ast.parse("[1, 2, 3]", mode="eval").body
        assert evaluate_ast_node(node) == [1, 2, 3]

    def test_evaluate_tuple(self):
        node = ast.parse("(1, 2, 3)", mode="eval").body
        assert evaluate_ast_node(node) == (1, 2, 3)

    def test_evaluate_unary_op(self):
        node = ast.parse("-10", mode="eval").body
        assert evaluate_ast_node(node) == -10

    def test_evaluate_bin_op(self):
        node = ast.parse("10 + 5", mode="eval").body
        assert evaluate_ast_node(node) == 15

    def test_evaluate_name(self):
        node = ast.parse("x", mode="eval").body
        assert evaluate_ast_node(node, self.context_map) == 10

    def test_evaluate_function_call(self):
        node = ast.parse("add(3, 4)", mode="eval").body
        assert evaluate_ast_node(node, self.context_map) == 7

    def test_unsupported_ast_node(self):
        node = ast.parse("lambda x: x + 1", mode="eval").body
        with pytest.raises(ValueError):
            evaluate_ast_node(node)

    def test_parse_function_call_expr_valid(self):
        func_expr = "add(3, 4)"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "add"
        assert args == [3, 4]
        assert kwargs == {}

    def test_parse_function_call_expr_with_kwargs(self):
        self.context_map["power"] = lambda x, y=2: x**y
        func_expr = "power(3, y=3)"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "power"
        assert args == [3]
        assert kwargs == {"y": 3}

    def test_parse_function_call_expr_invalid(self):
        func_expr = "3 + 4"
        with pytest.raises(ValueError):
            parse_function_call_expr(func_expr, self.context_map)

    def test_evaluate_nested_function_calls(self):
        node = ast.parse("add(multiply(2, 3), 4)", mode="eval").body
        assert evaluate_ast_node(node, self.context_map) == 10

    def test_evaluate_with_variable_replacement(self):
        func_expr = "add(x, y)"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "add"
        assert args == [10, 5]
        assert kwargs == {}
        assert self.context_map["add"](*args, **kwargs) == 15

    def test_evaluate_with_wrong_keyword(self):
        func_expr = "add(x, y=5)"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "add"
        assert args == [10]
        assert kwargs == {"y": 5}
        with pytest.raises(TypeError):
            self.context_map["add"](*args, **kwargs)

    def test_evaluate_with_variable_replacement_and_kwargs(self):
        func_expr = "add(x, b=y)"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "add"
        assert args == [10]
        assert kwargs == {"b": 5}
        assert self.context_map["add"](*args, **kwargs) == 15

    def test_evaluate_with_unknown_variable(self):
        func_expr = "add(x, z)"
        with pytest.raises(ValueError):
            parse_function_call_expr(func_expr, self.context_map)

    def test_evaluate_complex_list(self):
        node = ast.parse("[1, [2, 3], {'a': 4, 'b': [5, 6]}]", mode="eval").body
        result = evaluate_ast_node(node)
        assert result == [1, [2, 3], {"a": 4, "b": [5, 6]}]

    def test_evaluate_complex_dict(self):
        node = ast.parse(
            "{'key1': 1, 'key2': {'subkey1': [1, 2, 3], 'subkey2': {'subsubkey': 4}}}",
            mode="eval",
        ).body
        result = evaluate_ast_node(node)
        assert result == {
            "key1": 1,
            "key2": {"subkey1": [1, 2, 3], "subkey2": {"subsubkey": 4}},
        }

    def test_evaluate_with_dataclass(self):
        func_expr = "add_points(Point(1, 2), Point(3, 4))"
        func_name, args, kwargs = parse_function_call_expr(func_expr, self.context_map)
        assert func_name == "add_points"
        assert args == [Point(1, 2), Point(3, 4)]
        assert kwargs == {}
        result = self.context_map[func_name](*args, **kwargs)
        assert result == Point(4, 6)

    def test_evaluate_numpy_array(self):
        node = ast.parse("array([1, 2, 3, 4])", mode="eval").body
        result = evaluate_ast_node(node, self.context_map)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_evaluate_numpy_sum(self):
        node = ast.parse("sum(array([1, 2, 3, 4]))", mode="eval").body
        result = evaluate_ast_node(node, self.context_map)
        assert result == 10

    def test_evaluate_numpy_mean(self):
        node = ast.parse("mean(array([1, 2, 3, 4]))", mode="eval").body
        result = evaluate_ast_node(node, self.context_map)
        assert result == 2.5

    def test_evaluate_numpy_sum_2d(self):
        node = ast.parse("numpy_sum(arr=np.array([[1, 2], [3, 4]]))", mode="eval").body
        result = evaluate_ast_node(node, self.context_map)
        assert result == 10


if __name__ == "__main__":
    pytest.main()
