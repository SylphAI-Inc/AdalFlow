def examples_of_different_ways_to_parse_string():

    int_str = "42"
    float_str = "42.0"
    boolean_str = "True"  # json works with true/false
    None_str = "None"
    Null_str = "null"  # json works with null
    dict_str = '{"key": "value"}'
    list_str = '["key", "value"]'
    nested_dict_str = (
        '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}}'
    )
    yaml_dict_str = "key: value"
    yaml_nested_dict_str = (
        "name: John\nage: 30\nattributes:\n  height: 180\n  weight: 70"
    )
    yaml_list_str = "- key\n- value"

    # string to int/float/bool
    print("built-in parser:\n____________________")
    print(int(int_str))
    print(float(float_str))
    print(bool(boolean_str))

    # via json loads
    import json

    print("\njson parser:\n____________________")
    json_int = json.loads(int_str)
    json_float = json.loads(float_str)
    json_bool = json.loads(
        boolean_str.lower()
    )  # json.loads only accepts true or false, not True or False
    json_none = json.loads(Null_str)
    json_dict = json.loads(dict_str)
    json_list = json.loads(list_str)
    json_nested_dict = json.loads(nested_dict_str)
    # json_yaml_dict = json.loads(yaml_dict_str) # wont work
    # json_yaml_nested_dict = json.loads(yaml_nested_dict_str)
    # json_yaml_list = json.loads(yaml_list_str)

    print(int_str, type(json_int), json_int)
    print(float_str, type(json_float), json_float)
    print(boolean_str, type(json_bool), json_bool)
    print(None_str, type(json_none), json_none)
    print(dict_str, type(json_dict), json_dict)
    print(list_str, type(json_list), json_list)
    print(nested_dict_str, type(json_nested_dict), json_nested_dict)

    # via yaml
    import yaml

    print("\nyaml parser:\n____________________")

    yaml_int = yaml.safe_load(int_str)
    yaml_float = yaml.safe_load(float_str)
    yaml_bool = yaml.safe_load(boolean_str)
    yaml_bool_lower = yaml.safe_load(boolean_str.lower())
    yaml_null = yaml.safe_load(Null_str)
    yaml_none = yaml.safe_load(None_str)

    yaml_dict = yaml.safe_load(dict_str)
    yaml_list = yaml.safe_load(list_str)
    yaml_nested_dict = yaml.safe_load(nested_dict_str)
    yaml_yaml_dict = yaml.safe_load(yaml_dict_str)
    yaml_yaml_nested_dict = yaml.safe_load(yaml_nested_dict_str)
    yaml_yaml_list = yaml.safe_load(yaml_list_str)

    print(int_str, type(yaml_int), yaml_int)
    print(float_str, type(yaml_float), yaml_float)
    print(boolean_str, type(yaml_bool), yaml_bool)
    print(boolean_str.lower(), type(yaml_bool_lower), yaml_bool_lower)
    print(Null_str, type(yaml_null), yaml_null)
    print(None_str, type(yaml_none), yaml_none)
    print(dict_str, type(yaml_dict), yaml_dict)
    print(list_str, type(yaml_list), yaml_list)
    print(nested_dict_str, type(yaml_nested_dict), yaml_nested_dict)
    print(yaml_dict_str, type(yaml_yaml_dict), yaml_yaml_dict)
    print(yaml_nested_dict_str, type(yaml_yaml_nested_dict), yaml_yaml_nested_dict)
    print(yaml_list_str, type(yaml_yaml_list), yaml_yaml_list)

    # via ast for python literal
    import ast

    print("\nast parser:\n____________________\n")

    ast_int = ast.literal_eval(int_str)
    ast_float = ast.literal_eval(float_str)
    ast_bool = ast.literal_eval(boolean_str)
    ast_none = ast.literal_eval(None_str)
    ast_dict = ast.literal_eval(dict_str)
    ast_list = ast.literal_eval(list_str)
    ast_nested_dict = ast.literal_eval(nested_dict_str)

    print(int_str, type(ast_int), ast_int)
    print(float_str, type(ast_float), ast_float)
    print(boolean_str, type(ast_bool), ast_bool)
    print(None_str, type(ast_none), ast_none)
    print(dict_str, type(ast_dict), ast_dict)
    print(list_str, type(ast_list), ast_list)
    print(nested_dict_str, type(ast_nested_dict), ast_nested_dict)

    # via eval for any python expression, but not recommended for security reasons

    print("\n eval parser:\n____________________\n")

    eval_int = eval(int_str)
    eval_float = eval(float_str)
    eval_bool = eval(boolean_str)
    eval_dict = eval(dict_str)
    eval_list = eval(list_str)
    eval_nested = eval(nested_dict_str)
    # eval_yaml_dict = eval(yaml_dict_str) # wont work

    print(int_str, type(eval_int), eval_int)
    print(float_str, type(eval_float), eval_float)
    print(boolean_str, type(eval_bool), eval_bool)
    print(dict_str, type(eval_dict), eval_dict)
    print(list_str, type(eval_list), eval_list)
    print(nested_dict_str, type(eval_nested), eval_nested)


def int_parser():
    from lightrag.core.string_parser import IntParser

    int_str = "42"
    int_str_2 = "42.0"
    int_str_3 = "42.7"
    int_str_4 = "the answer is 42.75"

    # it will all return 42
    parser = IntParser()
    print(parser)
    print(parser(int_str))
    print(parser(int_str_2))
    print(parser(int_str_3))
    print(parser(int_str_4))


def float_parser():
    from lightrag.core.string_parser import FloatParser

    float_str = "42.0"
    float_str_2 = "42"
    float_str_3 = "42.7"
    float_str_4 = "the answer is 42.75"

    # it will all return 42.0
    parser = FloatParser()
    print(parser(float_str))
    print(parser(float_str_2))
    print(parser(float_str_3))
    print(parser(float_str_4))


def bool_parser():
    from lightrag.core.string_parser import BooleanParser

    bool_str = "True"
    bool_str_2 = "False"
    bool_str_3 = "true"
    bool_str_4 = "false"
    # bool_str_5 = "1"  # will fail
    # bool_str_6 = "0"  # will fail
    # bool_str_7 = "yes"  # will fail
    # bool_str_8 = "no"  # will fail

    # it will all return True/False
    parser = BooleanParser()
    print(parser(bool_str))
    print(parser(bool_str_2))
    print(parser(bool_str_3))
    print(parser(bool_str_4))
    # print(parser(bool_str_5))
    # print(parser(bool_str_6))
    # print(parser(bool_str_7))
    # print(parser(bool_str_8))


def list_parser():

    from lightrag.core.string_parser import ListParser

    list_str = '["key", "value"]'
    list_str_2 = 'prefix["key", 2]...'
    list_str_3 = '[{"key": "value"}, {"key": "value"}]'
    # dict_str = '{"key": "value"}'

    parser = ListParser()
    print(parser(list_str))
    print(parser(list_str_2))
    print(parser(list_str_3))
    # print(parser(dict_str)) # will raise ValueError


def json_parser():
    from lightrag.core.string_parser import JsonParser

    dict_str = '{"key": "value"}'
    nested_dict_str = (
        '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}}'
    )
    list_str = '["key", 2]'
    list_dict_str = '[{"key": "value"}, {"key": "value"}]'

    parser = JsonParser()
    print(parser)
    print(parser(dict_str))
    print(parser(nested_dict_str))
    print(parser(list_str))
    print(parser(list_dict_str))


def yaml_parser():
    from lightrag.core.string_parser import YamlParser

    yaml_dict_str = "key: value"
    yaml_nested_dict_str = (
        "name: John\nage: 30\nattributes:\n  height: 180\n  weight: 70"
    )
    yaml_list_str = "- key\n- value"

    parser = YamlParser()
    print(parser)
    print(parser(yaml_dict_str))
    print(parser(yaml_nested_dict_str))
    print(parser(yaml_list_str))


def json_output_parser():
    from dataclasses import dataclass, field
    from lightrag.components.output_parsers import JsonOutputParser
    from lightrag.core import DataClass

    @dataclass
    class User(DataClass):
        id: int = field(default=1, metadata={"description": "User ID"})
        name: str = field(default="John", metadata={"description": "User name"})

    user_example = User(id=1, name="John")

    user_to_parse = '{"id": 2, "name": "Jane"}'

    parser = JsonOutputParser(data_class=User, examples=[user_example])
    print(parser)
    output_format_str = parser.format_instructions()
    print(output_format_str)
    parsed_user = parser(user_to_parse)
    print(parsed_user)


def yaml_output_parser():
    from dataclasses import dataclass, field
    from lightrag.components.output_parsers import YamlOutputParser
    from lightrag.core import DataClass

    @dataclass
    class User(DataClass):
        id: int = field(default=1, metadata={"description": "User ID"})
        name: str = field(default="John", metadata={"description": "User name"})

    user_example = User(id=1, name="John")

    user_to_parse = "id: 2\nname: Jane"

    parser = YamlOutputParser(data_class=User, examples=[user_example])
    print(parser)
    output_format_str = parser.format_instructions()
    print(output_format_str)
    parsed_user = parser(user_to_parse)
    print(parsed_user)


if __name__ == "__main__":
    examples_of_different_ways_to_parse_string()
    int_parser()
    float_parser()
    bool_parser()
    list_parser()
    json_parser()
    yaml_parser()
    json_output_parser()
    yaml_output_parser()
