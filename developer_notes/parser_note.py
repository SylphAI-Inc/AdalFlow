def examples_of_different_ways_to_parse_string():
    # string to int/float
    print(int("42"))
    print(float("42.0"))

    # via json loads
    import json

    print(json.loads('{"key": "value"}'))

    # a more complicated case
    print(
        json.loads(
            '{"name": "John", "age": 30, "attributes": {"height": 180, "weight": 70}}'
        )
    )

    # json load for list
    print(json.loads('["key", "value"]'))

    # via yaml
    import yaml

    print(yaml.safe_load("key: value"))
    print(
        yaml.safe_load("name: John\nage: 30\nattributes:\n  height: 180\n  weight: 70")
    )
    print(yaml.safe_load("['key', 'value']"))

    # via ast for python literal
    import ast

    print(ast.literal_eval("42"))
    print(ast.literal_eval("{'key': 'value'}"))
    print(ast.literal_eval("['key', 'value']"))
    # complex case like dict
    print(
        ast.literal_eval(
            "{'name': 'John', 'age': 30, 'attributes': {'height': 180, 'weight': 70}}"
        )
    )

    # via regex

    # via eval for any python expression
    print(eval("42"))
    print(eval("{'key': 'value'}"))
    print(eval("['key', 'value']"))
    # complex case like dict
    print(
        eval("{'name': 'John', 'age': 30, 'attributes': {'height': 180, 'weight': 70}}")
    )

    #


if __name__ == "__main__":
    examples_of_different_ways_to_parse_string()
