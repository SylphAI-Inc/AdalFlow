Parser
=============

Parser is the `interpreter` of the LLM output.



Context
----------------

LLMs output text in string format.
Parsing is the process of `extracting` and `converting` the string to desired data structure per the use case.
This desired data structure can be:

- simple data types like string, int, float, boolean, etc.
- complex data types like list, dict, or data class instance.
- Code like Python, SQL, html, etc.

It honestly can be converted to any kind of formats that are required by the use case.
It is an important step for the LLM applications to interact with the external world, such as:

- to int to support classification and float to support regression.
- to list to support multiple choice selection.
- to json/yaml  which will be extracted to dict, and optional further to data class instance to support support cases like function calls.


Scope and Design
------------------

*Right now, we aim to cover the simple and complext data types but the code.*

**Parse**

The following list the scope of our current support of parsing:

.. code-block:: python

    int_str = "42"
    float_str = "42.0"
    boolean_str = "True"  # json works with true/false, yaml works for both True/False and true/false
    None_str = "None"
    Null_str = "null"  # json works with null, yaml works with both null and None
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

In Python, there are various ways to parse the string:
Use built-in functions like ``int``, ``float``, ``bool`` can handle the simple types.
We can use ``ast.literal_eval`` and ``json.loads()`` to handle the complex types like dict, list, and nested dict.
However, none of them is as robust as ``yaml.safe_load``. Yaml can:

- Parse `True/False` and 'true/false' to boolean.
- Parse `None` and 'null' to None.
- Handle nested dict and list in both yaml and json format.

Thus, we will use ``yaml.safe_load`` as the last resort for robust parsing to handle complex data types to get `List` and `Dict` data types.
We will use `int`, `float`, `bool` for simple data types.

Parser
~~~~~~~~~~~~~~

Our parser is located at :doc:`core.string_parser`.
It handles both `extracting` and `parsing` to python object types.
And it is designed to be robust.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parser Class
     - Target Python Object
     - Description
   * - :class:`BooleanParser<core.string_parser.BooleanParser>`
     - ``bool``
     - Extracts the first boolean value from the text with ``bool``. Supports both 'True/False' and 'true/false'.
   * - :class:`IntParser<core.string_parser.IntParser>`
     - ``int``
     - Extracts the first integer value from the text with ``int``.
   * - :class:`FloatParser<core.string_parser.FloatParser>`
     - ``float``
     - Extracts the first float value from the text with ``float``.
   * - :class:`ListParser<core.string_parser.ListParser>`
     - ``list``
     - Extracts '[]' and parses the first list string from the text. Uses both `json.loads` and `yaml.safe_load`.
   * - :class:`JsonParser<core.string_parser.JsonParser>`
     - ``dict``
     - Extracts '[]' and '{}' and parses JSON strings from the text. It resorts to `yaml.safe_load` for robust parsing.
   * - :class:`YamlParser<core.string_parser.YamlParser>`
     - ``dict``
     - Extracts '```yaml```', '```yml```' or the whole string and parses YAML strings from the text.


.. .. list-table:: Parser Classes
..    :header-rows: 1
..    :widths: 25 75

..    * - Parser Class
..      - Description
..    * - :class:`BooleanParser<core.string_parser.BooleanParser>`
..      - Extracts the first boolean value from the text with ``bool``. Supports both 'True/False' and 'true/false'.
..    * - :class:`IntParser<core.string_parser.IntParser>`
..      - Extracts the first integer value from the text with ``int``.
..    * - :class:`FloatParser<core.string_parser.FloatParser>`
..      - Extracts the first float value from the text with ``float``.
..    * - :class:`ListParser<core.string_parser.ListParser>`
..      - Extracts and parses the first list string from the text. Uses both `json.loads` and `yaml.safe_load`. Use this for ``list`` object type.
..    * - :class:`JsonParser<core.string_parser.JsonParser>`
..      - Extracts and parses JSON strings from the text. It resorts to `yaml.safe_load` for robust parsing. Use this for ``dict`` object type.
..    * - :class:`YamlParser<core.string_parser.YamlParser>`
..      - Extracts and parses YAML strings from the text. Use this for ``dict`` object type.



**Data Class Instance**

If your parsed object is dictionary, you can define and use ``DataClass`` instance.
With ``from_dict`` method, you can easily convert the dictionary to data class instance.

.. Converting string to structured data is similar to the step of deserialization in serialization-deserialization process.
.. We already have powerful ``DataClass`` to handle the serialization-deserialization for data class instance.
Output Parsers
~~~~~~~~~~~~~~~~~~~~

The above parsers do not come with output format instructions.
Thus, we created :class:`OutputParser<components.output_parsers.outputs.OutputParser>` to orchestrate both the formatting and parsing process.
It is an abstract component with two main methods:

- ``format_instructions``: to generate the output format instructions for the prompt.
- ``call``: to parse the output string to the desired python object.

If you are targetting at ``dict`` object, we already have ``DataClass`` to help us describe any data class type and instance that can be easily used to interact with LLMs.
Thus, ``JsonOutputParser`` and ``YamlOutputParser`` both takes the following arguments:

- ``data_class``: the ``DataClass`` type.
- ``examples``: the examples of the data class instance if you want to show the examples in the prompt.
- ``exclude``: the fields to exclude from both the data format and the examples.

.. TODO: a summary table and a diagram

Parser in Action
------------------
All of the parsers are quite straightforward to use.

**BooleanParser**

.. code-block:: python

    from adalflow.core.string_parser import BooleanParser

    bool_str = "True"
    bool_str_2 = "False"
    bool_str_3 = "true"
    bool_str_4 = "false"
    bool_str_5 = "1"  # will fail
    bool_str_6 = "0"  # will fail
    bool_str_7 = "yes"  # will fail
    bool_str_8 = "no"  # will fail

    # it will all return True/False
    parser = BooleanParser()
    print(parser(bool_str))
    print(parser(bool_str_2))
    print(parser(bool_str_3))
    print(parser(bool_str_4))

The printout will be:

.. code-block::

    True
    False
    True
    False

Boolean parsers will not work for '1', '0', 'yes', 'no' as they are not the standard boolean values.

**IntParser**

.. code-block:: python

    rom adalflow.core.string_parser import IntParser

    int_str = "42"
    int_str_2 = "42.0"
    int_str_3 = "42.7"
    int_str_4 = "the answer is 42.75"

    # it will all return 42
    parser = IntParser()
    print(parser(int_str))
    print(parser(int_str_2))
    print(parser(int_str_3))
    print(parser(int_str_4))

The printout will be:

.. code-block::

    42
    42
    42
    42

``IntParser`` will return the integer value of the first number in the string, even if it is a float.

**FloatParser**

.. code-block:: python

    from adalflow.core.string_parser import FloatParser

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

The printout will be:

.. code-block::

    42.0
    42.0
    42.7
    42.75


``FloatParser`` will return the float value of the first number in the string, even if it is an integer.

**ListParser**

.. code-block:: python

    from adalflow.core.string_parser import ListParser

    list_str = '["key", "value"]'
    list_str_2 = 'prefix["key", 2]...'
    list_str_3 = '[{"key": "value"}, {"key": "value"}]'

    parser = ListParser()
    print(parser(list_str))
    print(parser(list_str_2))
    print(parser(list_str_3))

The output will be:

.. code-block:: python

    ['key', 'value']
    ['key', 2]
    [{'key': 'value'}, {'key': 'value'}]

**JsonParser**

Even though it can work on lists, it is better to only use it for dictionaries.

.. code-block:: python

    from adalflow.core.string_parser import JsonParser

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

The output will be:

.. code-block:: python

    {'key': 'value'}
    {'name': 'John', 'age': 30, 'attributes': {'height': 180, 'weight': 70}}
    ['key', 2]
    [{'key': 'value'}, {'key': 'value'}]

**YamlParser**

Though it works almost on all of the previous examples, it is better to use it for yaml formatted dictionaries.

.. code-block:: python

    from adalflow.core.string_parser import YamlParser

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

The output will be:

.. code-block:: python

    {'key': 'value'}
    {'name': 'John', 'age': 30, 'attributes': {'height': 180, 'weight': 70}}
    ['key', 'value']

.. note::
    All parsers will raise ``ValueError`` if it fails at any step. Developers should process it accordingly.

Output Parsers in Action
--------------------------


We will create the following simple ``DataClass`` with one example.
And we will demonstrate how to use ``JsonOutputParser`` and ``YamlOutputParser`` to parse another example to dict object.

.. code-block:: python

    from dataclasses import dataclass, field
    from adalflow.core import DataClass

    @dataclass
    class User(DataClass):
        id: int = field(default=1, metadata={"description": "User ID"})
        name: str = field(default="John", metadata={"description": "User name"})

    user_example = User(id=1, name="John")

**JsonOutputParser**

Here is how to use ``JsonOutputParser``:

.. code-block:: python

    from adalflow.components.output_parsers import JsonOutputParser

    parser = JsonOutputParser(data_class=User, examples=[user_example])
    print(parser)

The structure of it:

.. code-block::

    JsonOutputParser(
        data_class=User, examples=[json_output_parser.<locals>.User(id=1, name='John')], exclude_fields=None
        (json_output_format_prompt): Prompt(
            template: Your output should be formatted as a standard JSON instance with the following schema:
            ```
            {{schema}}
            ```
            {% if example %}
            Examples:
            ```
            {{example}}
            ```
            {% endif %}
            -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
            -Use double quotes for the keys and string values.
            -Follow the JSON formatting conventions., prompt_variables: ['example', 'schema']
        )
        (output_processors): JsonParser()
    )

The output format string will be:

.. code-block::

    Your output should be formatted as a standard JSON instance with the following schema:
    ```
    {
        "id": " (int) (optional)",
        "name": " (str) (optional)"
    }
    ```
    Examples:
    ```
    {
        "id": 1,
        "name": "John"
    }
    ________
    ```
    -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
    -Use double quotes for the keys and string values.
    -Follow the JSON formatting conventions.

Call the parser with the following string:

.. code-block:: python

    user_to_parse = '{"id": 2, "name": "Jane"}'
    parsed_user = parser(user_to_parse)
    print(parsed_user)

The output will be:

.. code-block:: python

    {'id': 2, 'name': 'Jane'}

**YamlOutputParser**

The steps are totally the same as the ``JsonOutputParser``.

.. code-block:: python

    from adalflow.components.output_parsers import YamlOutputParser

    parser = YamlOutputParser(data_class=User, examples=[user_example])
    print(parser)

The structure of it:

.. code-block::

    YamlOutputParser(
    data_class=<class '__main__.yaml_output_parser.<locals>.User'>, examples=[yaml_output_parser.<locals>.User(id=1, name='John')]
    (yaml_output_format_prompt): Prompt(
        template: Your output should be formatted as a standard YAML instance with the following schema:
        ```
        {{schema}}
        ```
        {% if example %}
        Examples:
        ```
        {{example}}
        ```
        {% endif %}

        -Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
        -Follow the YAML formatting conventions with an indent of 2 spaces.
        -Quote the string values properly., prompt_variables: ['schema', 'example']
    )
    (output_processors): YamlParser()
    )

The output format string will be:

.. code-block::

    Your output should be formatted as a standard YAML instance with the following schema:
    ```
    id:  (int) (optional)
    name:  (str) (optional)
    ```
    Examples:
    ```
    id: 1
    name: John

    ________
    ```

    -Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
    -Follow the YAML formatting conventions with an indent of 2 spaces.
    -Quote the string values properly.

Now, let us parse the following string:

.. code-block:: python

    user_to_parse = "id: 2\nname: Jane"
    parsed_user = parser(user_to_parse)
    print(parsed_user)

The output will be:

.. code-block:: python

    {'id': 2, 'name': 'Jane'}
.. # todo
.. Evaluate Format following
.. --------------------------

.. .. admonition:: References
..    :class: highlight

..    .. [1] Jinja2: https://jinja.palletsprojects.com/en/3.1.x/
..    .. [2] Llama3 special tokens: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/


.. admonition:: API References
   :class: highlight

   - :ref:`string_parser<core-string_parser>`
   - :ref:`OutputParser<components-output_parsers>`
   - :class:`components.output_parsers.outputs.JsonOutputParser`
   - :class:`components.output_parsers.outputs.YamlOutputParser`
   - :class:`components.output_parsers.outputs.OutputParser`
   - :class:`components.output_parsers.outputs.BooleanOutputParser`
   - :class:`components.output_parsers.outputs.ListOutputParser`
