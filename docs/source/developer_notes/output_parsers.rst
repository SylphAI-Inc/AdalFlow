Parser
=============

In this note, we will explain LightRAG parser and output parsers.

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

Parsing is the `interpreter` of the LLM output.

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

.. list-table:: Parser Classes
   :header-rows: 1
   :widths: 25 75

   * - Parser Class
     - Description
   * - :class:`BooleanParser<core.string_parser.BooleanParser>`
     - Extracts the first boolean value from the text with ``bool``. Supports both `True/False` and 'true/false'.
   * - :class:`IntParser<core.string_parser.IntParser>`
     - Extracts the first integer value from the text with ``int``.
   * - :class:`FloatParser<core.string_parser.FloatParser>`
     - Extracts the first float value from the text with ``float``.
   * - :class:`ListParser<core.string_parser.ListParser>`
     - Extracts and parses the first list string from the text. Uses both `json.loads` and `yaml.safe_load`.
   * - :class:`JsonParser<core.string_parser.JsonParser>`
     - Extracts and parses JSON strings from the text. It resorts to `yaml.safe_load` for robust parsing.
   * - :class:`YamlParser<core.string_parser.YamlParser>`
     - Extracts and parses YAML strings from the text.



**Data Class Instance**

If your parsed object is dictionary, you can define and use ``DataClass`` instance.
With ``from_dict`` method, you can easily convert the dictionary to data class instance.

.. Converting string to structured data is similar to the step of deserialization in serialization-deserialization process.
.. We already have powerful ``DataClass`` to handle the serialization-deserialization for data class instance.
Output Parsers
~~~~~~~~~~~~~~~~~~~~



Parser in Action
------------------

Parser builts on top of that


Output Parsers in Action
--------------------------

Evaluate Format following
--------------------------

.. admonition:: References
   :class: highlight

   .. [1] Jinja2: https://jinja.palletsprojects.com/en/3.1.x/
   .. [2] Llama3 special tokens: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/


.. admonition:: API References
   :class: highlight

   - :ref:`string_parser<core-string_parser>`
   - :ref:`OutputParser<components-output_parsers>`
