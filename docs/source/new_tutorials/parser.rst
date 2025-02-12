.. _components-output_parser_note:

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_dataclasses.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/parser_note.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>

   </div>

Parser and Structured Output
==============================

Parser is the `interpreter` of the LLM output.

.. We have three types of parsers:

.. - **String Parsers**: it simply converts the string to the desired data type. They are located at :ref:`core.string_parser<core-string_parser>`.
.. - **Output Parsers**: it orchestrates the parsing and output formatting(in yaml, json and more) process. They are located at :ref:`components.output_parsers.outputs<components-output_parsers-outputs>`. :class:`JsonOutputParser` and :class:`YamlOutputParser` can work with :ref:`DataClass<core-dataclass>` for structured output.
.. - **DataClass Parser**: On top of :class:`YamlOutputParser<components.output_parsers.outputs.YamlOutputParser>` and :class:`JsonOutputParser<components.output_parsers.outputs.JsonOutputParser>`, :class:`DataClassParser<components.output_parsers.dataclass_parser.DataClassParser>` is the most compatible to work with :ref:`DataClass<core-dataclass>` for structured output.

Basic Parser
~~~~~~~~~~~~~~
For basic data formats where you do not need to create a data class for, you can use the following Parsers in the library.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parser Class
     - Target Python Object
     - Description
   * - :class:`BooleanParser<core.string_parser.BooleanParser>`
     - ``bool``
     - Extracts the first boolean value from the text as ``bool``. Supports both 'True/False' and 'true/false'.
   * - :class:`IntParser<core.string_parser.IntParser>`
     - ``int``
     - Extracts the first integer value from the text as ``int``.
   * - :class:`FloatParser<core.string_parser.FloatParser>`
     - ``float``
     - Extracts the first float value from the text as ``float``.
   * - :class:`ListParser<core.string_parser.ListParser>`
     - ``list``
     - Extracts '[]' and parses the first list string from the text. Uses both `json.loads` and `yaml.safe_load`.
   * - :class:`JsonParser<core.string_parser.JsonParser>`
     - ``dict``
     - Extracts '[]' and '{}' and parses JSON strings from the text. It resorts to `yaml.safe_load` for robust parsing.
   * - :class:`YamlParser<core.string_parser.YamlParser>`
     - ``dict``
     - Extracts '```yaml```', '```yml```' or the whole string and parses YAML strings from the text.


Here are some quick demonstrations:

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


DataClassParser
~~~~~~~~~~~~~~~~~~~~~~~
For more complicated data structures, we can use :class:`DataClass<core.base_data_class.DataClass>` to define it.
The usage of it is pretty much the same as native `dataclass` from `dataclasses`.

Let's try to define a User class:

.. code-block:: python

    from dataclasses import dataclass, field
    from adalflow.core import DataClass

    # no need to use Optional, when default is on, it is optional.
    .. code-block:: python

    @dataclass
    class SampleDataClass(DataClass):
        description: str = field(metadata={"desc": "A sample description"})
        category: str = field(metadata={"desc": "Category of the sample"})
        value: int = field(metadata={"desc": "A sample integer value"})
        status: str = field(metadata={"desc": "Status of the sample"})

        # input and output fields can work with DataClassParser
        __input_fields__ = [
            "description",
            "category",
        ]
        __output_fields__ = ["value", "status"]


We have three classes to work with structured data.
They are :class:`DataClassParser<components.output_parsers.outputs.DataClassParser>`,
:class:`JsonOutputParser<components.output_parsers.outputs.JsonOutputParser>`, and `YamlOutputParser<components.output_parsers.outputs.YamlOutputParser>`.
`DataClassParser` is the easiest to use.

Now, lets' create a parser that will use the `SampleDataClass` to parse the output json string back to the data class instance.

.. code-block:: python

    from adalflow.components.output_parsers import DataClassParser

    parser = DataClassParser(data_class=SampleDataClass, return_data_class=True, format_type="json")

Let's view the structure of the parser use `print(parser)`.

The output will be:

.. code-block::

    DataClassParser(
        data_class=SampleDataClass, format_type=json,            return_data_class=True, input_fields=['description', 'category'],            output_fields=['value', 'status']
        (_output_processor): JsonParser()
        (output_format_prompt): Prompt(
            template: Your output should be formatted as a standard JSON instance with the following schema:
            ```
            {{schema}}
            ```
            -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
            -Use double quotes for the keys and string values.
            -DO NOT mistaken the "properties" and "type" in the schema as the actual fields in the JSON output.
            -Follow the JSON formatting conventions., prompt_variables: ['schema']
        )
    )

You can get the output and input format strings using the following methods:

.. code-block:: python

    print(parser.get_input_format_str())
    print(parser.get_output_format_str())

The output for the output format string will be:

.. code-block::

    Your output should be formatted as a standard JSON instance with the following schema:
    ```
    {
        "value": " (int) (required)",
        "status": " (str) (required)"
    }
    ```
    -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
    -Use double quotes for the keys and string values.
    -DO NOT mistaken the "properties" and "type" in the schema as the actual fields in the JSON output.
    -Follow the JSON formatting conventions.

The input format string will be:

.. code-block::

    {
        "description": " (str) (required)",
        "category": " (str) (required)"
    }

Convert a json string to a data class instance:

.. code-block:: python

    user_input = '{"description": "Parsed description", "category": "Sample Category", "value": 100, "status": "active"}'
    parsed_instance = parser.call(user_input)

    print(parsed_instance)

The output will be:

.. code-block:: python

    SampleDataClass(description='Parsed description', category='Sample Category', value=100, status='active')

Try the examples string:

.. code-block:: python

    samples = [
        SampleDataClass(
            description="Sample description",
            category="Sample category",
            value=100,
            status="active",
        ),
        SampleDataClass(
            description="Another description",
            category="Another category",
            value=200,
            status="inactive",
        ),
    ]

    examples_str = parser.get_examples_str(examples=samples)
    print(examples_str)

The output will be:

.. code-block:: python

    examples_str:
    {
        "description": "Sample description",
        "category": "Sample category",
        "value": 100,
        "status": "active"
    }
    __________
    {
        "description": "Another description",
        "category": "Another category",
        "value": 200,
        "status": "inactive"
    }
    __________

You can check out :ref:`Deep Dive Parser <components-output_parser_note>` for more.




.. admonition:: API References
   :class: highlight

   - :ref:`string_parser<core-string_parser>`
   - :ref:`OutputParser<components-output_parsers>`
   - :class:`components.output_parsers.outputs.JsonOutputParser`
   - :class:`components.output_parsers.outputs.YamlOutputParser`
   - :class:`components.output_parsers.outputs.OutputParser`
   - :class:`components.output_parsers.outputs.BooleanOutputParser`
   - :class:`components.output_parsers.outputs.ListOutputParser`
   - :class:`components.output_parsers.dataclass_parser.DataClassParser`
   - :class:`core.base_data_class.DataClass`
