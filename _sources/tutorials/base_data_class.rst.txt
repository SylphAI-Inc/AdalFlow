.. _core-base_data_class_note:


.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_dataclasses.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/adalflow_dataclasses.py" target="_blank" style="display: flex; align-items: center; margin-right: 10px;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

DataClass
============

.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_


In LLM applications, data constantly needs to interact with LLMs in the form of strings via prompt and be parsed back to structured data from LLMs' text prediction.
:class:`DataClass<core.base_data_class.DataClass>` is designed to ease this data interaction with LLMs via prompt(input) and to parse the text prediction(output).
It is even more convenient to use together with :doc:`Parser<output_parsers>` to parse the output from LLMs.

.. figure:: /_static/images/dataclass.png
    :align: center
    :alt: DataClass
    :width: 680px

    DataClass is to ease the data interaction with LLMs.


Design
----------------
In Python, data is typically represented as a class with attributes.
To interact with LLM, we need great way to describe the data format and the data instance to LLMs and be able to convert back to data instance from the text prediction.
This overlaps with the serialization and deserialization of the data in the conventional programming.
Packages like ``Pydantic`` or ``Marshmallow`` can covers the seralization and deserialization, but it will end up with more complexity and less transparency to users.
LLM prompts are known to be sensitive, the details, controllability, and transparency of the data format are crucial here.

We eventually created a base class :class:`DataClass<core.base_data_class.DataClass>`  to handle data that will interact with LLMs, which builds on top of Python's native ``dataclasses`` module.
Here are our reasoning:

1. ``dataclasses`` module is lightweight, flexible, and is already widely used in Python for data classes.
2.  Using ``field`` (`metadata`, `default`, `default_factory`) in `dataclasses` adds more ways to describe the data.
3.  ``asdict()`` from `dataclasses` is already good at converting a data class instance to a dictionary for serialization.
4.  Getting data class schmea for data class is feasible.


Here is how users typically use the ``dataclasses`` module:

.. code-block:: python

    from dataclasses import dataclass, field

    @dataclass
    class TrecData:
        question: str = field(
            metadata={"desc": "The question asked by the user"}
        ) # Required field, you have to provide the question field at the instantiation
        label: int = field(
            metadata={"desc": "The label of the question"}, default=0
        ) # Optional field

``DataClass`` covers the following:

1. Generate the class ``schema`` and ``signature`` (less verbose) to describe the data format to LLMs.
2. Convert the data instance to a json or yaml string to show the data example to LLMs.
3. Load the data instance from a json or yaml string to get the data instance back to be processed in the program.

We also made the effort to provide more control:

1. **Keep the ordering of your data fields.** We provided :func:`required_field<core.base_data_class.required_field>` with ``default_factory`` to mark the field as required even if it is after optional fields. We also has to do customization to preserve their ordering while being converted to dictionary, json and yaml string.
2. **Signal the output/input fields.** We allow you to use ``__output_fields__`` and ``__input_fields__`` to explicitly signal the output and input fields. (1) It can be a subset of the fields in the data class. (2) You can specify the ordering in the `__output_fields__`.
3. **Exclude some fields from the output.**  All serialization methods support `exclude` parameter to exclude some fields even for nested dataclasses.
4. **Allow nested dataclasses, lists, and dictionaries.** All methods support nested dataclasses, lists, and dictionaries.
5. **Easy to use with Output parser.**  It works well with output parsers such as ``JsonOutputParser``, ``YamlOutputParser``, and ``DataClassParser``. You can refer to :doc:`Parser<output_parsers>`for more details.


Describing the Data Format (Data Class)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 70

   * - **Name**
     - **Description**
   * - ``__input_fields__``
     - A list of fields that are input fields.
   * - ``__output_fields__``
     - Used more often than ``__input_fields__``. A list of fields that are output fields. (1) It can be a subset of the fields in the data class. (2) You can specify the ordering in the `__output_fields__`. (3) Works well and only with :class:`DataClassParser<core.base_data_class.DataClassParser>`.
   * - ``to_schema(cls, exclude) -> Dict``
     - Generate a JSON schema which is more detailed than the signature.
   * - ``to_schema_str(cls, exclude) -> str``
     - Generate a JSON schema string which is more detailed than the signature.
   * - ``to_yaml_signature(cls, exclude) -> str``
     - Generate a YAML signature for the class from descriptions in metadata.
   * - ``to_json_signature(cls, exclude) -> str``
     - Generate a JSON signature (JSON string) for the class from descriptions in metadata.
   * - ``format_class_str(cls, format_type, exclude) -> str``
     - Generate data format string, covers ``to_schema_str``, ``to_yaml_signature``, and ``to_json_signature``.

Work with Data Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 70

   * - **Name**
     - **Description**
   * - ``from_dict(cls, data: Dict) -> "DataClass"``
     - Create a dataclass instance from a dictionary. Supports nested dataclasses, lists, and dictionaries.
   * - ``to_dict(self, exclude: ExcludeType) -> Dict``
     - Convert a dataclass object to a dictionary. Supports nested dataclasses, lists, and dictionaries. Allows exclusion of specific fields.
   * - ``to_json_obj(self, exclude: ExcludeType) -> Any``
     - Convert the dataclass instance to a JSON object, maintaining the order of fields.
   * - ``to_json(self, exclude: ExcludeType) -> str``
     - Convert the dataclass instance to a JSON string, maintaining the order of fields.
   * - ``to_yaml_obj(self, exclude: ExcludeType) -> Any``
     - Convert the dataclass instance to a YAML object, maintaining the order of fields.
   * - ``to_yaml(self, exclude: ExcludeType) -> str``
     - Convert the dataclass instance to a YAML string, maintaining the order of fields.
   * - ``from_json(cls, json_str: str) -> "DataClass"``
     - Create a dataclass instance from a JSON string.
   * - ``from_yaml(cls, yaml_str: str) -> "DataClass"``
     - Create a dataclass instance from a YAML string.
   * - ``format_example_str(self, format_type, exclude) -> str``
     - Generate data examples string, covers ``to_json`` and ``to_yaml``.

We have :class:`DataclassFormatType<core.base_data_class.DataClassFormatType>` to specify the format type for the data format methods.

.. note::

    To use ``DataClass``, you have to decorate your class with the ``dataclass`` decorator from the ``dataclasses`` module.


DataClass in Action
------------------------
Say you have a few of ``TrecData`` structued as follows that you want to engage with LLMs:

.. code-block:: python

    from dataclasses import dataclass, field

    @dataclass
    class Question:
        question: str = field(
            metadata={"desc": "The question asked by the user"}
        )
        metadata: dict = field(
            metadata={"desc": "The metadata of the question"}, default_factory=dict
        )

    @dataclass
    class TrecData:
        question: Question = field(
            metadata={"desc": "The question asked by the user"}
        ) # Required field, you have to provide the question field at the instantiation
        label: int = field(
            metadata={"desc": "The label of the question"}, default=0
        ) # Optional field

Describe the data format to LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We will create ``TrecData2`` class that subclasses from `DataClass`.
You decide to add a field ``metadata`` to the ``TrecData`` class to store the metadata of the question.
For your own reason, you want ``metadata`` to be a required field and you want to keep the ordering of your fields while being converted to strings.
``DataClass`` will help you achieve this using :func:`required_field<core.base_data_class.required_field>` on the `default_factory` of the field.
Normally, this is not possible with the native `dataclasses` module as it will raise an error if you put a required field after an optional field.

.. note::

    **Order of the fields** matter as in a typical Chain of Thought, we want the reasoning/thought field to be in the output ahead of the answer.

.. code-block:: python

    from adalflow.core import DataClass, required_field

    @dataclass
    class TrecData2(DataClass):
        question: Question = field(
            metadata={"desc": "The question asked by the user"}
        ) # Required field, you have to provide the question field at the instantiation
        label: int = field(
            metadata={"desc": "The label of the question"}, default=0
        ) # Optional field
        metadata: dict = field(
            metadata={"desc": "The metadata of the question"}, default_factory=required_field()
        ) # required field

**Schema**

Now, let us see the schema of the ``TrecData2`` class:

.. code-block:: python

    print(TrecData2.to_schema())

The output will be:

.. code-block::

    {
        "type": "TrecData2",
        "properties": {
            "question": {
                "type": "{'type': 'Question', 'properties': {'question': {'type': 'str', 'desc': 'The question asked by the user'}, 'metadata': {'type': 'dict', 'desc': 'The metadata of the question'}}, 'required': ['question']}",
                "desc": "The question asked by the user",
            },
            "label": {"type": "int", "desc": "The label of the question"},
            "metadata": {"type": "dict", "desc": "The metadata of the question"},
        },
        "required": ["question", "metadata"],
    }

As you can see, it handles the nested dataclass `Question` and the required field `metadata` correctly.



.. note::

    ``Optional`` type hint will not affect the field's required status. We recommend you not to use it in the `dataclasses` module especially when you are nesting many levels of dataclasses. It might end up confusing the LLMs.

**Signature**

As schema can be rather verbose, and sometimes it works better to be more concise, and to mimick the output data structure that you want.
Say, you want LLM to generate a ``yaml`` or ``json`` string and later you can convert it back to a dictionary or even your data instance.
We can do so using the signature:

.. code-block:: python

    print(TrecData2.to_json_signature())

The json signature output will be:

.. code-block::

    {
        "question": "The question asked by the user ({'type': 'Question', 'properties': {'question': {'type': 'str', 'desc': 'The question asked by the user'}, 'metadata': {'type': 'dict', 'desc': 'The metadata of the question'}}, 'required': ['question']}) (required)",
        "label": "The label of the question (int) (optional)",
        "metadata": "The metadata of the question (dict) (required)"
    }

To yaml signature:

.. code-block::

    question: The question asked by the user ({'type': 'Question', 'properties': {'question': {'type': 'str', 'desc': 'The question asked by the user'}, 'metadata': {'type': 'dict', 'desc': 'The metadata of the question'}}, 'required': ['question']}) (required)
    label: The label of the question (int) (optional)
    metadata: The metadata of the question (dict) (required)

.. note::

    If you use ``schema`` (json string) to instruct LLMs to output `yaml` data, the LLMs might get confused and can potentially output `json` data instead.

**Exclude**

Now, if you decide to not show some fields in the output, you can use the `exclude` parameter in the methods.
Let's exclude both the ``metadata`` from class ``TrecData2`` and the ``metadata`` from class ``Question``:

.. code-block:: python

    json_signature_exclude = TrecData2.to_json_signature(exclude={"TrecData2": ["metadata"], "Question": ["metadata"]})
    print(json_signature_exclude)

The output will be:

.. code-block::

    {
        "question": "The question asked by the user ({'type': 'Question', 'properties': {'question': {'type': 'str', 'desc': 'The question asked by the user'}}, 'required': ['question']}) (required)",
        "label": "The label of the question (int) (optional)"
    }

If you only want to exclude the ``metadata`` from class ``TrecData2``- the outer class, you can pass a list of strings simply:

.. code-block:: python

    json_signature_exclude = TrecData2.to_json_signature(exclude=["metadata"])
    print(json_signature_exclude)

The output will be:

.. code-block::

    {
        "question": "The question asked by the user ({'type': 'Question', 'properties': {'question': {'type': 'str', 'desc': 'The question asked by the user'}, 'metadata': {'type': 'dict', 'desc': 'The metadata of the question'}}, 'required': ['question']}) (required)",
        "label": "The label of the question (int) (optional)"
    }

The ``exclude`` parameter works the same across all methods.

**DataClassFormatType**

For data class format, we have :class:`DataClassFormatType<core.base_data_class.DataClassFormatType>` along with ``format_class_str`` method to specify the format type for the data format methods.

.. code-block:: python

    from adalflow.core import DataClassFormatType

    json_signature = TrecData2.format_class_str(DataClassFormatType.SIGNATURE_JSON)
    print(json_signature)

    yaml_signature = TrecData2.format_class_str(DataClassFormatType.SIGNATURE_YAML)
    print(yaml_signature)

    schema = TrecData2.format_class_str(DataClassFormatType.SCHEMA)
    print(schema)

.. Describe data to LLMs
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. Data Format
.. ^^^^^^^^^^^^^^^^^^^^^^^^^

.. We need to describe either the input/output data format to give LLMs context on how to understand the input data and to generate the output data.

.. What we want to let LLM know about our input/output data format:
.. In particular, it is important for LLMs to know these five things about the data format:

.. 1. **Description** of what this field is for.  We use `desc` key in the `metadata` of `field` to describe this field. Example:

.. .. code-block:: python

..     thought: str = field(
..         metadata={"desc": "The reasoning or thought behind the question."}
..     )

.. 2. **Required/Optional**. We use either `default` or `default_factory` to mark the field as optional except when our specialized function :func:`core.base_data_class.required_field` is used in `default_factory`, which marks the field as required.
.. 3. **Field Data Type** such as `str`, `int`, `float`, `bool`, `List`, `Dict`, etc.
.. 4. **Order of the fields** matter as in a typical Chain of Thought, we want the reasoning/thought field to be in the output ahead of the answer.
.. 5. The ablility to **exclude** some fields from the output.

.. We provide two ways: (1) ``schema`` and (2) ``signature`` to describe the data format in particular.

.. **Schema**

.. ``schema`` will be a dict or json string and it is more verbose compared with ``signature``.
.. ``signature`` imitates the exact data format (`yaml` or `json`) that you want LLMs to generate.

.. Here is a quick example on our ``schema`` for  the ``MyOutputs`` data class using the `to_schema` method:

.. .. code-block:: python

..    MyOutputs.to_schema()

.. The output will be a dict:

.. .. code-block:: json

..     {
..         "name": {
..             "type": "str",
..             "desc": "The name of the person",
..             "required": false
..         },
..         "age": {
..             "type": "int",
..             "desc": "The age of the person",
..             "required": true
..         }
..     }

.. You can use `to_schema_str` to have the json string output.

.. In comparison with the schema used in other libraries:

.. .. code-block:: json

..     {
..         "properties": {
..             "name": {
..                 "title": "Name",
..                 "description": "The name of the user",
..                 "default": "John Doe",
..                 "type": "string",
..             },
..             "age": {
..                 "title": "Age",
..                 "description": "The age of the user",
..                 "type": "integer",
..             },
..         },
..         "required": ["age"],
..     }

.. Even our ``schema`` is more token efficient as you can see. We opted out of the `default` field as it is more of a fallback value in the program
.. rather than a description of the data format to LLMs.




.. **Signature**

.. ``signature`` is a string that imitates the exact data format (here we support `yaml` or `json`) that you want LLMs to generate.

.. Let's use class methods ``to_json_signature`` and ``to_yaml_signature`` to generate the signature for the ``MyOutputs`` data class:

.. .. code-block:: python

..     print(MyOutputs.to_json_signature())
..     print(MyOutputs.to_yaml_signature())

.. The json signature output will be:

.. .. code-block:: json

..     {
..         "name": "The name of the person (str) (optional)",
..         "age": "The age of the person (int) (required)"
..     }

.. The yaml signature output will be:

.. .. code-block:: yaml

..     name: The name of the person (str) (optional)
..     age: The age of the person (int) (required)

.. All of the above methods support `exclude` parameter to exclude some fields from the output.

Show data examples & parse string to data instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our functionality on data instance will help you show data examples to LLMs.
This is mainly done via ``to_dict`` method, which you can further convert to json or yaml string.
To convert the raw string back to the data instance, either from json or yaml string, we leverage class method ``from_dict``.
So it is important for ``DataClass`` to be able to ensure the reconstructed data instance is the same as the original data instance.
Here is how you can do it with a ``DataClass`` subclass:

.. code-block:: python

    example = TrecData2(Question("What is the capital of France?"), 1, {"key": "value"})
    print(example)

    dict_example = example.to_dict()
    print(dict_example)

    reconstructed = TrecData2.from_dict(dict_example)
    print(reconstructed)

    print(reconstructed == example)

The output will be:

.. code-block:: python

    TrecData2(question=Question(question='What is the capital of France?', metadata={}), label=1, metadata={'key': 'value'})
    {'question': {'question': 'What is the capital of France?', 'metadata': {}}, 'label': 1, 'metadata': {'key': 'value'}}
    TrecData2(question=Question(question='What is the capital of France?', metadata={}), label=1, metadata={'key': 'value'})
    True

On top of ``from_dict`` and ``to_dict``, we make sure you can also directly work with:

*  ``from_yaml`` (from yaml string to reconstruct instance) and ``to_yaml`` (a yaml string)
*  ``from_json`` (from json string to reconstruct instance) and ``to_json`` (a json string)

Here is how it works with ``DataClass`` subclass:

.. code-block:: python

    json_str = example.to_json()
    print(json_str)

    yaml_str = example.to_yaml(example)
    print(yaml_str)

    reconstructed_from_json = TrecData2.from_json(json_str)
    print(reconstructed_from_json)
    print(reconstructed_from_json == example)

    reconstructed_from_yaml = TrecData2.from_yaml(yaml_str)
    print(reconstructed_from_yaml)
    print(reconstructed_from_yaml == example)

The output will be:

.. code-block::

    {
        "question": {
            "question": "What is the capital of France?",
            "metadata": {}
        },
        "label": 1,
        "metadata": {
            "key": "value"
        }
    }
    question:
        question: What is the capital of France?
        metadata: {}
    label: 1
    metadata:
        key: value

    TrecData2(question=Question(question='What is the capital of France?', metadata={}), label=1, metadata={'key': 'value'})
    True
    TrecData2(question=Question(question='What is the capital of France?', metadata={}), label=1, metadata={'key': 'value'})
    True


Similarly, (1) all ``to_dict``, ``to_json``, and ``to_yaml`` works with `exclude` parameter to exclude some fields from the output,
(2) you can use ``DataClassFormatType`` along with ``format_example_str`` method to specify the format type for the data example methods.

.. code-block:: python

    from adalflow.core import DataClassFormatType

    example_str = example.format_example_str(DataClassFormatType.EXAMPLE_JSON)
    print(example_str)

    example_str = example.format_example_str(DataClassFormatType.EXAMPLE_YAML)
    print(example_str)


.. Let's create an instance of ``TrecData2`` and get the json and yaml string of the instance:



.. To better demonstrate either the data format or provide examples seen in few-shot In-context learning,
.. we provide two methods: `to_json` and `to_yaml` to convert the data instance to json or yaml string.

.. First, let's create an instance of the `MyOutputs` and get the json and yaml string of the instance:

.. .. code-block:: python

..     instance = MyOutputs(name="Jane Doe", age=25)
..     print(instance.to_json())
..     print(instance.to_yaml())

.. The json output will be:

.. .. code-block:: json

..     {
..         "name": "Jane Doe",
..         "age": 25
..     }
.. You can use `json.loads` to convert the json string back to a dictionary.

.. The yaml output will be:

.. .. code-block:: yaml

..     name: "John Doe"
..     age: 25

.. You can use `yaml.safe_load` to convert the yaml string back to a dictionary.




Load data from dataset as example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we need to load or create an instance from a dataset,  which is typically from Pytorch dataset or huggingface dataset and each data point is in
the form of a dictionary.

How you want to describe your data format to LLMs might not match to the existing dataset's key and the field name.
You can simply do a bit customization to map the dataset's key to the field name in your data class.

.. code-block:: python

    @dataclass
    class OutputFormat(DataClass):
        thought: str = field(
            metadata={
                "desc": "Your reasoning to classify the question to class_name",
            }
        )
        class_name: str = field(metadata={"desc": "class_name"})
        class_index: int = field(metadata={"desc": "class_index in range[0, 5]"})

        @classmethod
        def from_dict(cls, data: Dict[str, object]):
            _COARSE_LABELS_DESC = [
                "Abbreviation",
                "Entity",
                "Description and abstract concept",
                "Human being",
                "Location",
                "Numeric value",
            ]
            data = {
                "thought": None,
                "class_index": data["coarse_label"],
                "class_name": _COARSE_LABELS_DESC[data["coarse_label"]],
            }
            return super().from_dict(data)

.. note::

    If you are looking for data types we used to support each component or any other class like `Optimizer`, you can check out the :ref:`core.types<core-types>` file.

About __output_fields__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Though you can use `exclude` in the :class:`JsonOutputParser<components.output_parsers.outputs.JsonOutputParser>` to exclude some fields from the output, it is less readable and less convenient than
directly use `__output_fields__` in the data class to signal the output fields and directly work with :class:`DataClassParser<components.output_parsers.dataclass_parser.DataClassParser>`.

.. admonition:: References
   :class: highlight

   1. Dataclasses: https://docs.python.org/3/library/dataclasses.html



.. admonition:: API References
   :class: highlight

   - :class:`core.base_data_class.DataClass`
   - :class:`core.base_data_class.DataClassFormatType`
   - :func:`core.functional.custom_asdict`
   - :ref:`core.base_data_class<core-base_data_class>`
   - :class:`core.base_data_class.required_field`
   - :class:`components.output_parsers.outputs.JsonOutputParser`
   - :class:`components.output_parsers.dataclass_parser.DataClassParser`

.. Document
.. ------------
.. We defined `Document` to function as a `string` container, and it can be used for any kind of text data along its `metadata` and relations
.. such as `parent_doc_id` if you have ever splitted the documents into chunks, and `embedding` if you have ever computed the embeddings for the document.

.. It functions as the data input type for some `string`-based components, such as `DocumentSplitter`, `Retriever`.
