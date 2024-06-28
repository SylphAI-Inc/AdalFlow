.. _core-base_data_class_note:
DataClass
============

.. admonition:: Author
   :class: highlight

   `Li Yin <https://github.com/liyin2015>`_

In PyTorch, ``Tensor`` is the data type used in ``Module`` and ``Optimizer`` across the library.
The data in particular is a multi-dimensional matrix such as such as weights, biases, and even inputs and predictions.
In LLM applications, you can think of the data as a freeform data class with various fields and types of data.
For instance:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class TrecData:
        question: str
        label: int

It is exactly a single input data item in a typical PyTorch ``Dataset`` or a `HuggingFace` ``Dataset``.
The unique thing is all data or tools interact with LLMs via prompt and text prediction, which is a single ``str``.

Most existing libraries use `Pydantic` to handle the serialization(convert to string) and deserialization(convert from string) of the data.
But, in LightRAG, we in particular designed :class:`core.base_data_class.DataClass` using native `dataclasses` module.
The reasons are:

1. ``dataclasses`` module's `dataclass` decorator, along with `field` (`metadata`, `default`) can be especially helpful to describe the data format to LLMs. `dataclass` also saves users time on writing the boilerplate code such as `__init__`, `__repr__`, `__str__` etc.

2. `dataclasses` native module is more lightweight, flexible, and user-friendly than `Pydantic`.

3. Though we need more customization on ``BaseClass`` compared with directly using `Pydantic`, we will enjoy more transparency and control over the data format.

Here is how users can define a data class with our customized methods in LightRAG:

.. code-block:: python

    from lightrag.core.base_data_class import (
        DataClass,
        required_field,
    )
    from dataclasses import field, dataclass

    @dataclass
    class MyOutputs(DataClass):
        name: str = field(
            default="John Doe",  # Optional field
            metadata={"desc": "The name of the person", "prefix": "Name:"},
        )
        age: int = field(
            default_factory=required_field, # Required field
            metadata={"desc": "The age of the person", "prefix": "Age:"},
        )

.. note::

    `required_field` is a helper function to mark the field as required. Otherwise, using either `default` or `default_factory` will make the field optional.
     ``Optional`` type hint will not affect the field's required status. You can use this to work with static type checkers such as `mypy` if you want to.
.. Now, let's see  how we design class and instance methods to describe the data format and the data instance to LLMs.


Describe data to LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data Format
^^^^^^^^^^^^^^^^^^^^^^^^^

We need to describe either the input/output data format to give LLMs context on how to understand the input data and to generate the output data.

What we want to let LLM know about our input/output data format:
In particular, it is important for LLMs to know these five things about the data format:

1. **Description** of what this field is for.  We use `desc` key in the `metadata` of `field` to describe this field. Example:

.. code-block:: python

    thought: str = field(
        metadata={"desc": "The reasoning or thought behind the question."}
    )

2. **Required/Optional**. We use either `default` or `default_factory` to mark the field as optional except when our specialized function :func:`core.base_data_class.required_field` is used in `default_factory`, which marks the field as required.
3. **Field Data Type** such as `str`, `int`, `float`, `bool`, `List`, `Dict`, etc.
4. **Order of the fields** matter as in a typical Chain of Thought, we want the reasoning/thought field to be in the output ahead of the answer.
5. The ablility to **exclude** some fields from the output.

We provide two ways: (1) ``schema`` and (2) ``signature`` to describe the data format in particular.

**Schema**

``schema`` will be a dict or json string and it is more verbose compared with ``signature``.
``signature`` imitates the exact data format (`yaml` or `json`) that you want LLMs to generate.

Here is a quick example on our ``schema`` for  the ``MyOutputs`` data class using the `to_schema` method:

.. code-block:: python

   MyOutputs.to_schema()

The output will be a dict:

.. code-block:: json

    {
        "name": {
            "type": "str",
            "desc": "The name of the person",
            "required": false
        },
        "age": {
            "type": "int",
            "desc": "The age of the person",
            "required": true
        }
    }

You can use `to_schema_str` to have the json string output.

In comparison with the schema used in other libraries:

.. code-block:: json

    {
        "properties": {
            "name": {
                "title": "Name",
                "description": "The name of the user",
                "default": "John Doe",
                "type": "string",
            },
            "age": {
                "title": "Age",
                "description": "The age of the user",
                "type": "integer",
            },
        },
        "required": ["age"],
    }

Even our ``schema`` is more token efficient as you can see. We opted out of the `default` field as it is more of a fallback value in the program
rather than a description of the data format to LLMs.

.. note::

    If you use ``schema`` (json string) to instruct LLMs to output `yaml` data, the LLMs might get confused and can potentially output `json` data instead.


**Signature**

``signature`` is a string that imitates the exact data format (here we support `yaml` or `json`) that you want LLMs to generate.

Let's use class methods ``to_json_signature`` and ``to_yaml_signature`` to generate the signature for the ``MyOutputs`` data class:

.. code-block:: python

    print(MyOutputs.to_json_signature())
    print(MyOutputs.to_yaml_signature())

The json signature output will be:

.. code-block:: json

    {
        "name": "The name of the person (str) (optional)",
        "age": "The age of the person (int) (required)"
    }

The yaml signature output will be:

.. code-block:: yaml

    name: The name of the person (str) (optional)
    age: The age of the person (int) (required)

All of the above methods support `exclude` parameter to exclude some fields from the output.

Data Instance or say Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To better demonstrate either the data format or provide examples seen in few-shot In-context learning,
we provide two methods: `to_json` and `to_yaml` to convert the data instance to json or yaml string.

First, let's create an instance of the `MyOutputs` and get the json and yaml string of the instance:

.. code-block:: python

    instance = MyOutputs(name="Jane Doe", age=25)
    print(instance.to_json())
    print(instance.to_yaml())

The json output will be:

.. code-block:: json

    {
        "name": "Jane Doe",
        "age": 25
    }
You can use `json.loads` to convert the json string back to a dictionary.

The yaml output will be:

.. code-block:: yaml

    name: "John Doe"
    age: 25

You can use `yaml.safe_load` to convert the yaml string back to a dictionary.




Load data from dataset as example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we need to load or create an instance from a dataset,  which is typically from Pytorch dataset or huggingface dataset and each data point is in
the form of a dictionary.

Let's create an instance of the `MyOutputs` from a dictionary:

.. code-block:: python

    data = {"name": "Jane Doe", "age": 25}
    print(MyOutputs.from_dict(data))

    # Output
    # MyOutputs(name='Jane Doe', age=25)

In most cases, your dataset's key and the field name might not directly match.
Instead of providing a mapping argument in the library, we suggest users to customize `from_dict` method for more **control** and **flexibility**.

Here is a real-world example:

.. code-block:: python

    class OutputFormat(DataClass):
        thought: str = field(
            metadata={
                "desc": "Your reasoning to classify the question to class_name",
            }
        )
        class_name: str = field(metadata={"desc": "class_name"})
        class_index: int = field(metadata={"desc": "class_index in range[0, 5]"})

        @classmethod
        def from_dict(cls, data: Dict[str, Any]):
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



.. Document
.. ------------
.. We defined `Document` to function as a `string` container, and it can be used for any kind of text data along its `metadata` and relations
.. such as `parent_doc_id` if you have ever splitted the documents into chunks, and `embedding` if you have ever computed the embeddings for the document.

.. It functions as the data input type for some `string`-based components, such as `DocumentSplitter`, `Retriever`.
