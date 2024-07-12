Introduction to BaseDataClass
=======================================

In this tutorial, we will discuss how to use ``BaseDataClass`` to streamline the data handling, serialization and description in LightRAG.

`OpenAI's cookbook <https://cookbook.openai.com/articles/techniques_to_improve_reliability>`_ emphasizes that LLM works better when operating **structured** and **consistent** data.
To solve this, ``LightRAG`` provides ``BaseDataClass`` for developers to manage data with control and flexibility, including:

* getting structured dataclass/instance metadata(`signature` or `schema`)
* formatting class instance to ``yaml``, ``dict`` or ``json``
* loading data from dictionary

Like the role of ``Tensor`` in ``PyTorch``, ``BaseDataClass`` in ``LightRAG`` is the base class across all **dataclasses**.
``BaseDataClass`` offers to create `signature` or `schema` from both classeses and instances. It will also generate structured instances data. Developers can use ``BaseDataClass`` to easily define and describe dataclasses that handle the data input or output in LLM applications, keeping data consistent and structured.
In the following tutorial, we will investigate the functionality of ``BaseDataClass`` with examples.

**1. Create Signature and Schema**

* **Signature:** Signature has simpler content and structure and hence more token efficient than schema. ``LightRAG`` supports ``json`` and ``yaml`` formating.

* **Schema:** Schema is enssentially a dictionary containing more keys to show detailed information. Because of the detailed content, schema can mislead the model if not used properly.

Example to get signature and schema from a dataclass:

.. code-block:: python

    from core.base_data_class import BaseDataClass
    from dataclasses import dataclass, field
    # Define a dataclass
    @dataclass
    class MyOutputs(BaseDataClass):
        age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
        name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})
    # Create signatures
    print(f"json signature:")
    print(MyOutputs.to_json_signature())
    print(f"yaml signature:")
    print(MyOutputs.to_yaml_signature())
    # Create class schema
    print(f"class schema:")
    print(MyOutputs.get_data_class_schema())

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Signature Output

        .. code-block:: python

            # json signature:
            {
                "age": "The age of the person (int) (required)",
                "name": "The name of the person (str) (required)"
            }
            # yaml signature:
            age: The age of the person (int) (required)
            name: The name of the person (str) (required)


    .. grid-item-card::  Schema Output

        .. code-block:: python

            {
                'age': {'type': 'int', 'description': '', 'required': True},
                'name': {'type': 'str', 'description': '', 'required': True}
            }

As `signature` and `schema` are well-structured, developers can use them to instruct the model the output data format.

Besides creating `signature` and `schema` for classes, ``BaseDataClass`` works for single instances as well.
Example to get signiture and schema from an instance:

.. code-block:: python

    from core.base_data_class import BaseDataClass
    from dataclasses import dataclass, field
    # Define a dataclass
    @dataclass
    class MyOutputs(BaseDataClass):
        age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
        name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})

    my_instance = MyOutputs(age=25, name="John Doe")
    # my_instance json signiture
    print(my_instance.to_json_signature())
    # my_instance yaml signiture
    print(my_instance.to_yaml_signature())
    # my_instance schema
    print(my_instance.get_data_class_schema())

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Signature Output

        .. code-block:: python

            # json signature:
            {
                "age": "The age of the person (int) (required)",
                "name": "The name of the person (str) (required)"
            }
            # yaml signature:
            age: The age of the person (int) (required)
            name: The name of the person (str) (required)

    .. grid-item-card::  Schema Output

        .. code-block:: python

            {
                'age': {'type': 'int', 'description': '', 'required': True},
                'name': {'type': 'str', 'description': '', 'required': True}
            }

**2. Format Instances**

Developers can use ``BaseDataClass`` not only to format the input or output, but also to format examples during tasks such as few-shot prompting.
Example to get structured instance examples(``yaml`` or ``json``):

.. code-block:: python

    from core.base_data_class import BaseDataClass
    from dataclasses import dataclass, field

    @dataclass
    class MyOutputs(BaseDataClass):
        age: int = field(metadata={"desc": "The age of the person", "prefix": "Age:"})
        name: str = field(metadata={"desc": "The name of the person", "prefix": "Name:"})

    my_instance = MyOutputs(age=25, name="John Doe")
    # my_instance json signiture
    print(my_instance.to_json())
    # my_instance yaml signiture
    print(my_instance.to_yaml())

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  json Output

        .. code-block:: python

            {
                "age": 25,
                "name": "John Doe"
            }


    .. grid-item-card::  yaml Output

        .. code-block:: python

            age: 25
            name: John Doe

The examples demonstrate how ``BaseDataClass`` works for describing dataclasses and formatting instance to ``yaml`` and ``json``.
Developers should select schema or signature depends on the use case.

**3. Load Data from Dictionary**

If developers want to load data from a dictionary to a certain data class, they can run:
``loaded_example = MyOutputs.from_dict({"age":10, "name":"Harry"})``.

(For details, please refer to :class:`core.base_data_class.BaseDataClass`.)


**4. Implement with Other Components**

What's more, developers can use the dataclasses to interact with the ``Prompt`` and ``Generator`` classes, enhancing the consistency and structure of the application data flow.
(``LightRAG`` uses ``jinja2`` for prompt template, make sure you've checked ``jinja2`` template tutorial before reading the example.)

Example:

.. code-block:: python

    from core.base_data_class import BaseDataClass
    from dataclasses import dataclass, field
    from core.prompt_builder import Prompt

    # define a dataclass formatting the data
    @dataclass
    class JokeOutput(BaseDataClass):
        setup: str = field(metadata={"desc": "question to set up a joke"}, default="")
        punchline: str = field(metadata={"desc": "answer to resolve the joke"}, default="")

    # initialize an example
    joke_example = JokeOutput(
        setup="Why did the scarecrow win an award?",
        punchline="Because he was outstanding in his field.",
    )

    OUTPUT_FORMAT = r"""
    Your output should be formatted as a standard YAML instance with the following schema:
    ```
    {{schema}}
    ```
    {% if example %}
    Here is an example:
    ```
    {{example}}
    ```
    {% endif %}
    """

    prompt_template = Prompt(template=OUTPUT_FORMAT)
    prompt = prompt_template(schema=JokeOutput.to_yaml_signature(), example=joke_example.to_yaml())

    print(prompt)

    # Your output should be formatted as a standard YAML instance with the following schema:
    # ```
    # setup: question to set up a joke (str) (optional)
    # punchline: answer to resolve the joke (str) (optional)
    # ```
    # Here is an example:
    # ```
    # punchline: Because he was outstanding in his field.
    # setup: Why did the scarecrow win an award?
    # ```

**5. Summary**

In this tutorial, we've covered how to use ``BaseDataClass`` to create structured dataclass/instance `signature` and `schema`, format instance, load data from dictionary to the dataclass, and implement the ``BaseDataClass`` with other components.
