Introduction to BaseDataClass
=======================================

In this tutorial, we will discuss how to use BaseDataClass to streamline the data handling, serialization and description in LightRAG.

What should you concern about the data flow in your LLM applications?

* **Data Format:** `OpenAI's cookbook <https://cookbook.openai.com/articles/techniques_to_improve_reliability>`_ emphasizes that LLM works better when operating **structured** and **consistent** data. 

* **Token Counts:** The number of input/output tokens matter a lot to the project budget as each token costs money. Long prompts often generate slow and less accurate responses. Meanwhile, the model context window, although getting longer during iteration, is still limited. Therefore, it is important to comsume tokens efficiently.
 
To address these concerns, ``LightRAG`` provides ``BaseDataClass`` for developers to manage data with control and flexibility.
Like the role of ``Tensor`` in ``PyTorch``, ``BaseDataClass`` in ``LightRAG`` is the base class accross all **dataclasses**. 

``BaseDataClass`` offers to create `signature` or `schema` from both classeses and instances. ``BaseDataClass`` can also help developers generate structured instances data.

* **Signature:** Signature has simpler content and structure and hence more token efficient than schema. ``LightRAG`` supports ``json`` and ``yaml`` formating. 

* **Schema:** Schema is enssentially a dictionary containing more keys to show detailed information. Because of the detailed content, schema can mislead the model if not used properly.

Example to get signature and schema from a dataclass:

.. code-block:: python

    from core.data_classes import BaseDataClass
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

Example to get signiture and schema from an instance:

.. code-block:: python

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


Example to get structured output of instance(``yaml`` or ``json``):

.. code-block:: python

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


For detailed methods, please check :class:`core.data_classes.BaseDataClass`.
The examples demonstrate how ``BaseDataClass`` works for describing dataclasses and structure instance to ``yaml`` and ``json`` output. 
Developers should select schema or signature depends on the use case.

With ``BaseDataClass``, developers can define data classes, use signatures for efficient token usage, and structure input/intermediate data/output.

What's more, developers can use the dataclasses to interact with the ``Prompt`` and ``Generator`` classes, enhancing the consistency and structure of the application data flow.





