.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/drive/1aD0C8-iMB8quIn8FKhrtFAGcrboRNg2C?usp=sharing" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/LightRAG/blob/main/adalflow/adalflow/core/component.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>


Component
============

.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_

.. What you will learn?

.. 1. What is ``Component`` and why is it designed this way?
.. 2. How to use ``Component`` along with helper classes like ``FunComponent`` and ``Sequential``?


:ref:`Component<core-component>` is to LLM task pipelines what `nn.Module` is to PyTorch models.
It is the base class for components such as ``Prompt``, ``ModelClient``, ``Generator``, ``Retriever`` in LightRAG.
Your task pipeline should also subclass from ``Component``.



Design
---------------------------------------

Different from PyTorch's nn.Module, which works exclusively with Tensor and Parameter to train models with weights and biases, our component can work with different types of data, from a string or a list of strings to a list of :class:`Document<core.types.Document>`.

..  `Parameter` that can be any data type for LLM in-context learning, from manual to auto prompt engineering.


Here is the comparison of writing a PyTorch model and a LightRAG task pipeline.


.. grid:: 1
    :gutter: 1

    .. grid-item-card::  PyTorch

        .. code-block:: python

            import torch.nn as nn
            import torch.nn.functional as F

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 20, 5)
                    self.conv2 = nn.Conv2d(20, 20, 5)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    return F.relu(self.conv2(x))

    .. grid-item-card::  AdalFlow

        .. code-block:: python

            from adalflow.core import Component, Generator
            from adalflow.components.model_client import OpenAIClient


            template_doc = r"""<SYS> You are a doctor </SYS> User: {{input_str}}"""

            class DocQA(Component):
                def __init__(self):
                    super().__init__()
                    self.doc = Generator(
                        template=template_doc,
                        model_client=OpenAIClient(),
                        model_kwargs={"model": "gpt-3.5-turbo"},
                    )

                def call(self, query: str) -> str:
                    return self.doc(prompt_kwargs={"input_str": query}).data

As the fundamental building block in LLM task pipelines, the component is designed to serve five main purposes:

1. **Standardize the interface for all components.**
   This includes the `__init__` method, the `call` method for synchronous calls, the `acall` method for asynchronous calls, and the `__call__` method, which by default calls the `call` method.

2. **Provide a unified way to visualize the structure of the task pipeline**
   via the `__repr__` method. Subclasses can additionally add the `_extra_repr` method to include more information than the default `__repr__` method.

3. **Track and add all subcomponents and parameters automatically and recursively**
   to assist in the building and optimizing process of the task pipeline.

4. **Manage the states and serialization**,
   with `state_dict` and `load_state_dict` methods specifically for parameters, and the `to_dict` method for serialization of all states within the component's attributes, from subcomponents to parameters, to any other attributes of various data types.

5. **Make all components configurable using `json` or `yaml` files**.
   This is especially useful for experimenting or building data processing pipelines.

These features are key to keeping the LightRAG pipeline transparent, flexible, and easy to use.
By subclassing from the `Component` class, you will get most of these features out of the box.


.. As the foundamental building block in LLM task pipeline, the component is designed to serve five main purposes:

.. 1. **Standarize the interface for all components.** This includes the `__init__` method, the `call` method for synchronous call, the `acall` method for asynchronous call, and the `__call__` which in default calls the `call` method.
.. 2. **Provide a unified way to visualize the structure of the task pipeline** via `__repr__` method. And subclass can additional add `_extra_repr` method to add more information than the default `__repr__` method.
.. 3. **Tracks, adds all subcomponents and parameters automatically and recursively** to assistant the building and optimizing process of the task pipeline.
.. 4. **Manages the states and serialization**, with `state_dict` and `load_state_dict` methods in particular for parameters and `to_dict` method for serialization of all the states fall into the component's attributes, from subcomponents to parameters, to any other attributes of various data type.
.. 5. **Make all components configurable from using `json` or `yaml` files**. This is especially useful for experimenting or building data processing pipelines.

.. These features are key to keep LightRAG pipeline transparent, flexible, and easy to use.
.. By subclassing from the `Component` class, you will get most of these features out of the box.


Component in Action
---------------------------------------




In this note, we are creating an AI doctor to answer medical questions.
Run the ``DocQA`` on a query:


.. code-block:: python

    doc = DocQA()
    print(doc("What is the best treatment for headache?"))

The response is:

.. code-block::

    As a doctor, the best treatment for a headache would depend on the underlying cause of the headache. Typically, over-the-counter pain relievers such as acetaminophen, ibuprofen, or aspirin can help to alleviate the pain. However, if the headache is severe or persistent, it is important to see a doctor for further evaluation and to determine the most appropriate treatment option. Other treatment options may include prescription medications, lifestyle modifications, stress management techniques, and relaxation techniques.

Print the structure
~~~~~~~~~~~~~~~~~~~~~

We can easily visualize the structure via `print`:

.. code-block:: python

    doc = DocQA()
    print(doc)

The printout:

.. code-block::


    DocQA(
        (doc): Generator(
            model_kwargs={'model': 'gpt-3.5-turbo'}, model_type=ModelType.LLM
            (prompt): Prompt(template: <SYS> You are a doctor </SYS> User: {{input_str}}, prompt_variables: ['input_str'])
            (model_client): OpenAIClient()
        )
    )


Configure from file
~~~~~~~~~~~~~~~~~~~~~



.. Flexibility
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the above example shows, we added subcomponent via attributes.
We can also use methods to add more subcomponnents or parameters.


.. code-block:: python

    from adalflow.core.parameter import Parameter

    doc.register_parameter("demo", param=Parameter(data="demo"))
    # list all parameters
    for param in doc.named_parameters():
        print(param)

The output:

.. code-block::

    ('demo', Parameter: demo)

You can easily save the detailed states:

.. code-block:: python

    from utils.file_io import save_json

    save_json(doc.to_dict(), "doc.json")

To add even more flexibility, we provide :class:`FunComponent<core.component.FunComponent>` and :class:`Sequential<core.container.Sequential>` for more advanced use cases.



Serialization and deserialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide the ``is_pickable`` method to check if the component is pickable.
It is good practice to ensure that any of your components are pickable.





FunComponent
--------------
 Use :func:`fun_to_component<core.component.fun_to_component>` as a decorator to convert any function to a Component with its unique class name.

:class:`FunComponent<core.component.FunComponent>` is a subclass of :class:`Component<core.component.Component>` that allows you to define a component with a function.
You can directly use this class as:

.. code-block:: python

    from adalflow.core.component import FunComponent

    def add_one(x):
        return x + 1

    fun_component = FunComponent(add_one)
    print(fun_component(1))
    print(type(fun_component))

The printout:

.. code-block::

    2
    <class 'core.component.FunComponent'>



We also have :func:`fun_to_component<core.component.fun_to_component>` to convert a function to a `FunComponent` via a decorator or by directly calling the function.
This approach gives you a unique component converted from the function name.

Via direct call:


.. code-block:: python

    from adalflow.core.component import fun_to_component

    fun_component = fun_to_component(add_one)
    print(fun_component(1))
    print(type(fun_component))

The output:

.. code-block::

    2
    <class 'adalflow.core.component.AddOneComponent'>




Using a decorator is an even more convenient way to create a component from a function:

.. code-block:: python

    @fun_to_component
    def add_one(x):
        return x + 1

    print(add_one(1))
    print(type(add_one))

    # output:
    # 2
    # <class 'adalflow.core.component.AddOneComponent'>

Sequential
--------------



We have the :class:`Sequential<core.container.Sequential>` class, which is similar to PyTorch's ``nn.Sequential`` class.
This is especially useful for chaining together components in a sequence, much like the concept of ``chain`` or ``pipeline`` in other LLM libraries.
Let's put the `FunComponent`` and `DocQA`` together in a sequence:

.. code-block:: python

    from adalflow.core.container import Sequential

    @fun_to_component
    def enhance_query(query:str) -> str:
        return query + "Please be concise and only list the top treatments."

    seq = Sequential(enhance_query, doc)

    query = "What is the best treatment for headache?"
    print(seq(query))

We automatically enhance users' queries before passing them to the `DocQA` component.
The output is:




.. code-block::

    1. Over-the-counter pain relievers like acetaminophen, ibuprofen, or aspirin
    2. Rest and relaxation
    3. Stay hydrated and drink plenty of water

The structure of the sequence using ``print(seq)``:

.. code-block::

    Sequential(
    (0): EnhanceQueryComponent()
    (1): DocQA(
            (doc): Generator(
            model_kwargs={'model': 'gpt-3.5-turbo'}, model_type=ModelType.LLM
            (prompt): Prompt(template: <SYS> You are a doctor </SYS> User: {{input_str}}, prompt_variables: ['input_str'])
            (model_client): OpenAIClient()
            )
        )
    )

.. admonition:: API reference
   :class: highlight

   - :class:`core.component.Component`
   - :class:`core.component.FunComponent`
   - :class:`core.container.Sequential`
   - :func:`core.component.fun_to_component`


We will cover more advanced use cases in the upcoming tutorials.
