Component
============
What you will learn?

1. What is ``Component`` and why is it designed this way?
2. How to use ``Component`` along with helper classes like ``FunComponent`` and ``Sequential``?

Component
---------------------------------------
 :ref:`Component<core-component>` is to LLM task pipelines what ``nn.Module`` is to PyTorch models.

It is the base class for components, such as ``Prompt``, ``ModelClient``, ``Generator``, ``Retriever`` in LightRAG.
Your task pipeline should subclass from ``Component`` too. Instead of working with ``Tensor`` and ``Parameter`` to train models with weights and biases, our component works with any data, ``Parameter`` that can be any data type for LLM in-context learning, from manual to auto prompt engineering.
We name it differently to avoid confusion and also for better compatibility with `PyTorch`.



Here is the comparison of writing a PyTorch model and a LightRAG task component.

.. grid:: 2
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

    .. grid-item-card::  LightRAG

        .. code-block:: python

            from lightrag.core import Component, Generator
            from lightrag.components.model_client import OpenAIClient


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

In this note, we are creating an AI doctor to answer medical questions.
Run the ``DocQA`` on a query:

.. code-block:: python

    doc = DocQA()
    print(doc("What is the best treatment for headache?"))

The response is:

.. code-block::

    As a doctor, the best treatment for a headache would depend on the underlying cause of the headache. Typically, over-the-counter pain relievers such as acetaminophen, ibuprofen, or aspirin can help to alleviate the pain. However, if the headache is severe or persistent, it is important to see a doctor for further evaluation and to determine the most appropriate treatment option. Other treatment options may include prescription medications, lifestyle modifications, stress management techniques, and relaxation techniques.

As the foundamental building block in LLM task pipeline, the component is designed to serve four main purposes:

1. **Standarize the interface for all components.** This includes the `__init__` method, the `call` method for synchronous call, the `acall` method for asynchronous call, and the `__call__` which in default calls the `call` method.
2. **Provide a unified way to visualize the structure of the task pipeline** via `__repr__` method. And subclass can additional add `_extra_repr` method to add more information than the default `__repr__` method.
3. **Tracks, adds all subcomponents and parameters automatically and recursively** to assistant the building and optimizing process of the task pipeline.
4. **Manages the states and serialization**, with `state_dict` and `load_state_dict` methods in particular for parameters and `to_dict` method for serialization of all the states fall into the component's attributes, from subcomponents to parameters, to any other attributes of various data type.


Here are the benefits of using the Component class:

- Transparency.
- Flexibility.
- Searialization and deserialization.

.. Transparency
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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






.. Flexibility
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the above example shows, we added subcomponent via attributes.
We can also use methods to add more subcomponnents or parameters.

.. code-block:: python

    from lightrag.core.parameter import Parameter

    doc.register_parameter("demo", param=Parameter(data="demo"))
    # list all parameters
    for param in doc.named_parameters():
        print(param)
    # output
    # ('demo', Parameter: demo)

You can easily save the detailed states:

.. code-block:: python

    from utils.file_io import save_json

    save_json(doc.to_dict(), "doc.json")


To adds even more flexibility, we provide :class:`core.component.FunComponent` and :class:`core.component.Sequential` for more advanced use cases.


**Searalization and deserialization**

We provide ``is_pickable`` method to check if the component is pickable.
And any of your component, it is a good practise to ensure it is pickable.

FunComponent
--------------
 Use :func:`core.component.fun_to_component` as a decorator to convert any function to a Component with its unique class name.

:class:`core.component.FunComponent` is a subclass of :class:`core.component.Component` that allows you to define a component with a function.
You can directly use this class as:

.. code-block:: python

    from lightrag.core.component import FunComponent

    def add_one(x):
        return x + 1

    fun_component = FunComponent(add_one)
    print(fun_component(1))
    print(type(fun_component))

    # output:
    # 2
    # <class 'core.component.FunComponent'>


We also have :func:`core.component.fun_to_component` to convert a function to a FunComponent via decorator or directly call the function.
This approach gives you a unique component converted from the function name.

Via direct call:

.. code-block:: python

    from lightrag.core.component import fun_to_component

    fun_component = fun_to_component(add_one)
    print(fun_component(1))
    print(type(fun_component))

    # output:
    # 2
    # <class 'lightrag.core.component.AddOneComponent'>


Via decorator will be even more convenient to have a component from a function:

.. code-block:: python

    .. @fun_to_component
    def add_one(x):
        return x + 1

    print(add_one(1))
    print(type(add_one))

    # output:
    # 2
    # <class 'lightrag.core.component.AddOneComponent'>

Sequential
--------------
We have :class:`core.component.Sequential` class to PyTorch's ``nn.Sequential`` class. This is especially useful to chain together components in a sequence.  Much like the concept of ``chain`` or ``pipeline`` in other LLM libraries.
Let's put the FunComponent and DocQA together in a sequence:

.. code-block:: python

    from lightrag.core.component import Sequential

    @fun_to_component
    def enhance_query(query:str) -> str:
        return query + "Please be concise and only list the top treatments."

    seq = Sequential(enhance_query, doc)

    query = "What is the best treatment for headache?"
    print(seq(query))

We automatically enhance users' queries before passing them to the DocQA component.
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
   - :class:`core.component.Sequential`
   - :func:`core.component.fun_to_component`


We will have more advanced use cases in the upcoming tutorials.
