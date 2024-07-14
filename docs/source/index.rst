
.. image:: https://raw.githubusercontent.com/SylphAI-Inc/LightRAG/main/docs/source/_static/images/LightRAG-logo-doc.jpeg
   :width: 100%
   :alt: LightRAG Logo




.. raw:: html

   <div style="text-align: center; margin-bottom: 20px;">
      <a href="https://pypi.org/project/lightRAG/"><img src="https://img.shields.io/pypi/v/lightRAG?style=flat-square" alt="PyPI Version"></a>
      <a href="https://star-history.com/#SylphAI-Inc/LightRAG"><img src="https://img.shields.io/github/stars/SylphAI-Inc/LightRAG?style=flat-square" alt="GitHub Stars"></a>
      <a href="https://discord.gg/ezzszrRZvT">  <img src="https://img.shields.io/discord/1065084981904429126?style=flat-square" alt="Discord"></a>
      <a href="https://opensource.org/license/MIT"><img src="https://img.shields.io/github/license/SylphAI-Inc/LightRAG" alt="License"></a>
   </div>



..  <a href="https://pypistats.org/packages/lightRAG"><img src="https://img.shields.io/pypi/dm/lightRAG?style=flat-square" alt="PyPI Downloads"></a>


.. raw:: html

    <h1 style="text-align: center; font-size: 2em; margin-top: 10px;">⚡ The Lightning Library for Large Language Model Applications ⚡</h1>

    <div style="text-align: center;">
        <p>
            <em>LightRAG</em> helps developers build and optimize <em>Retriever-Agent-Generator</em> pipelines.<br>
            Embracing a design philosophy similar to PyTorch, it is light, modular, and robust, with a 100% readable codebase.
        </p>
    </div>









.. and Customizability


Light
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LightRAG shares similar design pattern as `PyTorch` for deep learning modeling.
We provide developers with fundamental building blocks of *100% clarity and simplicity*.

- Only two fundamental but powerful base classes: `Component` for the pipeline and `DataClass` for data interaction with LLMs.
- A highly readable codebase and less than two levels of class inheritance. :doc:`tutorials/class_hierarchy`.
- We maximize the library's tooling and prompting capabilities to minimize the reliance on LLM API features such as tools and JSON format.
- The result is a library with bare minimum abstraction, providing developers with *maximum customizability*.


.. grid:: 1
   :gutter: 1

   .. grid-item-card::  PyTorch

      .. code-block:: python

            import torch
            import torch.nn as nn

            class Net(nn.Module):
               def __init__(self):
                  super(Net, self).__init__()
                  self.conv1 = nn.Conv2d(1, 32, 3, 1)
                  self.conv2 = nn.Conv2d(32, 64, 3, 1)
                  self.dropout1 = nn.Dropout2d(0.25)
                  self.dropout2 = nn.Dropout2d(0.5)
                  self.fc1 = nn.Linear(9216, 128)
                  self.fc2 = nn.Linear(128, 10)

               def forward(self, x):
                  x = self.conv1(x)
                  x = self.conv2(x)
                  x = self.dropout1(x)
                  x = self.dropout2(x)
                  x = self.fc1(x)
                  return self.fc2(x)

   .. grid-item-card::  LightRAG

      .. code-block:: python

         from lightrag.core import Component, Generator
         from lightrag.components.model_client import GroqAPIClient


         class SimpleQA(Component):
            def __init__(self):
               super().__init__()
               template = r"""<SYS>
               You are a helpful assistant.
               </SYS>
               User: {{input_str}}
               You:
               """
               self.generator = Generator(
                     model_client=GroqAPIClient(),
                     model_kwargs={"model": "llama3-8b-8192"},
                     template=template,
               )

            def call(self, query):
               return self.generator({"input_str": query})

            async def acall(self, query):
               return await self.generator.acall({"input_str": query})

.. - We use 10X less code than other libraries to achieve 10X more robustness and flexibility.


.. Each developer has unique data needs to build their own models/components, experiment with In-context Learning (ICL) or model finetuning, and deploy the LLM applications to production. This means the library must provide fundamental lower-level building blocks and strive for clarity and simplicity:



Modular
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightRAG resembles PyTorch in the way that we provide a modular and composable structure for developers to build and to optimize their LLM applications.

- `Component` and `DataClass` are to LightRAG for LLM Applications what  `module` and `Tensor` are to PyTorch for deep learning modeling.
- `ModelClient` to bridge the gap between the LLM API and the LightRAG pipeline.
- `Orchestrator` components like `Retriever`, `Embedder`, `Generator`, and `Agent` are all model-agnostic (you can use the component on different models from different providers).


Similar to the PyTorch `module`, our `Component` provides excellent visualization of the pipeline structure.

.. code-block::

   SimpleQA(
      (generator): Generator(
         model_kwargs={'model': 'llama3-8b-8192'},
         (prompt): Prompt(
            template: <SYS>
                  You are a helpful assistant.
                  </SYS>
                  User: {{input_str}}
                  You:
                  , prompt_variables: ['input_str']
         )
         (model_client): GroqAPIClient()
      )
   )

To switch to `gpt-3.5-turbo` by OpenAI, simply update the `model_client`` and `model_kwargs` in the Generator component.

.. code-block:: python

   from lightrag.components.model_client import OpenAIClient

   self.generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
        template=template,
    )


.. and Robustness


Robust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Our simplicity did not come from doing less.
On the contrary, we have to do more and go deeper and wider on any topic to offer developers *maximum control and robustness*.

- LLMs are sensitive to the prompt. We allow developers full control over their prompts without relying on LLM API features such as tools and JSON format with components like `Prompt`, `OutputParser`, `FunctionTool`, and `ToolManager`.
- Our goal is not to optimize for integration, but to provide a robust abstraction with representative examples. See this in :ref:`ModelClient<tutorials-model_client>` and :ref:`Retriever<tutorials-retriever>` components.
- All integrations, such as different API SDKs, are formed as optional packages but all within the same library. You can easily switch to any models from different providers that we officially support.



.. Coming from a deep AI research background, we understand that the more control and transparency developers have over their prompts, the better. In default:

.. - LightRAG simplifies what developers need to send to LLM proprietary APIs to just two messages each time: a `system message` and a `user message`. This minimizes reliance on and manipulation by API providers.

.. - LightRAG provides advanced tooling for developers to build `agents`, `tools/function calls`, etc., without relying on any proprietary API provider's 'advanced' features such as `OpenAI` assistant, tools, and JSON format

.. It is the future of LLM applications

Unites Research and Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Our team has experience in both AI research and production.
We are building a library that unites the two worlds, forming a healthy LLM application ecosystem.

- To resemble the PyTorch library makes it easier for LLM researchers to use the library.
- Researchers building on LightRAG enable production engineers to easily adopt, test, and iterate on their production data.
- Our 100% control and clarity of the source code further make it easy for product teams to build on and for researchers to extend their new methods.


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   get_started/index




.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   tutorials/index
   .. :caption: Tutorials - How each part works
   .. :hidden:


.. Hide the use cases for now
   toctree::
   .. :maxdepth: 1
   .. :caption: Use Cases - How different parts are used to build various LLM applications
   .. :hidden:

   .. tutorials/index


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   apis/index


      .. :caption: Benchmarks

      .. Manually add documents for the code in benchmarks


..    :glob:
..    :maxdepth: 1
..    :caption: Resources

..    resources/index

.. hide the for contributors now

   .. :glob:
   .. :maxdepth: 1
   .. :caption: For Contributors
   .. :hidden:

   .. contributor/index
