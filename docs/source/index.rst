
.. image:: https://raw.githubusercontent.com/SylphAI-Inc/LightRAG/main/docs/source/_static/images/adalflow-logo.png
   :width: 100%
   :alt: Adalflow Logo




.. .. raw:: html

..    <p align="center">


..     <a href="https://colab.research.google.com/drive/1TKw_JHE42Z_AWo8UuRYZCO2iuMgyslTZ?usp=sharing">
..         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
..     </a>
..    </p>

..    <div style="text-align: center; margin-bottom: 20px;">
..       <a href="https://pypi.org/project/adalflow/"><img src="https://img.shields.io/pypi/v/adalflow?style=flat-square" alt="PyPI Version"></a>
..       <a href="https://github.com/SylphAI-Inc/AdalFlow" style="display: inline-block; margin-left: 10px;">
..          <img src="https://img.shields.io/badge/GitHub-AdalFlow-blue?logo=github&style=flat-square" alt="GitHub Repo">
..       </a>
..       <a href="https://star-history.com/#SylphAI-Inc/LightRAG"><img src="https://img.shields.io/github/stars/SylphAI-Inc/LightRAG?style=flat-square" alt="GitHub Stars"></a>
..       <a href="https://discord.gg/ezzszrRZvT">
..         <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/ezzszrRZvT?style=flat">
..       </a>
..       <a href="https://opensource.org/license/MIT"><img src="https://img.shields.io/github/license/SylphAI-Inc/LightRAG" alt="License"></a>
..    </div>



..  <a href="https://pypistats.org/packages/lightrag"><img src="https://img.shields.io/pypi/dm/lightRAG?style=flat-square" alt="PyPI Downloads"></a>


.. raw:: html

    <div class="desktop-only">
      <h2 style="text-align: center; font-size: 2.5em; margin-top: 0px;">
      ⚡ The Library to Build and Auto-optimize Any LLM Task Pipeline ⚡
      </h2>
      <h3 style="text-align: center; font-size: 1.5em; margin-top: 20px; margin-bottom: 20px;">
        Embracing a design philosophy similar to PyTorch, AdalFlow is powerful, light, modular, and robust.
      </h3>
    </div>
    <div class="mobile-only">
      <h2 style="text-align: center; font-size: 1.8em; margin-top: 0px;">
      ⚡ The Library to Build and Auto-optimize Any LLM Task Pipeline ⚡
      </h2>
      <h3 style="text-align: center; font-size: 0.9em; margin-top: 20px; margin-bottom: 20px;">
        Embracing a design philosophy similar to PyTorch, AdalFlow is powerful, light, modular, and robust.
      </h3>
    </div>

    <style>
      .mobile-only {
        display: none;
      }

      @media (max-width: 600px) {
        .desktop-only {
          display: none;
        }
        .mobile-only {
          display: block;
        }
      }
    </style>


..  <div style="text-align: center;">
..      <p>
..          <em>AdalFlow</em> helps developers build and optimize <em>Retriever-Agent-Generator</em> pipelines.<br>
..          Embracing a design philosophy similar to PyTorch, it is light, modular, and robust, with a 100% readable codebase.
..      </p>
..  </div>



.. Embracing the PyTorch-like design philosophy, AdalFlow is a powerful, light, modular, and robust library to build and auto-optimize any LLM task pipeline.
.. AdalFlow is a powerful library to build and auto-optimize any LLM task pipeline with PyTorch-like design philosophy.


.. # TODO: make this using the new tool, show both the building and the training.


.. .. grid:: 1
..    :gutter: 1

..    .. grid-item-card::  PyTorch

..       .. code-block:: python

..             import torch
..             import torch.nn as nn

..             class Net(nn.Module):
..                def __init__(self):
..                   super(Net, self).__init__()
..                   self.conv1 = nn.Conv2d(1, 32, 3, 1)
..                   self.conv2 = nn.Conv2d(32, 64, 3, 1)
..                   self.dropout1 = nn.Dropout2d(0.25)
..                   self.dropout2 = nn.Dropout2d(0.5)
..                   self.fc1 = nn.Linear(9216, 128)
..                   self.fc2 = nn.Linear(128, 10)

..                def forward(self, x):
..                   x = self.conv1(x)
..                   x = self.conv2(x)
..                   x = self.dropout1(x)
..                   x = self.dropout2(x)
..                   x = self.fc1(x)
..                   return self.fc2(x)

..    .. grid-item-card::  AdalFlow

..       .. code-block:: python

..          import adalflow as adal
..          from adalflow.components.model_client import GroqAPIClient


..          class SimpleQA(adal.Component):
..             def __init__(self):
..                super().__init__()
..                template = r"""<SYS>
..                You are a helpful assistant.
..                </SYS>
..                User: {{input_str}}
..                You:
..                """
..                self.generator = adal.Generator(
..                      model_client=GroqAPIClient(),
..                      model_kwargs={"model": "llama3-8b-8192"},
..                      template=template,
..                )

..             def call(self, query):
..                return self.generator({"input_str": query})

..             async def acall(self, query):
..                return await self.generator.acall({"input_str": query})
.. raw:: html

    <h3 style="text-align: left; font-size: 1.5em; margin-top: 50px;">
    Light, Modular, and Model-agnositc Task Pipeline
    </h3>

.. Light, Modular, and Model-agnositc Task Pipeline
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLMs are like water; AdalFlow help developers quickly shape them into any applications, from GenAI applications such as chatbots, translation, summarization, code generation, RAG, and autonomous agents to classical NLP tasks like text classification and named entity recognition.


Only two fundamental but powerful base classes: `Component` for the pipeline and `DataClass` for data interaction with LLMs.
The result is a library with bare minimum abstraction, providing developers with *maximum customizability*.

You have full control over the prompt template, the model you use, and the output parsing for your task pipeline.


.. figure:: /_static/images/AdalFlow_task_pipeline.png
   :alt: AdalFlow Task Pipeline
   :align: center



.. raw:: html

    <h3 style="text-align: left; font-size: 1.5em; margin-top: 10px;">
    Unified Framework for Auto-Optimization
    </h3>

.. AdalFlow provides token-efficient and high-performing prompt optimization within a unified framework.
.. To optimize your pipeline, simply define a ``Parameter`` and pass it to our ``Generator``.
.. Wheter it is to optimize the task instruction or the few-shot demonstrations, our unified framework
.. provides you easy way to ``diagnose``, ``visualize``, ``debug``, and to ``train`` your pipeline.

.. This trace graph shows how our auto-diffentiation works :doc:`trace_graph <../tutorials/trace_graph>`.

AdalFlow provides token-efficient and high-performing prompt optimization within a unified framework.
To optimize your pipeline, simply define a ``Parameter`` and pass it to our ``Generator``.
Whether you need to optimize task instructions or few-shot demonstrations,
our unified framework offers an easy way to **diagnose**, **visualize**, **debug**, and **train** your pipeline.

This trace graph demonstrates how our auto-differentiation works: :doc:`trace_graph <../tutorials/trace_graph>`

**Trainable Task Pipeline**

Just define it as a ``Parameter`` and pass it to our ``Generator``.


.. figure:: /_static/images/Trainable_task_pipeline.png
   :alt: AdalFlow Trainable Task Pipeline
   :align: center




**AdalComponent & Trainer**

``AdalComponent`` acts as the `interpreter`  between task pipeline and the trainer, defining training and validation steps, optimizers, evaluators, loss functions, backward engine for textual gradients or tracing the demonstrations, the teacher generator.


.. figure:: /_static/images/trainer.png
   :alt: AdalFlow AdalComponent & Trainer
   :align: center



.. and Customizability


.. Light
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. AdalFlow shares similar design pattern as `PyTorch` for deep learning modeling.
.. We provide developers with fundamental building blocks of *100% clarity and simplicity*.

.. - Only two fundamental but powerful base classes: `Component` for the pipeline and `DataClass` for data interaction with LLMs.
.. - A highly readable codebase and less than two levels of class inheritance. :doc:`tutorials/class_hierarchy`.
.. - We maximize the library's tooling and prompting capabilities to minimize the reliance on LLM API features such as tools and JSON format.
.. - The result is a library with bare minimum abstraction, providing developers with *maximum customizability*.



.. Modular
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. AdalFlow resembles PyTorch in the way that we provide a modular and composable structure for developers to build and to optimize their LLM applications.

.. - `Component` and `DataClass` are to AdalFlow for LLM Applications what  `module` and `Tensor` are to PyTorch for deep learning modeling.
.. - `ModelClient` to bridge the gap between the LLM API and the AdalFlow pipeline.
.. - `Orchestrator` components like `Retriever`, `Embedder`, `Generator`, and `Agent` are all model-agnostic (you can use the component on different models from different providers).


.. Similar to the PyTorch `module`, our `Component` provides excellent visualization of the pipeline structure.

.. .. code-block::

..    SimpleQA(
..       (generator): Generator(
..          model_kwargs={'model': 'llama3-8b-8192'},
..          (prompt): Prompt(
..             template: <SYS>
..                   You are a helpful assistant.
..                   </SYS>
..                   User: {{input_str}}
..                   You:
..                   , prompt_variables: ['input_str']
..          )
..          (model_client): GroqAPIClient()
..       )
..    )

.. To switch to `gpt-3.5-turbo` by OpenAI, simply update the `model_client`` and `model_kwargs` in the Generator component.

.. .. code-block:: python

..    from adalflow.components.model_client import OpenAIClient

..    self.generator = adal.Generator(
..         model_client=OpenAIClient(),
..         model_kwargs={"model": "gpt-3.5-turbo"},
..         template=template,
..     )




.. Robust
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. Our simplicity did not come from doing less.
.. On the contrary, we have to do more and go deeper and wider on any topic to offer developers *maximum control and robustness*.

.. - LLMs are sensitive to the prompt. We allow developers full control over their prompts without relying on LLM API features such as tools and JSON format with components like `Prompt`, `OutputParser`, `FunctionTool`, and `ToolManager`.
.. - Our goal is not to optimize for integration, but to provide a robust abstraction with representative examples. See this in :ref:`ModelClient<tutorials-model_client>` and :ref:`Retriever<tutorials-retriever>` components.
.. - All integrations, such as different API SDKs, are formed as optional packages but all within the same library. You can easily switch to any models from different providers that we officially support.





Unites Research and Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Our team has experience in both AI research and production.
We are building a library that unites the two worlds, forming a healthy LLM application ecosystem.

- To resemble the PyTorch library makes it easier for LLM researchers to use the library.
- Researchers building on AdalFlow enable production engineers to easily adopt, test, and iterate on their production data.
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


.. .. Hide the use cases for now
.. toctree::
   :maxdepth: 1
   :caption: Use Cases - How different parts are used to build various LLM applications
   :hidden:

   use_cases/index


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
