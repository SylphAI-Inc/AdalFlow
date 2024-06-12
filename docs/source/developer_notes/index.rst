.. _developer_notes:


Developer Notes 
=============================

*How each part works*

Learn LightRAG design phisolophy and the reasoning(`why` and `how-to`) behind each core part within the LightRAG library.
This is also our tutorials showing how each part works before we move ahead to build use cases (LLM applications).

.. note::

   You can read interchangably between :ref:`Use Cases <use_cases>`.



.. figure:: /_static/LLM_arch.png
   :alt: LLM application is no different from a mode training/eval workflow
   :align: center

   LLM application is no different from a mode training/eval workflow

   .. :height: 100px
   .. :width: 200px
   
LightRAG library focus on providing building blocks for developers to **build** and **optimize** the `task pipeline`.
We have clear design phisolophy:



.. toctree::
   :maxdepth: 1

   lightrag_design_philosophy

   llm_intro




Building
=============================

Base classes
---------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Base Class
     - Description
   * - :doc:`component`
     - Similar to ``Module`` in `PyTorch`, it standardizes the interface of all components with `call`, `acall`, and `__call__` methods, handles states, and serialization. Components can be easily chained togehter via `Sequential` for now.
   * - :doc:`base_data_class`
     - Leverages the ``dataclasses`` module in Python to ease the data interaction with prompt and serialization.


.. create side bar navigation
.. toctree::
   :maxdepth: 1
   :hidden:

   component
   base_data_class

Core Components
-------------------
.. In the core, lies our ``Generator``, it orchestrates three components: (1) Model Client, (2) Prompt, and (3) Output Processors.
  
.. Assisted with ``DataClass`` for input and output data formating.

.. With generator being in the center, all things are built around it via the prompt.
.. - Retriever  (Enhance Generator to be more factual and less hallucination), provide `context_str` in prompt.
.. - 

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Component
     - Description
   * - :doc:`prompt`
     - Built on ``jinja2``, it programmablly and flexibly help users format prompt as input to the generator.
   * - :doc:`model_client`
     - ``ModelClient`` is the protocol and base class for all Models to communicate with components, either via APIs or local models.
   * - :doc:`generator`
     - The core component that orchestrates the model client(LLMs in particular), prompt, and output processors.
   * - :doc:`embedder`
     - The component that orchestrates model client (Embedding models in particular) and output processors.
   * - :doc:`retriever`
     - The base class for all retrievers who in particular retrieve documents from a given database.


.. toctree::
   :maxdepth: 1
   :hidden:

   prompt
   model_client
   generator
   embedder
   retriever

Core functionals
-------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Functional
     - Description
   * - :doc:`string_parser`
     - Parse the output string to structured data.
   * - :doc:`tool_helper`
     - Provide tools to interact with the generator.
   * - :doc:`document_splitter`
     - For embedder and retriever to split the document.
   * - :doc:`memory`
     - Store the history of the conversation.



.. toctree::
   :maxdepth: 1


Advanced Components
-------------------
- Agent (Enhance Generator with tools, planning, and reasoning)

.. toctree::
   :maxdepth: 1

  agent
  react_agent_xy
   
 

Optimizing
=============================

Datasets and Evaulation 

.. toctree::
   :maxdepth: 1

   evaluation


Optimizer & Trainer

.. toctree::
   :maxdepth: 1

   parameter

   optimizer
   trainer


Logging & Tracing
=============================


.. toctree::
   :maxdepth: 1

   logging_tracing