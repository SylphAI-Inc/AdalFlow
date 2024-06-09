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
There are two base classes, ``Component`` and ``DataClass``.
- ``Component`` plays the same role as ``Module`` in PyTorch. It standardize the interface of all componnets with `call`, `acall`, and `__call__` methods, handles states, and serialization. 
Components can be easily chained togehter via `Sequential` for now.
- ``DataClass``, leverages the ``dataclasses`` module in Python to ease how data is interacting each component and especially how to formatted as string in Prompt, beside of the serialization.


.. toctree::
   :maxdepth: 1

   component
   base_data_class

Core Components and functional
-------------------
In the core, lies our ``Generator``.
- Model Client  
- Prompt
- Output Parser
  
Assisted with ``DataClass`` for input and output data formating.

With generator being in the center, all things are built around it via the prompt.
- Retriever  (Enhance Generator to be more factual and less hallucination), provide `context_str` in prompt.
- 


.. toctree::
   :maxdepth: 1

   api_client
   prompt
   generator
   embedder
   retriever


.. Core Components - Agent (Enhance Generator with tools, planning, and reasoning)

tools and execturos, react and how react relates to generator.

Core functional:

- string_parser (parse the output string to structured data)
- tool_helper (provide tools to interact with the generator)
- memory (store the history of the conversation)

.. toctree::
   :maxdepth: 1

   
 

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