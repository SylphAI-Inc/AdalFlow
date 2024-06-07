.. _developer_notes:


Developer Notes 
=============================

*How each part works*

Learn LightRAG design phisolophy and the reasoning(`why` and `how-to`) behind each core part within the LightRAG library.
This is also our tutorials showing how each part works before we move ahead to build use cases (LLM applications).

.. note::

   You can read interchangably between :ref:`Use Cases <use_cases>`.


TODO: 

1. provide a graph of the LightRAG architecture
2. Put `why` and `how-to` guide by core parts with secondary toctree


With generator being in the center, all things are built around it via the prompt.

.. toctree::
   :maxdepth: 2

   lightrag_design_philosophy

   llm_intro
   data_classes

Core Components - Generator

Model cliens are for both

.. toctree::
   :maxdepth: 2

   api_client
   component
   prompt
   generator
   optimizer

1. string processing and output parser  (helper for Generator)



Core Components - Retriever  (Enhance Generator to be more factual and less hallucination)

.. toctree::
   :maxdepth: 2

   embedder
   retriever


Core Components - Agent (Enhance Generator with tools, planning, and reasoning)

tools and execturos, react and how react relates to generator.

.. toctree::
   :maxdepth: 2

   model


Like enhance its memoty! etc
 

Datasets and Evaulation 

.. toctree::
   :maxdepth: 2

   evaluation


Optimizer & Trainer


Debugging & Tracing