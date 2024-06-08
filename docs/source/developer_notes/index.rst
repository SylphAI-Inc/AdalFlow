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


.. toctree::
   :maxdepth: 1

   data_classes



With generator being in the center, all things are built around it via the prompt.


Core Components - Generator

Model cliens are for both

.. toctree::
   :maxdepth: 1

   api_client
   component
   prompt
   generator
   optimizer

1. string processing and output parser  (helper for Generator)



Core Components - Retriever  (Enhance Generator to be more factual and less hallucination)

.. toctree::
   :maxdepth: 1

   embedder
   retriever


Core Components - Agent (Enhance Generator with tools, planning, and reasoning)

tools and execturos, react and how react relates to generator.

.. toctree::
   :maxdepth: 1

   model


Like enhance its memoty! etc
 

Datasets and Evaulation 

.. toctree::
   :maxdepth: 1

   evaluation


Optimizer & Trainer


Debugging & Tracing