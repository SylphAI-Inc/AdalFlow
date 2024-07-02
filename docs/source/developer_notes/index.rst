.. _developer_notes:


Tutorials
=============================

.. *Why and How Each Part works*

Learn the `why` and `how-to` (customize and integrate) behind each core part within the `LightRAG` library.
These are our most important tutorials before you move ahead to build use cases  (LLM applications) end to end.


.. raw::

  .. note::

    You can read interchangably between :ref:`Use Cases <use_cases>`.



.. figure:: /_static/images/LLM_arch.png
   :alt: LLM application is no different from a mode training/eval workflow
   :align: center
   :width: 600px

   LLM application is no different from a mode training/evaluation workflow

   .. :height: 100px
   .. :width: 200px


The `LightRAG` library focuses on providing building blocks for developers to **build** and **optimize** the task pipeline.
We have a clear :doc:`lightrag_design_philosophy`, which results in this :doc:`class_hierarchy`.

.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   lightrag_design_philosophy
   class_hierarchy




Building
-------------------
Base classes
~~~~~~~~~~~~~~~~~~~~~~
Code path: :ref:`lightrag.core <apis-core>`.


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Base Class
     - Description
   * - :doc:`component`
     - The building block for task pipeline. It standardizes the interface of all components with `call`, `acall`, and `__call__` methods, handles state serialization, nested components, and parameters for optimization. Components can be easily chained together via ``Sequential``.
   * - :doc:`base_data_class`
     - The base class for data. It eases the data interaction with LLMs for both prompt formatting and output parsing.




.. create side bar navigation

.. toctree::
   :maxdepth: 1
   :caption: Base Classes
   :hidden:

   component
   base_data_class

RAG Essentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RAG components
^^^^^^^^^^^^^^^^^^^


Code path: :ref:`lightrag.core<apis-core>`. For abstract classes:

- ``ModelClient``: the functional subclass is in :ref:`lightrag.components.model_client<components-model_client>`.
- ``Retriever``: the functional subclass is in :ref:`lightrag.components.retriever<components-retriever>`.


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`prompt`
     - Built on `jinja2`, it programmatically and flexibly formats prompts as input to the generator.
   * - :doc:`model_client`
     - ``ModelClient`` is the protocol and base class for LightRAG to **integrate all models**, either APIs or local, LLMs or Embedding models or any others.
   * - :doc:`generator`
     - The orchestrator for LLM prediction. It streamlines three components: `ModelClient`, `Prompt`, and `output_processors` and works with optimizer for prompt optimization.
     - The **center component** that orchestrates the model client(LLMs in particular), prompt, and output processors for format parsing or any post processing.
   * - :doc:`output_parsers`
     - The component that parses the output string to structured data.
   * - :doc:`embedder`
     - The component that orchestrates model client (Embedding models in particular) and output processors.
   * - :doc:`retriever`
     - The base class for all retrievers who in particular retrieve relevant documents from a given database to add **context** to the generator.

Data Pipeline and Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Processing: including transformer, pipeline, and storage. Code path: ``lightrag.components.data_process``, ``lightrag.core.db``, and ``lightrag.database``.
Components work on a sequence of ``Document`` and return a sequence of ``Document``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`text_splitter`
     - To split long text into smaller chunks to fit into the token limits of embedder and generator or to ensure more relevant context while being used in RAG.
   * - :doc:`db`
     - Understanding the **data modeling, processing, and storage** as a whole. We will build a chatbot with enhanced memory and memoy retrieval in this note (RAG).


..  * - :doc:`data_pipeline`
..    - The pipeline to process data, including text splitting, embedding, and retrieval.

.. Let us put all of these components together to build a :doc:`rag` (Retrieval Augmented Generation), which requires data processing pipeline along with a task pipeline to run user queries.

.. toctree::
   :maxdepth: 1
   :caption: RAG Essentials
   :hidden:

   prompt
   model_client
   generator
   output_parsers
   embedder
   retriever
   text_splitter
   db



Agent Essentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent in ``components.agent`` is LLM great with reasoning, planning, and using tools to interact and accomplish tasks.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`tool_helper`
     - Provide tools (function calls) to interact with the generator.
   * - :doc:`agent`
     - The ReactAgent.

.. toctree::
    :maxdepth: 1
    :caption: Agent Essentials
    :hidden:

    tool_helper
    agent

.. Core functionals
.. -------------------
.. Code path: ``lightrag.core``

..    :widths: 20 80
..    :header-rows: 1

..    * - Functional
..      - Description
..    * - :doc:`string_parser`
..      - Parse the output string to structured data.
..    * - :doc:`tool_helper`
..      - Provide tools to interact with the generator.

..    * - :doc:`memory`
..      - Store the history of the conversation.








Optimizing
-------------------

Datasets and Evaulation

.. toctree::
   :maxdepth: 1
   :caption: Datasets and Evaulation


   datasets

   evaluation


Optimizer & Trainer

.. toctree::
   :maxdepth: 1
   :caption: Optimizer & Trainer

   parameter

   optimizer
   trainer


Logging & Tracing & Configurations
------------------------------------
Code path: ``lightrag.utils``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`logging`
     - LightRAG uses ``logging`` module as the first defense line to help users debug the code. We made the effort to help you set it up easily.

.. toctree::
   :maxdepth: 1
   :caption: Logging & Tracing & Configurations
   :hidden:


   logging
   logging_tracing
   configs
