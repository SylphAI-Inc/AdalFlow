Core
===================

The core section of the LightRAG API documentation provides detailed information about the foundational components of the LightRAG system. These components are essential for the basic operations and serve as the building blocks for higher-level functionalities.

Overview
----------
.. autosummary::

   core.base_data_class  
   core.component
   core.db
   core.default_prompt_template
   core.embedder
   core.functional
   core.generator
   core.memory
   core.model_client
   core.parameter
   core.prompt_builder
   core.retriever
   core.string_parser
   core.tokenizer
   core.func_tool
   core.tool_manager
   core.types


Model Client
---------------
.. toctree::
   :maxdepth: 1

   core.model_client

Component
--------------
.. toctree::
   :maxdepth: 1

   core.component

Data Handling
-------------
.. toctree::
   :maxdepth: 1

   core.base_data_class
   core.types

   core.db

Prompts and Templates
---------------------
.. toctree::
   :maxdepth: 1

   core.default_prompt_template
   core.prompt_builder

.. Document Processing
.. -------------------
.. .. toctree::
..    :maxdepth: 1

   .. core.document_splitter
   core.text_splitter

Embedding and Retrieval
-----------------------
.. toctree::
   :maxdepth: 1

   core.embedder
   core.retriever

Generation and Utilities
------------------------
.. toctree::
   :maxdepth: 1

   core.generator
   core.functional
   core.memory

------------------------
.. toctree::
   :maxdepth: 1

   core.string_parser
   core.tokenizer
   core.func_tool

Parameters
------------------------
.. toctree::
   :maxdepth: 1

   core.parameter
