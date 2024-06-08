Core
===================

The core section of the LightRAG API documentation provides detailed information about the foundational components of the LightRAG system. These components are essential for the basic operations and serve as the building blocks for higher-level functionalities.

Overview
----------
.. autosummary::

   core.model_client
   core.component
   core.data_components
   core.db
   core.default_prompt_template
   core.document_splitter
   core.embedder
   core.functional
   core.generator
   core.memory
   core.parameter
   core.prompt_builder
   core.retriever
   core.string_parser
   core.tokenizer
   core.tool_helper
   core.types


Model Client
----------
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
   core.data_components
   core.db

Prompt and Template
---------------------
.. toctree::
   :maxdepth: 1

   core.default_prompt_template
   core.prompt_builder

Document Processing
-------------------
.. toctree::
   :maxdepth: 1

   core.document_splitter

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

Parsing and Tokenization
------------------------
.. toctree::
   :maxdepth: 1

   core.string_parser
   core.tokenizer
   core.tool_helper

Parameters
------------------------
.. toctree::
   :maxdepth: 1

   core.parameter