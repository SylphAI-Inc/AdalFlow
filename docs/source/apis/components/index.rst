.. _apis-components:

Components
==============

The components section of the LightRAG API documentation outlines the detailed specifications and functionalities of various API components. Each component plays a crucial role in the LightRAG framework, providing specialized capabilities and interactions.

Overview
----------
.. autosummary::

   components.agent.react

   components.model_client.anthropic_client
   components.model_client.cohere_client
   components.model_client.google_client
   components.model_client.groq_client
   components.model_client.openai_client
   components.model_client.transformers_client
   components.model_client.utils

   components.data_process.data_components
   components.data_process.text_splitter

   components.reasoning.chain_of_thought

   components.retriever.bm25_retriever
   components.retriever.faiss_retriever
   components.retriever.llm_retriever
   components.retriever.postgres_retriever
   components.retriever.reranker_retriever

   components.output_parsers.outputs


Output Parsers
--------------
.. toctree::
   :maxdepth: 1

   components.output_parsers

Agents
------
.. toctree::
   :maxdepth: 1

   components.agent

Model Clients
-----------------
.. toctree::
   :maxdepth: 1

   components.model_client

Data Process
----------------
.. toctree::
   :maxdepth: 1

   components.data_process

.. Embedders
.. ---------
.. .. toctree::
..    :maxdepth: 1

..    components.embedder

.. Reasoners
.. ---------
.. .. toctree::
..    :maxdepth: 1

..    components.reasoning

Retrievers
----------
.. toctree::
   :maxdepth: 1

   components.retriever
