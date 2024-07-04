.. _apis-components:

Components
==============

.. The components section of the LightRAG API documentation outlines the detailed specifications and functionalities of various API components. Each component plays a crucial role in the LightRAG framework, providing specialized capabilities and interactions.


ModelClient
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   components.model_client.anthropic_client

   components.model_client.cohere_client

   components.model_client.google_client

   components.model_client.groq_client

   components.model_client.openai_client

   components.model_client.transformers_client

   components.model_client.utils



Retriever
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   components.retriever.bm25_retriever
   components.retriever.faiss_retriever
   components.retriever.llm_retriever

   components.retriever.postgres_retriever

   components.retriever.reranker_retriever


Output Parsers
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   components.output_parsers.outputs

Agent
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   components.agent.react

Data Process
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:


   components.data_process.text_splitter

   components.data_process.data_components

Memory
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   components.memory.memory

Reasoning
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   components.reasoning.chain_of_thought


.. toctree::
   :maxdepth: 2
   :hidden:

   components.model_client
   components.retriever
   components.output_parsers
   components.agent
   components.data_process
   components.reasoning
   components.memory
