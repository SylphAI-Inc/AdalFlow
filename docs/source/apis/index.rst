.. _apis:

API Reference
=============

Welcome to `AdalFlow`.
The API reference is organized by subdirectories.

..  This section provides detailed documentation of the internal APIs that make up the LightRAG framework. Explore the APIs to understand how to effectively utilize and integrate LightRAG components into your projects.


Core
----------
All base/abstract classes, core components like generator, embedder, and basic functions are here.


.. The core section of the LightRAG API documentation provides detailed information about the foundational components of the LightRAG system.
.. These components are essential for the basic operations and serve as the building blocks for higher-level functionalities.

.. autosummary::
   core.component
   core.container
   core.base_data_class
   core.default_prompt_template
   core.model_client
   core.db
   core.functional

   core.generator
   core.string_parser
   core.embedder
   core.retriever

   core.prompt_builder
   core.tokenizer
   core.func_tool
   core.tool_manager
   core.types
   core.parameter

Components
-----------
Functional components like model client, retriever, agent, local data processing, and output parsers are here.

.. The components section of the LightRAG API documentation outlines the detailed specifications and functionalities of various API components.
.. Each component plays a crucial role in the LightRAG framework, providing specialized capabilities and interactions.

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

Datasets
-----------

.. autosummary::

   datasets.big_bench_hard
   datasets.trec
   datasets.hotpot_qa
   datasets.types




Evaluation
------------
.. autosummary::

   eval.base
   eval.answer_match_acc
   eval.retriever_recall
   eval.llm_as_judge
   eval.g_eval


Optimization
--------------
.. autosummary::


   optim.parameter
   optim.optimizer
   optim.grad_component

   optim.types
   optim.function
   optim.few_shot.bootstrap_optimizer
   optim.text_grad.text_loss_with_eval_fn
   optim.text_grad.tgd_optimizer
   optim.text_grad.llm_text_loss
   optim.trainer.trainer
   optim.trainer.adal






Tracing
----------
.. autosummary::

   tracing.decorators
   tracing.generator_state_logger
   tracing.generator_call_logger


Utils
----------
.. autosummary::


   utils.data
   utils.logger
   utils.setup_env
   utils.lazy_import
   utils.serialization
   utils.config
   utils.registry


.. toctree::
   :maxdepth: 2
   :hidden:

   core/index
   components/index
   datasets/index
   eval/index
   optim/index
   tracing/index
   utils/index
