API Reference
=============

Welcome to the LightRAG API reference! This section provides detailed documentation of the internal APIs that make up the LightRAG framework. Explore the APIs to understand how to effectively utilize and integrate LightRAG components into your projects.


Core
----------

The core section of the LightRAG API documentation provides detailed information about the foundational components of the LightRAG system. These components are essential for the basic operations and serve as the building blocks for higher-level functionalities.

.. autosummary::

   core.base_data_class
   core.model_client
   core.component
   core.data_components
   core.db
   core.default_prompt_template
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


Components
-----------

The components section of the LightRAG API documentation outlines the detailed specifications and functionalities of various API components. Each component plays a crucial role in the LightRAG framework, providing specialized capabilities and interactions.

.. autosummary::

   components.agent
   components.model_client
   componnets.data_process
   .. components.reasoning
   
   components.retriever
   components.output_parsers


Evaluation
----------
.. autosummary::

   eval.answer_match_acc
   eval.retriever_recall
   eval.retriever_relevance
   eval.llm_as_judge


Optimizer
----------
.. autosummary::
    :maxdepth: 2

   optim.optimizer
   optim.sampler
   optim.few_shot_optimizer
   optim.llm_augment
   optim.llm_optimizer


Tracing
----------
.. autosummary::

   tracing.decorators
   tracing.generator_state_logger
   tracing.generator_call_logger


Utils
----------
.. autosummary::

   utils.logger
   utils.serialization
   utils.config
   utils.registry
   utils.setup_env


.. toctree::
   :maxdepth: 2
   :hidden:

   core/index


.. toctree::
   :maxdepth: 2
   :hidden:

   components/index


.. toctree::
   :maxdepth: 2
   :hidden:

   optim/index

.. toctree::
   :maxdepth: 2
   :hidden:

   tracing/index

.. toctree::
   :maxdepth: 2
   :hidden:

   eval/index

.. toctree::
   :maxdepth: 2
   :hidden:

   utils/index