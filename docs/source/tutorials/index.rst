.. _source-tutorials:

.. _developer_notes:


Developer Notes
=============================

.. *Why and How Each Part works*

Learn the `why` and `how-to` (customize and integrate) behind each core part within the `AdalFlow` library.
These are our most important tutorials before you move ahead to build your use cases end to end.


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


The `AdalFlow` library focuses on providing building blocks for developers to **build** and **optimize** the task pipeline.
We have a clear :doc:`lightrag_design_philosophy`, which results in this :doc:`class_hierarchy`.

.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   lightrag_design_philosophy
   class_hierarchy
   trace_graph


Introduction
-------------------


:ref:`Component<core-component>` is to LLM task pipelines what `nn.Module` is to PyTorch models.
An LLM task pipeline in AdalFlow mainly consists of components, such as a `Prompt`, `ModelClient`, `Generator`, `Retriever`, `Agent`, or any other custom components.
This pipeline can be `Sequential` or a Directed Acyclic Graph (DAG) of components.
A `Prompt` will work with `DataClass` to ease data interaction with the LLM model.
A `Retriever` will work with databases to retrieve context and overcome the hallucination and knowledge limitations of LLM, following the paradigm of Retrieval-Augmented Generation (RAG).
An `Agent` will work with tools and an LLM planner for enhanced ability to reason, plan, and act on real-world tasks.


Additionally, what shines in AdalFlow is that all orchestrator components, like `Retriever`, `Embedder`, `Generator`, and `Agent`, are model-agnostic.
You can easily make each component work with different models from different providers by switching out the `ModelClient` and its `model_kwargs`.


We will introduce the library starting from the core base classes, then move to the RAG essentials, and finally to the agent essentials.
With these building blocks, we will further introduce optimizing, where the optimizer uses building blocks such as Generator for auto-prompting and retriever for dynamic few-shot in-context learning (ICL).

Building
-------------------



Base classes
~~~~~~~~~~~~~~~~~~~~~~
Code path: :ref:`adalflow.core <apis-core>`.


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


Code path: :ref:`adalflow.core<apis-core>`. For abstract classes:

- ``ModelClient``: the functional subclass is in :ref:`adalflow.components.model_client<components-model_client>`.
- ``Retriever``: the functional subclass is in :ref:`adalflow.components.retriever<components-retriever>`.


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`prompt`
     - Built on `jinja2`, it programmatically and flexibly formats prompts as input to the generator.
   * - :doc:`model_client`
     - The standard `protocol` to intergrate LLMs, Embedding models, ranking models, etc into respective `orchestrator` components, either via APIs or local to reach to `model agnostic`.
   * - :doc:`generator`
     - The `orchestrator` for LLM prediction. It streamlines three components: `ModelClient`, `Prompt`, and `output_processors` and works with optimizer for prompt optimization.
   * - :doc:`output_parsers`
     - The `interpreter` of the LLM output. The component that parses the output string to structured data.
   * - :doc:`embedder`
     - The component that orchestrates model client (Embedding models in particular) and output processors.
   * - :doc:`retriever`
     - The base class for all retrievers, which in particular retrieve relevant documents from a given database to add *context* to the generator.

Data Pipeline and Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Processing: including transformer, pipeline, and storage. Code path: ``adalflow.components.data_process``, ``adalflow.core.db``, and ``adalflow.database``.
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
.. Code path: ``adalflow.core``

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








Optimization
-------------------
AdalFlow auto-optimization provides a powerful and unified framework to optimize every single part of the prompt: (1) instruction, (2) few-shot examples, and (3) the prompt template,
for any task pipeline you have just built. We leverage all SOTA prompt optimization from Dspy, Text-grad, ORPO, to our own research in the library.


..  covers: (1) simple prompt optimization, (2) few-shot examples, (3) the powerful and general textual auto-diff optimizer that can be applied to both LLM prediction and the prompts/system instructions.

The optimization requires users to have at least one dataset, an evaluator, and define optimizor to use.
This section we will briefly cover the datasets and evaluation metrics supported in the library.


Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can not optimize what you can not meature.
In this section, we provide a general guide to the evaluation datasets, metrics, and methods to productionize your LLM tasks and to publish your research.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`evaluation`
     - A quick guide to the evaluation datasets, metrics, and methods.
   * - :doc:`datasets`
     - How to load and use the datasets in the library.

.. toctree::
   :maxdepth: 1
   :caption: Evaluating
   :hidden:

   evaluation
   datasets


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code path: ``adalflow.optim``.

Adalflow defines four important classes for auto-optimization: (1) ``Parameter``, similar to role of ``nn.Tensor`` in PyTorch,
(2) ``Optimizer`` wh, (3) ``AdalComponent`` to define the training and validation steps, and (4) ``Trainer`` to run the training and validation steps on either data loaders or datasets.

We will first introduce these classes, from their design to important features each class provides.

Classes
^^^^^^^^^^^^^^^^^^
Note: Documentation is work in progress for this section.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`parameter_`
     - The `Parameter` class stores the text, textual gradidents(feedback), and manage the states and applies the backpropagation in auto-diff.
   * - :doc:`optimizer_`
     - The  `Optimizer` to define a structure and to manage `propose`, `revert`, and `step` methods. We defined two variants: `DemoOptimizer` and `TextOptimizer` to cover the prompt optimization and the few-shot optimization.
   * - :doc:`few_shot_optimizer_`
     - Subclassed from ``DemoOptimizer``, the few-shot optimizer to optimize the few-shot in-context learning.
   * - :doc:`auto_text_grad_`
     - Subclassed from ``TextOptimizer``, Auto textual gradient for prompt optimization. It is the most capable and general optimizer in the library to optimize instructions or generator output.
   * - :doc:`adalcomponent_`
     - The ``intepreter`` between task pipeline and the trainer, defining train, validate steps, optimizers, evaluator, loss function, and backward engine.
   * - :doc:`trainer_`
     - The ``Trainer`` will take the ``AdalComponent`` and run the training and validation steps on either data loaders or datasets.

.. toctree::
   :maxdepth: 1
   :caption: Training - Classes
   :hidden:

  ..  parameter
  ..  optimizer
  ..  few_shot_optimizer
  ..  auto_text_grad
  ..  adalcomponent
  ..  trainer

   trace_graph






Logging & Tracing
------------------------------------
Code path:  :ref:`adalflow.utils <apis-utils>` and :ref:`adalflow.tracing <apis-tracing>`.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`logging`
     - AdalFlow uses native ``logging`` module as the first line of debugging tooling. We made the effort to help you set it up easily.
   * - :doc:`logging_tracing`
     - We provide two tracing methods to help you develop and improve the Generator:
       1. Trace the history change(states) on prompt during your development process.
       2. Trace all failed LLM predictions in a unified file for further improvement.

.. toctree::
   :maxdepth: 1
   :caption: Logging & Tracing
   :hidden:


   logging
   logging_tracing



.. Configurations
.. -------------------
.. Code path:  :ref:`adalflow.utils <apis-utils>`.

..    :widths: 20 80
..    :header-rows: 1

..    * - Part
..      - Description
..    * - :doc:`configs`
..      - The configurations for the components.



..    :maxdepth: 1
..    :caption: Configurations
..    :hidden:


..    configs
