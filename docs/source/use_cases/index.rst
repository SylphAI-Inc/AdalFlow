.. _use_cases:

Use Cases
=============================

  How different parts are used to build and to auto-optimize various LLM applications.


We will build use cases end-to-end, ranging from classification (classical NLP tasks) to question answering, retrieval-augmented generation (RAG), and multi-generator pipelines.


RAG
----------------
.. list-table::
  :widths: 30 70
  :header-rows: 1

  * - Part
    - Description
  * - :doc:`rag_playbook`
    - Comprehensive RAG playbook according to the sota research and the best practices in the industry.
  * - :doc:`build_a_rag`
    - Designing a RAG pipeline, from offline data processing to online inference.
  * - :doc:`eval_a_rag`
    - Question Answering with `bhh_hard_object_count` dataset, including textual-gradient descent and few-shot boostrap optimization.

.. toctree::
  :maxdepth: 1
  :caption: RAG vibe
  :hidden:

  rag_playbook
  build_a_rag
  eval_a_rag

Optimization
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`question_answering`
     - Question Answering with `bhh_hard_object_count` dataset, including textual-gradient descent and few-shot boostrap optimization.
   * - :doc:`classification`
     - Classification with `gpt-3.5-turbo`. The optimized task pipeline performs on-par with `gpt-4o`.
   * - :doc:`rag_opt`
     - RAG and multi-hop question answering with hotpotqa dataset, two generators, and one retriever, optimizing zero-shot and few-shot learning (coming soon).





.. toctree::
   :maxdepth: 1
   :caption: End-to-End
   :hidden:



   question_answering
   classification
   rag_opt
