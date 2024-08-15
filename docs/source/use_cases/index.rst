.. _use_cases:

Use Cases
=============================

  How different parts are used to build and to auto-optimize various LLM applications.


We will build use cases end-to-end, ranging from classification (classical NLP tasks) to question answering, retrieval-augmented generation (RAG), and multi-generator pipelines.


.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`question_answering`
     - Question Answering with `bhh_hard_object_count` dataset, including textual-gradient descent and few-shot boostrap optimization.
   * - :doc:`classification`
     - Classification with llama3.1-8b model and dataset (coming soon).
   * - :doc:`rag_opt`
     - RAG and multi-hop question answering with hotpotqa dataset, two generators, and one retriever, optimizing zero-shot and few-shot learning (coming soon).


.. toctree::
   :maxdepth: 1
   :caption:
   :hidden:

   question_answering
   classification
   rag_opt




.. :maxdepth: 2

.. eval_a_rag
.. introduction_to_basedataclass
