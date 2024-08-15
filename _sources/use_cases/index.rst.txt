.. _use_cases:

Use Cases
=============================

How different parts are used to build and auto-optimize various LLM applications.


We will build use cases end to end, from classification (classicial NLP tasks), to question answering, to RAG, to multiple generators pipeline.


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Part
     - Description
   * - :doc:`question_answering`
     - Question Answering with `bhh_hard_object_count` dataset, including zero-shot and few-shot learning.
   * - :doc:`classification`
     - Classification with llama3.1-8b model and dataset .
   * - :doc:`rag_opt`
     - RAG and multi-hop question answering with hotpotqa dataset, two generators, and one retriever, optimizing zero-shot and few-shot learning.


.. toctree::
   :maxdepth: 1
   :caption: Training - Use Cases
   :hidden:

   classification
   question_answering
   rag_opt




.. :maxdepth: 2

.. eval_a_rag
.. introduction_to_basedataclass
