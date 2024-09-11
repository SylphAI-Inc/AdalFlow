.. <a href="https://colab.research.google.com/drive/1gmxeX1UuUxZDouWhkLGQYrD4hAdt9IVX?usp=sharing" target="_blank" style="margin-right: 10px;">
..     <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
.. </a>

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">

      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/use_cases/rag/build" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

Designing RAG
================


.. figure:: /_static/images/generator.png
    :align: center
    :alt: AdalFlow generator design
    :width: 700px

    Generator - The Orchestrator for LLM Prediction

Retrieval-Augmented Generation (RAG) is a paradigm that combines the strengths of retrieval and generation models.
Given a user query, RAG retrieves relevant passages from a large corpus and then generates a response based on the retrieved passages.
This formulation opens up a wide range of use cases such as conversational search engine, question answering on a customized knowledge base,
customer support, fact-checking.
RAGs eliminate the hallucination and offers a degree of transparency and interpretability via citing the sources.

However, the flexibility of the RAG also means that it requires careful design and tuning to achieve optimal performance.
For each use case, we need to answer:

1. What retrieval to use? And how many stages it should be? Do we need a reranker or even LLM to help with the retrieval stages?

2. Which cloud-database can go well with the retrieval strategy and be able to scale?

3. How do I evaluate the performance of the RAG as a whole? And what metrics can help me understand the retrieval stage?

4. Do I need query expansion or any other techniques to improve the retrieval performance?

5. How do I optimize the RAG hyperparameters such as the number of retrieved passages, the size of the chunk, and the overlap between chunks, or even the chunking strategy?

6. Sometimes you need to even create your own customized/finetuned embedding models. How do I do that?

7. How do I auto-optimize the RAG pipeline with In-context learning(ICLs) with zero-shot prompting and few-shot prompting?

8. What about finetuning? How to do it and would it be more token efficient or more effective?


References
------------------------------------------
.. [1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:https://arxiv.org/abs/2005.11401
.. [2] RAG playbook: https://playbooks.capdev.govtext.gov.sg/
