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

RAG Playbook
================




In this playbook, we will provide a comprehensive RAG playbook according the sota research and the best practices in the industry.
The outline of the playbook is as follows:

- RAG Overview
- From First RAG Paper to the diverse RAG design architecture
- RAG design and tuning strategies for each component


RAG Overview
----------------

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

**First RAG Paper**

RAG was introduced in 2020 by Lewis et al. [1]_ which is an architecture that finetunes both the query encoder (bi-encoder like most embedding models) and the generator (LLM) jointly with only final answer supervision.
It did not mention document chunking as most of the time, their text length is usally short and also fits into the context length of the embedding models.
As both the embedding model and LLM model scales up in terms of knowledge and parameters (400M LLM model used in the paper), RAG can achieve high performance in few-shot (prompt engineering) setup without the finetune.


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

**RAU (Retrieval Augmented Understanding)**

There is also RAU.

Designing RAG
----------------------------------

========================  ==============================  =========================================
RAG Component              Techniques                      Metrics
========================  ==============================  =========================================
Data Preparation           - Text preprocessing
                           - Chunking Strategy

Data Storage               - AdalFlow LocalDB
                           - Cloud Database
                           - Postgres + PgVector
                           - qdrant
                           - ...

Embedding                  - Embedding Fine-tuning

Indexing                   -

Retrieval                  - Retrieval Optimization          - HIT@K
                           - Query Enhancement               - MRR@K
                           - Reranking                       - MAP@K
                                                             - NDCG@K
                                                             - AdalFlow context recall
                                                             - Ragas context relevancy, precision, recall

Generator                  - Manual Prompt Engineering       - Ragas answer relevancy
                           - Auto Prompt Engineering         - ROUGE
                           - LLM Fine-tuning                 - BLEU
                                                             - METEOR
                                                             - F1 Score
                                                             - BERTScore
                                                             - AdalFlow AnswerMatchAcc
                                                             - AdalFlow LLM judge
                                                             - AdalFlow G-Eval
                                                             - UniEval
========================  ==============================  =========================================

TODO: make this a table that i can put in links. so that i can link together other tutorials to form a comprehensive playbook.
- move this in the tutorial section.

For benchmarking datasets and metrics, please refer to :ref:`Evaluation Guideline <tutorials-llm-evaluation>`.
Additionally, FlashRAG [3]_ provides more references to RAG datasets and research.


Data Preparation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Document Retrieval & Reranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-stage retrieval from the cheapest, fastest, and least accurate to the most expensive, slowest, and most accurate is introduced in :ref:`Retriever <tutorials-retriever>`.


Query Expansion
~~~~~~~~~~~~~~~~~~~~~~~


References
------------------------------------------
.. [1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:https://arxiv.org/abs/2005.11401
.. [2] GOVTech Singapore's RAG playbook: https://playbooks.capdev.govtext.gov.sg/improving_rag/
.. [3] FlashRAG: Python toolkit for the reproduction and development of RAG research: https://github.com/RUC-NLPIR/FlashRAG
.. [4] RAG and RAU: A Survey on Retrieval-Augmented Language Model inNatural Language Processing: https://github.com/2471023025/RALM_Survey
.. [5] Ruochen Zhao, Hailin Chen, Weishi Wang, FangkaiJiao, Xuan Long Do, Chengwei Qin, BoshengDing, Xiaobao Guo, Minzhi Li, Xingxuan Li, et al.2023. Retrieving multimodal information for aug-mented generation: A survey. arXiv preprintarXiv:2303.10868.
