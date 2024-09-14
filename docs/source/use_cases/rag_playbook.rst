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

.. code-block:: python

    RAG_PROMPT_TEMPLATE = r"""<START_OF_SYSTEM_MESSAGE>
    {{task_desc}}
    <END_OF_SYSTEM_MESSAGE>
    <START_OF_USER>
    {{input_str}}
    {{context_str}}
    <END_OF_USER>
    """



Retrieval-Augmented Generation (RAG) is a paradigm that combines the strengths of retrieval and generation models.
Given a user query, RAG retrieves relevant passages from a large corpus and then generates a response based on the retrieved passages.
This formulation opens up a wide range of use cases such as conversational search engine, question answering on a customized knowledge base,
customer support, fact-checking.
RAGs eliminate the hallucination and offers a degree of transparency and interpretability via citing the sources.

The template above shows the most commonly used format of RAG, where we pass a task description, concatenate the input string, and retrieve passages into a context string, which is then passed to an LLM model for generation.
But, RAG is way more than that. Let's dive in.

**First RAG Paper**

RAG was introduced in 2020 by Lewis et al. [1]_ which is an architecture that finetunes both the query encoder (bi-encoder like most embedding models) and the generator (LLM) jointly with only final answer supervision.
It did not mention document chunking as most of the time, their text length is usally short and also fits into the context length of the embedding models.
As both the embedding model and LLM model scales up in terms of knowledge and parameters (400M LLM model used in the paper), RAG can achieve high performance in few-shot (prompt engineering) setup without the finetune.


However, the flexibility of the RAG also means that it requires careful design and tuning to achieve optimal performance.
For each use case, we need to answer:

1. What retrieval to use? And how many stages it should be? Do we need a reranker or even LLM to help with the retrieval stages?

2. Which cloud-database can go well with the retrieval strategy and be able to scale?

3. How do I evaluate the performance of the RAG as a whole? And what metrics can help me understand the retrieval stage in particular so that I know it is not hurting the overall performance?

4. Do I need query expansion or any other techniques to improve the retrieval performance? How to avoid the performance degradation due to feeding the LLM irrelevant passages?

5. How do I optimize the RAG hyperparameters such as the number of retrieved passages, the size of the chunk, and the overlap between chunks, or even the chunking strategy?

6. Sometimes you need to even create your own customized/finetuned embedding/retriever models. How do I do that?

7. How do I auto-optimize the RAG pipeline with In-context learning(ICLs) with zero-shot prompting and few-shot prompting?

8. What about finetuning? How to do it and would it be more token efficient or more effective?

**RAU (Retrieval Augmented Understanding)**

There is also RAU.

Designing RAG
----------------------------------

.. figure:: /_static/images/RAG_Enhancements.png
    :align: center
    :alt: RAG Enhancements
    :width: 700px

    RAG Enhancements from [8]_. Click to view the full image.

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

RAG optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most common optimization is the hyperparameters such as the chunking size, overlap, and the number of retrieved passages for the final generation.

We can either optimize each component such as retriever or the generator, or optimize them jointly.
Sometimes we can use an agentic approach, such as Self-RAG [11]_.

For generator, we have three options:


1. Prompt-engineering: use zero-shot or few-shot learning to optimize the generator, or improve the generator response via more test-time tokens (e.g., chain-of-thought, reflection).

2. Finetune the generator.

Retrieval optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As irrelevant passages, especially those positioned on top of the context can degrade the final performance, it is important to both measure and optimize the retrieval performance.
We categorize the retrieval performance optimization into four categories:

1. Query Expansion
2. Adapt the embedder with LLM supervision
3. Reranking
4. Use Retrieval Evaluator

**Query Expansion**


To improve the Recall, Query Expansion can be useful. We do not need to finetune and can use another LLM to generate the expanded query.

**Adapt the embedder with LLM supervision**

To improve the retrieval recall and precision, we can adapt the embedder with LLM supervision.
The cheapest solutions requires only a linear layer on top of the embedding model along with a synthesized dataset of query-passage pairs generated from the data source using LLM models.
This approach also applys to black-box embedding models. AdalFlow will consider to open-source this technique in the future.

# TODO: replug is not as good as the emsemble is a bit hard to do and no source code.

A second approach is to finetune the embedder directly. Replug [6]_ is a good example of this approach.
Replug can be used with or without finetune.

.. figure:: /_static/images/replug.png
    :align: center
    :alt: Replug inference pipeline
    :width: 700px

    Replug inference pipeline [6]_.

When we do Replug, it computes the LLM output of each query and document pair separately in parallel and ensembles all the outputs to get the final score.
This is especially helpful for inference speed and surpass the context length limitation of the LLM model.

..
    REPLUG LSR (REPLUGwith LM-Supervised Retrieval), which adapts the retrieverin REPLUG by using the LM itself to provide supervisionabout which documents should be retrieved.
    This approach can be seen as adjusting the probabilities of the retrieved documents to match the probabilities of the output sequence perplexi-ties of the language model.
    In theory, it is to align the retriever's probabilities likelihood on the retrieved passage with the probabilitie likelihood of the LLM model on the ground truth answer via KL-divergence.
    This use the `logprobs` of the black-box LLM model. Read the more on logprob cookbook [9]_.

.. The above replug lsr is not that more effective than the replug itself and it is meaningless to go through the hassle of implementing it.

**Reranking**


Rerankers are often cross-encoder between the query and documents. It is computationally more expensive but also more accurate. Cohere and Transformers both offer sota rerankers.

**Use Retrieval Evaluator**

C-RAG [10]_ proposed a lightweight retrieval evaluator that was finetuned on the training split of the testing datasets.
More expensively, but without the need to train a model, we can use  LLM to classify the relevance of the retrieved passages, using labels such as "correct", "incorrect", "ambiguous", etc.

Generator optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Self-RAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/images/self_rag.png
    :align: center
    :alt: Self-RAG architecture
    :width: 700px

    Self-RAG architecture [11]_.


Self-RAG is interesting as it is programmed to decide if retrieval is needed, it handles the retrieved passages separately in parallel to generate y_t for each query x and passage d_t.
For each (x, d_t, y_t) pair it "reflects" on three metrics:

- ISREL: use (x, d_t) to check if d_t provides useful information to solve x by outputing two labels (is_relevant, is_irrelevant).
- ISSUP: use (x, d_t, y_t) to check if all of the worthy statements(answers the question) in y_t is supported by d_t by outputing three labels (is_supported, partically_supported, not_supported).
- ISUSE: use (x, y_t) to check if y_t is useful to solve x by outputing 5 labels (5, 4, 3, 2, 1).

It computes a single segment score unifying the three metrics and uses it to rerank the answer and pick the answer with the highest score as the final answer.
The paper also mentioned how to create synthesized training dataset and train the `critic` and `generator` model.
Good thing is Self-RAG can be used with or without finetune.

Self-RAG can be applied on complicated tasks that require high accuracy, but it is way more complicated than a vanila RAG.




References
------------------------------------------
.. [1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:https://arxiv.org/abs/2005.11401
.. [2] GOVTech Singapore's RAG playbook: https://playbooks.capdev.govtext.gov.sg/improving_rag/
.. [3] FlashRAG: Python toolkit for the reproduction and development of RAG research: https://github.com/RUC-NLPIR/FlashRAG
.. [4] RAG and RAU: A Survey on Retrieval-Augmented Language Model inNatural Language Processing: https://github.com/2471023025/RALM_Survey
.. [5] Ruochen Zhao, Hailin Chen, Weishi Wang, FangkaiJiao, Xuan Long Do, Chengwei Qin, BoshengDing, Xiaobao Guo, Minzhi Li, Xingxuan Li, et al.2023. Retrieving multimodal information for aug-mented generation: A survey. arXiv preprintarXiv:2303.10868.
.. [6] Replug: Retrieval-augmented black-box language models. arXivpreprint arXiv:2301.12652
.. [7]
.. [8] Retrieval-Augmented Generation for AI-Generated Content: A Survey
.. [9] OpenAI logprobs cookbook: https://cookbook.openai.com/examples/using_logprobs
.. [10] C-RAG: Corrective retrieval augmented generation.arXiv preprint arXiv:2401.15884.
.. [11] Self-RAG: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, andHannaneh Hajishirzi. 2023. Self-rag: Learning toretrieve, generate, and critique through self-reflection.CoRR, abs/2310.11511.
.. [12] Replug implemented: https://github.com/IntelLabs/fastRAG/blob/main/examples/replug_parallel_reader.ipynb
.. [13] FastRAG: https://github.com/IntelLabs/fastRAG



..
    TODO:
     - replug generator implementation(current fast rag implemented it with haystack)
     - self-RAG implementation (raw response, and we might need to add a api response to save logprobs and everything that user can customize)
     - opensource the embedder finetune.
     - extend: all these research can be provided as extend and we need to think of a way to organize it.
