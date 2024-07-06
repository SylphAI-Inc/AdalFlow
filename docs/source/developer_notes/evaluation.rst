LLM Evaluation
====================================

.. .. admonition:: Author
..    :class: highlight

..    `Meng Liu <https://github.com/mengliu1998>`_

"You cannot improve what you cannot measure". This is especially true in the context of LLMs, which have become increasingly popular due to their impressive performance on a wide range of tasks. Evaluating LLMs and their applications is crucial in both research and production to understand their capabilities and limitations.
Overall, such evaluation is a complex and multifaceted process. Below, we provide a guideline for evaluating LLMs and their applications, incorporating aspects outlined by *Chang et al.* [1]_:

* **What to evaluate**: the tasks and capabilities that LLMs are evaluated on.
* **Where to evaluate**: the datasets and benchmarks that are used for evaluation.
* **How to evaluate**: the protocols and metrics that are used for evaluation.


What to evaluate?
------------------------------------------
When we are considering the LLM evaluation, the first question that arises is what to evaluate. Deciding what tasks to evaluate or which capabilities to assess is crucial, as it influences both the selection of appropriate benchmarks (where to evaluate) and the choice of evaluation methods (how to evaluate). Below are some commonly evaluated tasks and capabilities of LLMs:

* *Natural language understanding* (NLU) tasks, such as text classification and sentiment analysis, which evaluate the LLM's ability to understand natural language.
* *Natural language generation* (NLG) tasks, such as text summarization, translation, and question answering, which evaluate the LLM's ability to generate natural language.
* *Reasoning* tasks, such as mathematical, logic, and common-sense reasoning, which evaluate the LLM's ability to perform reasoning and inference to obtain the correct answer.
* *Robustness*, which evaluate the LLM's ability to generalize to unexpected inputs.
* *Fairness*, which evaluate the LLM's ability to make unbiased decisions.
* *Domain adaptation*, which evaluate the LLM's ability to adapt from general language to specific new domains, such as medical or legal texts, coding, etc.
* *Agent applications*, which evaluate the LLM's ability to use external tools and APIs to perform tasks, such as web search.

For a more detailed and comprehensive description of the tasks and capabilities that LLMs are evaluated on, please refer to the review papers by *Chang et al.* [1]_ and *Guo et al.* [2]_.

Where to evaluate?
------------------------------------------
Once we have decided what to evaluate, the next question is where to evaluate. The selection of datasets and benchmarks is important, as it determines the quality and relevance of the evaluation.

To comprehensively assess the capabilities of LLMs, researchers typically utilize benchmarks and datasets that span a broad spectrum of tasks. For example, in the GPT-4 technical report [3]_, the authors employed a variety of general language benchmarks, such as MMLU [4]_, and academic exams, such as the SAT, GRE, and AP courses, to evaluate the diverse capabilities of GPT-4. Below are some commonly used datasets and benchmarks for evaluating LLMs.

* *MMLU* [4]_, which evaluates the LLM's ability to perform a wide range of language understanding tasks.
* *HumanEval* [5]_, which measures the LLM's capability in writing Python code.
* `HELM <https://crfm.stanford.edu/helm/>`_, which evaluates LLMs across diverse aspects such as language understanding, generation, common-sense reasoning, and domain adaptation.
* `Chatbot Arena <https://arena.lmsys.org/>`_, which is an open platform to evaluate LLMs through human voting.
* `API-Bank <https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank>`_, which evaluates LLMs' ability to use external tools and APIs to perform tasks.

Please refer to the review papers (*Chang et al.* [1]_, *Guo et al.* [2]_, and *Liu et al.* [6]_) for a more comprehensive overview of the datasets and benchmarks used in LLM evaluations. Additionally, a lot of datasets are readily accessible via the `Hugging Face Datasets <https://huggingface.co/datasets>`_ library. For instance, the MMLU dataset can be easily loaded from the Hub using the following code snippet.

.. code-block:: python
    :linenos:

    from datasets import load_dataset
    dataset = load_dataset(path="cais/mmlu", name='abstract_algebra')
    print(dataset["test"])
    # Dataset({
    # features: ['question', 'subject', 'choices', 'answer'],
    # num_rows: 100
    # })

How to evaluate?
------------------------------------------

The final question is how to evaluate. Evaluation methods can be divided into *automated evaluation* and *human evaluation* (*Chang et al.* [1]_ and *Liu et al.* [6]_). Automated evaluation typically involves using metrics such as accuracy and BERTScore or employing an LLM as the judge, to quantitatively assess the performance of LLMs on specific tasks. Human evaluation, on the other hand, involves human in the loop to evaluate the quality of the generated text or the performance of the LLM. Here, we recommend a few automated evaluation methods that can be used to evaluate LLMs and their applications.

If you are interested in computing metrics such as accuracy, F1-score, ROUGE, BERTScore, perplexity, etc for LLMs and LLM applications, you can check out the metrics provided by `Hugging Face Metrics <https://huggingface.co/metrics>`_ or `TorchMetrics <https://lightning.ai/docs/torchmetrics>`_. For instance, to compute the BERTScore, you can use the corresponding metric function provided by Hugging Face, which uses the pre-trained contextual embeddings from BERT and matched words in generated text and reference text by cosine similarity.

.. code-block:: python
    :linenos:

    from datasets import load_metric
    bertscore = load_metric("bertscore")
    generated_text = ["life is good", "aim for the stars"]
    reference_text = ["life is great", "make it to the moon"]
    results = bertscore.compute(predictions=generated_text, references=reference_text, model_type="distilbert-base-uncased")
    print(results)
    # {'precision': [0.9419728517532349, 0.7959791421890259], 'recall': [0.9419728517532349, 0.7749403119087219], 'f1': [0.9419728517532349, 0.7853187918663025], 'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.38.2)'}

If you are particulay interested in evaluating RAG (Retrieval-Augmented Generation) pipelines, we have several metrics available in LightRAG to assess both the quality of the retrieved context and the quality of the final generated answer.

- :class:`RetrieverRecall <eval.retriever_recall>`: This is used to evaluate the recall of the retriever component of the RAG pipeline.
- :class:`RetrieverRelevance <eval.retriever_relevance>`: This is used to evaluate the relevance of the retrieved context to the query.
- :class:`AnswerMatchAcc <eval.answer_match_acc>`: This calculates the exact match accuracy or fuzzy match accuracy of the generated answers by comparing them to the ground truth answers.
- :class:`LLMasJudge <eval.llm_as_judge>`: This uses an LLM to get the judgement of the generated answer for a list of questions. The task description and the judgement query of the LLM judge can be customized. It computes the judgement score, which is the number of generated answers that are judged as correct by the LLM divided by the total number of generated answers.

For example, you can use the following code snippet to compute the recall and relevance of the retriever component of the RAG pipeline for a single query.

.. code-block:: python
    :linenos:

    from lightrag.eval import RetrieverRecall, RetrieverRelevance
    retrieved_contexts = [
        "Apple is founded before Google.",
        "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year.",
    ]
    gt_contexts = [
        [
            "Apple is founded in 1976.",
            "Google is founded in 1998.",
            "Apple is founded before Google.",
        ],
        ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"],
    ]
    retriever_recall = RetrieverRecall()
    avg_recall, recall_list = retriever_recall.compute(retrieved_contexts, gt_contexts) # Compute the recall of the retriever
    print(f"Recall: {avg_recall}, Recall List: {recall_list}")
    # Recall: 0.6666666666666666, Recall List: [0.3333333333333333, 1.0]
    retriever_relevance = RetrieverRelevance()
    avg_relevance, relevance_list = retriever_relevance.compute(retrieved_contexts, gt_contexts) # Compute the relevance of the retriever
    print(f"Relevance: {avg_relevance}, Relevance List: {relevance_list}")
    # Relevance: 0.803030303030303, Relevance List: [1.0, 0.6060606060606061]

For a more detailed instructions on how build and evaluate RAG pipelines, you can refer to the use case on :doc:`Evaluating a RAG Pipeline <../tutorials/eval_a_rag>`.

If you intent to use metrics that are not available in the LightRAG library, you can also implement your own custom metric functions or use other libraries such as `RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_ to compute the desired metrics for evaluating RAG pipelines.


.. [1] Chang, Yupeng, et al. "A survey on evaluation of large language models." ACM Transactions on Intelligent Systems and Technology 15.3 (2024): 1-45.
.. [2] Guo, Zishan, et al. "Evaluating large language models: A comprehensive survey." arXiv preprint arXiv:2310.19736 (2023).
.. [3] Achiam, Josh, et al. "GPT-4 technical report." arXiv preprint arXiv:2303.08774 (2023).
.. [4] Hendrycks, Dan, et al. "Measuring massive multitask language understanding." International Conference on Learning Representations. 2020.
.. [5] Chen, Mark, et al. "Evaluating large language models trained on code." arXiv preprint arXiv:2107.03374 (2021).
.. [6] Liu, Yang, et al. "Datasets for Large Language Models: A Comprehensive Survey." arXiv preprint arXiv:2402.18041 (2024).
