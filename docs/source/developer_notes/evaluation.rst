A Guideline on LLM Evaluation
============

Evaluating LLMs and their applications is crucial for understanding their capabilities and limitations. Overall, such evaluation is a complex and multifaceted process. Below, we provide a guideline for evaluating LLMs and their applications, incorporating aspects outlined by *Chang et al.* [1]_:

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

Please refer to the review papers [1]_, [2]_, [6]_ for a more comprehensive overview of the datasets and benchmarks used in LLM evaluations. Additionally, a lot of datasets are readily accessible via the `Hugging Face Datasets <https://huggingface.co/datasets>`_ library. For instance, the MMLU dataset can be easily loaded from the Hub using the following code snippet.

.. code-block:: python

    from datasets import load_dataset
    dataset = load_dataset(path="mmlu", name='abstract_algebra')
    print(dataset)

How to evaluate?
------------------------------------------

The final question is how to evaluate. Evaluation methods can be divided into *automated evaluation* and *human evaluation* [1]_, [6]_. Automated evaluation typically involves using metrics such as accuracy and BERTScore or employing an LLM as the judge, to quantitatively assess the performance of LLMs on specific tasks. Human evaluation, on the other hand, involves human in the loop to evaluate the quality of the generated text or the performance of the LLM on specific tasks. Here, we recommend a few evaluation methods for LLMs.

If you are interested in computing metrics such as accuracy, F1-score, ROUGE, BERTScore, perplexity, etc for LLMs and LLM applications, you can check out the metrics provided by `Hugging Face Metrics <https://huggingface.co/metrics>`_ or `TorchMetrics <https://lightning.ai/docs/torchmetrics>`_. For instance, you can use the following code snippet to compute the BERTScore, which uses the pre-trained contextual embeddings from BERT and matched words in generated text and reference text by cosine similarity.

.. code-block:: python

    from datasets import load_dataset
    bertscore = load_metric("bertscore")
    generated_text = ["life is good", "aim for the stars"]
    reference_text = ["life is great", "make it to the moon"]
    results = bertscore.compute(predictions=generated_text, references=reference_text, model_type="distilbert-base-uncased")
    print(results)
    {'precision': [0.9419728517532349, 0.7959791421890259], 'recall': [0.9419728517532349, 0.7749403119087219], 'f1': [0.9419728517532349, 0.7853187918663025], 'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.38.2)'}

If you are particulay interested in evaluating RAG (Retrieval-Augmented Generation) pipelines, we have several metrics available in LightRAG to assess both the quality of the retrieved context and the quality of the final generated answer.

- :class:`RetrieverEvaluator <eval.evaluators.RetrieverEvaluator>`: This evaluator is used to evaluate the performance of the retriever component of the RAG pipeline. It has the following metric functions:
    - :obj:`compute_recall`: This function computes the recall of the retriever. It is defined as the number of relevant documents retrieved by the retriever divided by the total number of relevant documents in the knowledge base.
    - :obj:`compute_context_relevance`: This function computes the relevance of the retrieved context. It is defined as the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
- :class:`AnswerMacthEvaluator <eval.evaluators.AnswerMacthEvaluator>`: This evaluator is used to evaluate the performance of the generator component of the RAG pipeline. It has the following metric functions:
    - :obj:`compute_match_acc (if type is 'exact_match')`: This function computes the exact match accuracy of the generated answer. It is defined as the number of generated answers that exactly match the ground truth answer divided by the total number of generated answers.
    - :obj:`compute_match_acc (if type is 'fuzzy_match')`: This function computes the fuzzy match accuracy of the generated answer. It is defined as the number of generated answers that contain the ground truth answer divided by the total number of generated answers.
- :class:`LLMasJudge <eval.evaluators.LLMasJudge>`: This evaluator uses an LLM to get the judgement of the predicted answer for a list of questions. The task description and the judgement query of the LLM judge can be customized.
    - :obj:`compute_judgement`: This function computes the judgement of the predicted answer. It is defined as the number of generated answers that are judged as correct by the LLM divided by the total number of generated answers.

Please refer to the tutorial on `Evaluating a RAG Pipeline <>`_ for more details on how to use these evaluators. For more metrics for evaluating RAG pipelines, you can check out the `RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_ library, which also has a set of metrics for evaluating RAG pipelines.


.. [1] Chang, Yupeng, et al. "A survey on evaluation of large language models." ACM Transactions on Intelligent Systems and Technology 15.3 (2024): 1-45.
.. [2] Guo, Zishan, et al. "Evaluating large language models: A comprehensive survey." arXiv preprint arXiv:2310.19736 (2023).
.. [3] Achiam, Josh, et al. "GPT-4 technical report." arXiv preprint arXiv:2303.08774 (2023).
.. [4] Hendrycks, Dan, et al. "Measuring massive multitask language understanding." International Conference on Learning Representations. 2020.
.. [5] Chen, Mark, et al. "Evaluating large language models trained on code." arXiv preprint arXiv:2107.03374 (2021).
.. [6] Liu, Yang, et al. "Datasets for Large Language Models: A Comprehensive Survey." arXiv preprint arXiv:2402.18041 (2024).
.. [7] Li, Minghao, et al. "API-Bank: A comprehensive benchmark for tool-augmented llms." The 2023 Conference on Empirical Methods in Natural Language Processing. 2023.
