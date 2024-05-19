Evaluation and Metrics
=========

Evaluating an LLM application essentially involves various metric functions. You can write your own metric functions or import from other libraries. In LightRAG, we provide a set of metrics in :ref:`our evaluators <evaluators>`. In this tutorial, we will show how to use them to evaluate the performance of the retriever and generator components of a RAG pipeline


Evaluating a RAG Pipeline
---------------------------------------
The full code for this tutorial can be found in `use_cases/rag_hotpotqa.py <https://github.com/SylphAI-Inc/LightRAG/blob/main/use_cases/rag_hotpotqa.py>`_.

RAG (Retrieval-Augmented Generation) pipelines leverage a retriever to fetch relevant context from a knowledge base (e.g., a document database) which is then fed to an LLM generator with the query to produce the answer. This allows the model to generate more contextually relevant answers.

Thus, to evaluate a RAG pipeline, we can assess both the quality of the retrieved context and the quality of the final generated answer. Speciafically, we can use the following evaluators and their corresponding metrics.

* :class:`RetrieverEvaluator <eval.evaluators.RetrieverEvaluator>`: This evaluator is used to evaluate the performance of the retriever component of the RAG pipeline. It has the following metric functions:
    * :obj:`compute_recall`: This function computes the recall of the retriever. It is defined as the number of relevant strings retrieved by the retriever divided by the total number of relevant strings in the knowledge base.
    * :obj:`compute_context_relevance`: This function computes the relevance of the retrieved context. It is defined as the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
* :class:`AnswerMacthEvaluator <eval.evaluators.AnswerMacthEvaluator>`: This evaluator is used to evaluate the performance of the generator component of the RAG pipeline. It has the following metric functions:
    * :obj:`compute_match_acc (if type is 'exact_match')`: This function computes the exact match accuracy of the generated answer. It is defined as the number of generated answers that exactly match the ground truth answer divided by the total number of generated answers.
    * :obj:`compute_match_acc (if type is 'fuzzy_match')`: This function computes the fuzzy match accuracy of the generated answer. It is defined as the number of generated answers that contain the ground truth answer divided by the total number of generated answers.
* :class:`LLMasJudge <eval.evaluators.LLMasJudge>`: This evaluator uses an LLM to get the judgement of the predicted answer for a list of questions. The task description and the judgement query of the LLM judge can be customized.
    * :obj:`compute_judgement`: This function computes the judgement of the predicted answer. It is defined as the number of generated answers that are judged as correct by the LLM divided by the total number of generated answers.


TODO
