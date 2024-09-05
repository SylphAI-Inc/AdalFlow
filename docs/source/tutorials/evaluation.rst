
.. _tutorials-llm-evaluation:

.. todo: link to source code and colab version.


.. <a href="https://colab.research.google.com/drive/1gmxeX1UuUxZDouWhkLGQYrD4hAdt9IVX?usp=sharing" target="_blank" style="margin-right: 10px;">
..     <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
.. </a>

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">

      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/evaluation" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

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


Tasks and Capabilities to Evaluate
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

Datasets and Benchmarks
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

The output will be a Dataset object containing the test set of the MMLU dataset.

.. code-block:: json
    Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })

Evaluation Metrics
------------------------------------------

The final question is how to evaluate.
Evaluation methods can be divided into *automated evaluation* and *human evaluation* (*Chang et al.* [1]_ and *Liu et al.* [6]_).
Automated evaluation typically involves using metrics such as accuracy and BERTScore or employing an LLM as the judge, to quantitatively assess the performance of LLMs on specific tasks.
Human evaluation, on the other hand, involves human in the loop to evaluate the quality of the generated text or the performance of the LLM.

Here, we recommend a few automated evaluation methods that can be used to evaluate LLMs and their applications.

1. For classicial NLU tasks, such as text classification and sentiment analysis, you can use metrics such as accuracy, F1-score, and ROC-AUC to evaluate the performance of LLM response just like you would do using non-genAI models. You can check out `TorchMetrics <https://lightning.ai/docs/torchmetrics>`_.

2. For NLG tasks, such as text summarization, translation, and question answering: (1) you can use metrics such as ROUGE, BLEU, METEOR, and BERTScore, perplexity, :class:`LLMasJudge <eval.llm_as_judge>` etc to evaluate the quality of the generated text with respect to the reference text.
   You can check out the metrics provided by `Hugging Face Metrics <https://huggingface.co/metrics>`_.
   For instance, to compute the BERTScore, you can use the corresponding metric function provided by Hugging Face, which uses the pre-trained contextual embeddings from BERT and matched words in generated text and reference text by cosine similarity.
   (2) When you have no reference text, :class:`LLMasJudge <eval.llm_as_judge>` with advanced model can be used to evaluate the generated text on the fly.

3. For RAG (Retrieval-Augmented Generation) pipelines, you can use metrics such as :class:`RetrieverRecall <eval.retriever_recall>`, :class:`RetrieverRelevance <eval.retriever_relevance>`, :class:`AnswerMatchAcc <eval.answer_match_acc>`, and :class:`LLMasJudge <eval.llm_as_judge>` to evaluate the quality of the retrieved context and the generated answer.

NLG Evaluation Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classicial String Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest metric would be :class:`AnswerMatchAcc <eval.answer_match_acc>`: This calculates the exact match accuracy or fuzzy match accuracy of the generated answers by comparing them to the ground truth answers.


There are more advanced traditional metrics such as BLEU[8]_, ROUGE[9]_, and METEOR[12]_ may fail to capture the semantic similarity between the reference text and the generated text, resulting low correlation with human judgement.
You can use `TorchMetrics` [10]_ to compute these two metrics.

For instance

.. code-block:: python

    gt = "Brazil has won 5 FIFA World Cup titles"
    pred = "Brazil is the five-time champion of the FIFA WorldCup."

    def compute_rouge(gt, pred):
        from torchmetrics.text.rouge import ROUGEScore

        rouge = ROUGEScore()
        return rouge(pred, gt)


    def compute_bleu(gt, pred):
        from torchmetrics.text.bleu import BLEUScore

        bleu = BLEUScore()
        return bleu([pred], [[gt]])

The output Rouge score is:

.. code-block:: json

    {'rouge1_fmeasure': tensor(0.2222), 'rouge1_precision': tensor(0.2000), 'rouge1_recall': tensor(0.2500), 'rouge2_fmeasure': tensor(0.), 'rouge2_precision': tensor(0.), 'rouge2_recall': tensor(0.), 'rougeL_fmeasure': tensor(0.2222), 'rougeL_precision': tensor(0.2000), 'rougeL_recall': tensor(0.2500), 'rougeLsum_fmeasure': tensor(0.2222), 'rougeLsum_precision': tensor(0.2000), 'rougeLsum_recall': tensor(0.2500)}

The output BLEU score is: 0.0

These two sentences totally mean the same, but it scored low in BLEU and ROUGE.

Model-based Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

To make up to this, model based metrics or neural evaluators such as BERTScore was created.
You can find BERT score from both `Hugging Face Metrics <https://huggingface.co/metrics>`_ and `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/text/bertscore.html>`_.

.. code-block:: python

    def compute_bertscore(gt, pred):
        r"""
        https://lightning.ai/docs/torchmetrics/stable/text/bert_score.html
        """
        from torchmetrics.text.bert import BERTScore

        bertscore = BERTScore()
        return bertscore([pred], [gt])

The output BERT score is:

.. code-block:: json

    {'precision': tensor(0.9752), 'recall': tensor(0.9827), 'f1': tensor(0.9789)}

This score does reflect the semantic similarity between the two sentences almost perfectly.
However, the downside of all the above metrics is that you need to have a reference text to compare with.
Labeling such as reference text can be quite challenging in many NLG tasks, such as a summarization task.


.. .. code-block:: python
..     :linenos:

..     from datasets import load_metric
..     bertscore = load_metric("bertscore")
..     generated_text = ["life is good", "aim for the stars"]
..     reference_text = ["life is great", "make it to the moon"]
..     results = bertscore.compute(predictions=generated_text, references=reference_text, model_type="distilbert-base-uncased")
..     print(results)


.. The output will be a dictionary containing the precision, recall, and F1-score of the BERTScore metric for the generated text compared to the reference text.

.. .. code-block:: json

..     {'precision': [0.9419728517532349, 0.7959791421890259], 'recall': [0.9419728517532349, 0.7749403119087219], 'f1': [0.9419728517532349, 0.7853187918663025], 'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.38.2)'}

.. In general, BERT score works much better but you still need to label a ground truth.

G-eval
^^^^^^^^^^^^^^^^^^^^^^^^^
If you have no reference text, you can use G-eval [11]_ to evaluate the generated text on the fly.
:class:`LLMasJudge <eval.llm_as_judge>` works whether you have a reference text or not.

**With References**


.. code-block:: python

    def compute_llm_as_judge():
        import adalflow as adal
        from adalflow.eval.llm_as_judge import LLMasJudge

        adal.setup_env()

        questions = [
            "Is Beijing in China?",
            "Is Apple founded before Google?",
            "Is earth flat?",
        ]
        pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
        gt_answers = ["Yes", "Yes", "No"]
        # judgement_query = (
        #     "For the question, does the predicted answer contain the ground truth answer?"
        # )
        llm_judge = LLMasJudge()
        avg_judgement, judgement_list = llm_judge.compute(
            questions, gt_answers, pred_answers
        )
        print(avg_judgement)
        print(judgement_list)

The output will be:

.. code-block:: json

    0.6666666666666666
    [True, True, False]

You can view the prompt we used simply using `print(llm_judge)`:

.. code-block:: python

    llm_evaluator=DefaultLLMJudge(
        judgement_query= Does the predicted answer contain the ground truth answer? Say True if yes, False if no.
        (model_client): OpenAIClient()
        (llm_evaluator): Generator(
            model_kwargs={'model': 'gpt-3.5-turbo', 'temperature': 0.3, 'stream': False},
            (prompt): Prompt(
            template: <START_OF_SYSTEM_PROMPT>
            {# task desc #}
            You are an evaluator. Given the question, ground truth answer, and predicted answer,
            {# judgement question #}
            {{judgement_str}}
            <END_OF_SYSTEM_PROMPT>
            ---------------------
            <START_OF_USER>
            {# question #}
            Question: {{question_str}}
            {# ground truth answer #}
            Ground truth answer: {{gt_answer_str}}
            {# predicted answer #}
            Predicted answer: {{pred_answer_str}}
            {# assistant response #}
            <END_OF_USER>
            , prompt_variables: ['pred_answer_str', 'judgement_str', 'gt_answer_str', 'question_str']
            )
            (model_client): OpenAIClient()
        )
    )

RAG Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RAG (Retrieval-Augmented Generation) pipelines are a combination of a retriever and a generator. The retriever retrieves relevant context from a large corpus, and the generator generates the final answer based on the retrieved context.
When a retriever failed to retrieve relevant context, the generator may fail.
Therefore, besides of evaluating RAG pipelines as a whole using NLG metrics, it is also important to evaluate the retriever and to optimize the evalulation metrics from both stages to best improve the final performance.

For the retriever, the metrics used are nothing new but from the standard information retrieval literature.
Often, we have Mean Reciprocal Rank(MRR@k), Recall@k, Precision@k, F1@k, MAP@k, NDCG@k, etc.
All of these metrics, you can find at `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/>`_.


.. For the retriever:

.. - :class:`RetrieverRecall <eval.retriever_recall>`: This is used to evaluate the recall of the retriever component of the RAG pipeline.
.. .. - :class:`RetrieverRelevance <eval.retriever_relevance>`: This is used to evaluate the relevance of the retrieved context to the query.

.. For the generator:

.. - :class:`LLMasJudge <eval.llm_as_judge>`: This uses an LLM to get the judgement of the generated answer for a list of questions. The task description and the judgement query of the LLM judge can be customized. It computes the judgement score, which is the number of generated answers that are judged as correct by the LLM divided by the total number of generated answers.

For example, you can use the following code snippet to compute the recall and relevance of the retriever component of the RAG pipeline for a single query.

.. code-block:: python

    from adalflow.eval import RetrieverRecall, RetrieverRelevance

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

The output will be:

.. code-block:: json
    Recall: 0.6666666666666666, Recall List: [0.3333333333333333, 1.0]


.. retriever_relevance = RetrieverRelevance()
.. avg_relevance, relevance_list = retriever_relevance.compute(retrieved_contexts, gt_contexts) # Compute the relevance of the retriever
.. print(f"Relevance: {avg_relevance}, Relevance List: {relevance_list}")
.. # Relevance: 0.803030303030303, Relevance List: [1.0, 0.6060606060606061]

For a more detailed instructions on how build and evaluate RAG pipelines, you can refer to the use case on :doc:`Evaluating a RAG Pipeline <../tutorials/eval_a_rag>`.

If you intent to use metrics that are not available in the AdalFlow library, you can also implement your own custom metric functions or use other libraries such as `RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_ to compute the desired metrics for evaluating RAG pipelines.

Additionally, there are more research for RAG evaluation, such as SemScore[13]_, ARES[14]_, RGB[15]_, etc.

References
------------------------------------------

.. [1] Chang, Yupeng, et al. "A survey on evaluation of large language models." ACM Transactions on Intelligent Systems and Technology 15.3 (2024): 1-45.
.. [2] Guo, Zishan, et al. "Evaluating large language models: A comprehensive survey." arXiv preprint arXiv:2310.19736 (2023).
.. [3] Achiam, Josh, et al. "GPT-4 technical report." arXiv preprint arXiv:2303.08774 (2023).
.. [4] Hendrycks, Dan, et al. "Measuring massive multitask language understanding." International Conference on Learning Representations. 2020.
.. [5] Chen, Mark, et al. "Evaluating large language models trained on code." arXiv preprint arXiv:2107.03374 (2021).
.. [6] Liu, Yang, et al. "Datasets for Large Language Models: A Comprehensive Survey." arXiv preprint arXiv:2402.18041 (2024).
.. [7] Finardi, Paulo, et al. "The Chronicles of RAG: The Retriever, the Chunk and the Generator." arXiv preprint arXiv:2401.07883 (2024).
.. [8]  K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine transla-tion,” in Proceedings of the 40th annual meeting on association for computational linguistics. Association for Computational Linguistics, 2002, pp. 311–318.
.. [9]  C.-Y. Lin, “Rouge: a package for automatic evaluation of summaries,” 2004.
.. [10] https://lightning.ai/docs/torchmetrics/stable/text/rouge_score.html
.. [11] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: Nlg evaluation using gpt-4 with better humanalignment,” 2023.
.. [12] Satanjeev Banerjee and Alon Lavie. 2005. Meteor: Anautomatic metric for mt evaluation with improved cor-relation with human judgments. In Proceedings ofthe acl workshop on intrinsic and extrinsic evaluationmeasures for machine translation and/or summariza-tion, pages 65–72.
.. [13] SemScore: https://arxiv.org/abs/2401.17072
.. [14] ARES: https://arxiv.org/abs/2311.09476
.. [15] RGB: https://ojs.aaai.org/index.php/AAAI/article/view/29728


.. admonition:: Evaluation Metrics libraries
   :class: highlight

   - `TorchMetrics <https://lightning.ai/docs/torchmetrics>`_
   - `Hugging Face Metrics <https://huggingface.co/metrics>`_
   - `RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_
   - `G-eval <https://arxiv.org/abs/2303.08774>`_
