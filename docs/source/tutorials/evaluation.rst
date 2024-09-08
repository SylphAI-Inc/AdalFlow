
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
TODO: how come they did not mention RAG evaluation?

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

NLG Evaluation
------------------------------------------

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

Embedding-based Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

To make up to this, Embedding-based  metrics or neural evaluators such as BERTScore was created.
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




LLM as Judge
^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluating the LLM application using LLM as a judge is no different from building LLM task pipeline.
Developers should better know the underlying prompt to the LLM judge to decide if the default judge is enough or that they need customization.
Because of so, AdalFlow decided to provide a comprehensive set of LLM as Judge instead of sending our developers to other evaluation packages.
We did research on both the research papers and the existing libraries and found there is no library that has provided such evaluators with 100% clarity and without enforcing developers to install many other dependencies.

You can use LLM as judge for cases where you have or do not have reference text.
The key is to define the metric clearly using text.
**We are building LLM judge to replace human labelers, increasing the overall efficiency and reducing financial costs.**

**With References**

The most straightforward LLM judge is to predict a yes/no answer or a float score in range [0, 1] between the generated text and the reference text
per a judgement query.

Here is AdalFlow's default judegement query:

.. code-block:: python

    DEFAULT_JUDGEMENT_QUERY = "Does the predicted answer contain the ground truth answer? Say True if yes, False if no."

Now, you can use the following code to the final score per the judgement query:

.. code-block:: python

    def compute_llm_as_judge():
        import adalflow as adal
        from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge
        from adalflow.components.model_client import OpenAIClient

        adal.setup_env()

        questions = [
            "Is Beijing in China?",
            "Is Apple founded before Google?",
            "Is earth flat?",
        ]
        pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
        gt_answers = ["Yes", "Yes", "No"]

        llm_judge = DefaultLLMJudge(
            model_client=OpenAIClient(),
            model_kwargs={
                "model": "gpt-4o",
                "temperature": 1.0,
                "max_tokens": 10,
            },
        )
        llm_evaluator = LLMasJudge(llm_judge=llm_judge)
        print(llm_judge)
        avg_judgement, confidence_interval = llm_evaluator.compute(
            questions, gt_answers, pred_answers
        )
        print(avg_judgement)
        print(confidence_interval)

To be more rigid, you can compute a 95% confidence interval for the judgement score.
When the evaluation dataset is small, the confidence interval can have a large range, which indicates that the judgement score is not very reliable.


The output will be:

.. code-block:: json

    0.6666666666666666
    (0.013333333333333197, 1)

This type of LLM judege is seen in text-grad [17]_.
You can view the prompt we used simply using `print(llm_judge)`:

.. code-block:: python

    DefaultLLMJudge(
        judgement_query= Does the predicted answer contain the ground truth answer? Say True if yes, False if no.,
        (model_client): OpenAIClient()
        (llm_evaluator): Generator(
            model_kwargs={'model': 'gpt-4o', 'temperature': 1.0, 'max_tokens': 10}, trainable_prompt_kwargs=['task_desc_str', 'examples_str']
            (prompt): Prompt(
            template: <START_OF_SYSTEM_PROMPT>
            {# task desc #}
            {{task_desc_str}}
            {# examples #}
            {% if examples_str %}
            {{examples_str}}
            {% endif %}
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
            , prompt_kwargs: {'task_desc_str': 'You are an evaluator. Given the question, ground truth answer, and predicted answer, Does the predicted answer contain the ground truth answer? Say True if yes, False if no.', 'examples_str': None}, prompt_variables: ['examples_str', 'pred_answer_str', 'task_desc_str', 'gt_answer_str', 'question_str']
            )
            (model_client): OpenAIClient()
        )
    )


**Without References (G-eval)**

.. figure:: /_static/images/G_eval_structure.png
    :align: center
    :alt: G-eval structure
    :width: 700px

    G-eval framework structure

If you have no reference text, you can use G-eval [11]_ to evaluate the generated text on the fly.
G-eval provided a way to evaluate:

- `relevance`: evaluates how relevant the summarized text to the source text.
- `fluency`: the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
- `consistency`: evaluates the collective quality of all sentences.
- `coherence`: evaluates the the factual alignment between the summary and the summarized source.

In our library, we provides the prompt for task `Summarization` and `Chatbot` as default.


Train/Align LLM Judge
^^^^^^^^^^^^^^^^^^^^^^^^^

We should better align the LLM judge with a human preference dataset that has the (generated text, ground truth text, score) triplets.
This process is the same as optimize the task pipeline, you can create an `AdalComponent` and call our `Trainer` to do the in-context learning.
As you can see two trainable_prompt_kwargs in the `DefaultLLMJudge` from the printout.

In this case, we might want to compute a correlation score between the human judge and the LLM judge.
You have different choice, such as:

1. Pearson Correlation Coefficient
2. Kendallrank correlation coefficient from ARES [14]_ in particular for the ranking system (Retrieval).


RAG Evaluation
------------------------------------------
RAG (Retrieval-Augmented Generation) pipelines are a combination of a retriever and a generator.
The retriever retrieves relevant context from a large corpus, and the generator generates the final answer based on the retrieved context.
When a retriever failed to retrieve relevant context, the generator may fail.
Therefore, besides of evaluating RAG pipelines as a whole using NLG metrics, it is also important to evaluate the retriever and to optimize the evalulation metrics from both stages to best improve the final performance.

With GT for Retriever
^^^^^^^^^^^^^^^^^^^^^^^^^
For the retriever, the metrics used are nothing new but from the standard information retrieval/ranking literature.
Often, we have

1. Recall@k: the proportion of relevant documents that are retrieved out of the total number of relevant documents.

2. Mean Reciprocal Rank(MRR@k)

3. NDCG@k

4. Precision@k, MAP@k etc.

For defails of these metrics, please refer to [18]_.
All of these metrics, you can also find at `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/>`_.


For example, you can use the following code snippet to compute the recall@k the retriever component of the RAG pipeline for a single query if
the ground truth context is provided.
In this example, the retrieved contexts is a joined string of the retrieved context chunks, and the gt_contexts is a list of ground truth context chunks for each query.

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

For the first query, only one out of three relevant documents is retrieved, resulting in a recall of 0.33.
For the second query, all relevant documents are retrieved, resulting in a recall of 1.0.


Without gt_contexts
^^^^^^^^^^^^^^^^^^^^^^^^^

RAGAS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ideally, for each query, we will retrieve the top k (@k) chunks and to get the above score, we expect each query, retrieved chunk pair comes with a ground truth labeling.
But this is highly unrealistic especially if corpora is large.
If we have 100 test queries, and a corpus of size 1000 chunks, the pairs we need to annoate is 10^5.
There are different strategies to handle this problem but we could not dive into all of them here.

There is one new way is to indirectly use the ground truth answers from the generator to evaluate the retriever.
`RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_ framework provides one way to do this.

    Recall = [GT statements that can be attributed to the retrieved context] / [GT statements]


LLM or model based judge for Retriever Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**LLM judge with in-context prompting**

LLM judge to directly straightforward way to evaluate the top k score on the fly.

We can create a subset of query, retrieved chunk pairs and manually label them, and we train an LLM judge to predict the score.
If the judge can achieve a high accuracy then we are able to annotate any metric in the retriever given the query and the retrieved chunk pairs.

**ARES with finetuned classifier with synthetic data**

ARES [14]_ proposed to create a synthetic dataset from an in-domain corpora.
The generated data represent both positive and negative examples of `query–passage–answer triples`` (e.g.,relevant/irrelevant passages and correct/incorrectanswers).


The synthetic dataset is used to train a classifier consists of  embedding and a classification head.
It claims to be able to adapt to other domains where the classifier is not trained on.
The cost of this approach is quite low as you can compute the embedding for only once for each query and each chunk in the corpus.


See the evaluation on datasets at :doc:`Evaluating a RAG Pipeline <../tutorials/eval_a_rag>`.

Additionally, there are more research for RAG evaluation, such as SemScore [13]_, ARES [14]_, RGB [15]_, etc.


For Contributors
------------------------------------------
There are way too many metrics and evaluation methods that AdalFlow can cover in the library.
We encourage contributors who work on evaluation research and production to build evaluator that is compatible with AdalFlow.
This means that:

1. The evaluator can potentially output a single float score in range [0, 1] so that AdalFlow Trainer can use it to optimize the pipeline.

2. For using LLM as judge, the judge should be built similar to `DefaultLLMJudge` so that there are trainable_prompt_kwargs that users can further align the judge with human preference dataset.

For instance, for the research papers we have listed here, it would be great to have a version that is easily compatible with AdalFlow.

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
.. [14] ARES: https://arxiv.org/abs/2311.09476, https://github.com/stanford-futuredata/ARES
.. [15] RGB: https://ojs.aaai.org/index.php/AAAI/article/view/29728
.. [16] G-eval: https://github.com/nlpyang/geval
.. [17] Text-grad: https://arxiv.org/abs/2309.03409
.. [18] Pretrained Transformers for Text Ranking: BERT and Beyond: https://arxiv.org/pdf/2010.06467


.. admonition:: Evaluation Metrics libraries
   :class: highlight

   - `TorchMetrics <https://lightning.ai/docs/torchmetrics>`_
   - `Hugging Face Metrics <https://huggingface.co/metrics>`_
   - `RAGAS <https://docs.ragas.io/en/stable/getstarted/index.html>`_
   - `G-eval <https://arxiv.org/abs/2303.08774>`_



.. admonition:: API Reference
   :class: highlight

   - `RetrieverRecall <eval.retriever_recall>`
   - `DefaultLLMJudge <eval.llm_as_judge>`
