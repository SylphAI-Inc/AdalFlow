Evaluating RAG
==========================

As RAG admits so many design choices, it is difficult to build a traditional benchmark dataset from scratch.
When it comes to evaluating RAG, we have to define the metrics, the evaluation datasets, and to annotate such datasets and ensure the metrics can reflect

In AdalFlow, we provide a set of metrics in :ref:`our evaluators <evaluators>`. In this tutorial, we will show how to use them to evaluate the performance of the retriever and generator components of a RAG pipeline

As RAG consists of two stages:

1. Retrieval: The retriever fetches relevant context from a knowledge base.
   The metrics used here are nothing new but from the standard information retrieval literature.
   Often, we have Mean Reciprocal Rank(MRR@k), Recall@k, Precision@k, F1@k, MAP@k, NDCG@k, etc.
   Please read our :ref:`Retriever <tutorials-retriever>` for more details on the retrieval itself.
   All of these metrics, you can find at `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/>`_.

2. Generation: The metrics used for evaluating the response is more diverse and highly dependent on the tasks.
   You can refer to our :ref:`Evaluation guidelines <tutorials-llm-evaluation>` for more details.

The full code for this tutorial can be found in `use_cases/rag_hotpotqa.py <https://github.com/SylphAI-Inc/AdalFlow/blob/main/use_cases/rag_hotpotqa.py>`_.

RAG (Retrieval-Augmented Generation) pipelines leverage a retriever to fetch relevant context from a knowledge base (e.g., a document database) which is then fed to an LLM generator with the query to produce the answer. This allows the model to generate more contextually relevant answers.

Thus, to evaluate a RAG pipeline, we can assess both the quality of the retrieved context and the quality of the final generated answer. Speciafically, in this tutorial, we will use the following evaluators and their corresponding metrics.

- :class:`RetrieverEvaluator <eval.evaluators.RetrieverEvaluator>`: This evaluator is used to evaluate the performance of the retriever component of the RAG pipeline. It has the following metric functions:
    - :obj:`compute_recall`: This function computes the recall of the retriever. It is defined as the number of relevant documents retrieved by the retriever divided by the total number of relevant documents in the knowledge base.
    - :obj:`compute_context_relevance`: This function computes the relevance of the retrieved context. It is defined as the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
- :class:`AnswerMacthEvaluator <eval.evaluators.AnswerMacthEvaluator>`: This evaluator is used to evaluate the performance of the generator component of the RAG pipeline. It has the following metric functions:
    - :obj:`compute_match_acc (if type is 'exact_match')`: This function computes the exact match accuracy of the generated answer. It is defined as the number of generated answers that exactly match the ground truth answer divided by the total number of generated answers.
    - :obj:`compute_match_acc (if type is 'fuzzy_match')`: This function computes the fuzzy match accuracy of the generated answer. It is defined as the number of generated answers that contain the ground truth answer divided by the total number of generated answers.
- :class:`LLMasJudge <eval.evaluators.LLMasJudge>`: This evaluator uses an LLM to get the judgement of the predicted answer for a list of questions. The task description and the judgement query of the LLM judge can be customized.
    - :obj:`compute_judgement`: This function computes the judgement of the predicted answer. It is defined as the number of generated answers that are judged as correct by the LLM divided by the total number of generated answers.


Let's walk through the code to evaluate a RAG pipeline step by step.

**Step 1: import dependencies.**
We import the necessary dependencies for our evaluation script. These include modules for loading datasets, constructing a RAG pipeline, and evaluating the performance of the RAG pipeline.

.. code-block::
    :linenos:

    import yaml

    from datasets import load_dataset

    from core.openai_client import OpenAIClient
    from core.generator import Generator
    from core.base_data_class import Document
    from core.string_parser import JsonParser
    from core.component import Sequential
    from eval.evaluators import (
        RetrieverEvaluator,
        AnswerMacthEvaluator,
        LLMasJudge,
        DEFAULT_LLM_EVALUATOR_PROMPT,
    )
    from core.prompt_builder import Prompt
    from use_cases.rag import RAG

**Step 2: define the configuration.**
We load the configuration settings from `a YAML file <https://github.com/SylphAI-Inc/AdalFlow/blob/main/use_cases/configs/rag_hotpotqa.yaml>`_. This file contains various parameters for the RAG pipeline. You can customize these settings based on your requirements.

.. code-block::
    :linenos:

    with open("./configs/rag_hotpotqa.yaml", "r") as file:
        settings = yaml.safe_load(file)

**Step 3: load the dataset.**
In this tutorial, we use the `HotpotQA dataset <https://huggingface.co/datasets/hotpot_qa>`_ as an example. Each data sample in HotpotQA has *question*, *answer*, *context* and *supporting_facts* selected from the whole context. We load the HotpotQA dataset using the :obj:`load_dataset` function from the `datasets <https://huggingface.co/docs/datasets>`_ module. We select a subset of the dataset as an example for evaluation purposes.

.. code-block::
    :linenos:

    dataset = load_dataset(path="hotpot_qa", name="fullwiki")
    dataset = dataset["train"].select(range(5))

**Step 4: build the document list for each sample in the dataset.**
For each sample in the dataset, we create a list of documents to retrieve from according to its corresponding *context* in the dataset. Each document has a title and a list of sentences. We use the :obj:`Document` class from the :obj:`core.base_data_class` module to represent each document.

.. code-block::
    :linenos:

    for data in dataset:
        num_docs = len(data["context"]["title"])
        doc_list = [
            Document(
                meta_data={"title": data["context"]["title"][i]},
                text=" ".join(data["context"]["sentences"][i]),
            )
            for i in range(num_docs)
        ]

**Step 5: build the RAG pipeline.**
We initialize the RAG pipeline by creating an instance of the :obj:`RAG` class with the loaded configuration settings. We then build the index using the document list created in the previous step.

.. code-block::
    :linenos:

    for data in dataset:
        # following the previous code snippet
        rag = RAG(settings)
        rag.build_index(doc_list)

**Step 6: retrieve the context and generate the answer.**
For each sample in the dataset, we retrieve the context and generate the answer using the RAG pipeline. We can print the query, response, ground truth response, context string, and ground truth context string for each sample.

To get the ground truth context string from the *supporting_facts* filed in HotpotQA. We have implemented a :obj:`get_supporting_sentences` function, which extract the supporting sentences from the context based on the *supporting_facts*. This function is specific to the HotpotQA dataset, which is available in `use_cases/rag_hotpotqa.py <https://github.com/SylphAI-Inc/AdalFlow/blob/main/use_cases/rag_hotpotqa.py>`_.

.. code-block::
    :linenos:

    all_questions = []
    all_retrieved_context = []
    all_gt_context = []
    all_pred_answer = []
    all_gt_answer = []
    for data in dataset:
        # following the previous code snippet
        query = data["question"]
        response, context_str = rag.call(query)
        gt_context_sentence_list = get_supporting_sentences(
            data["supporting_facts"], data["context"]
        )
        all_questions.append(query)
        all_retrieved_context.append(context_str)
        all_gt_context.append(gt_context_sentence_list)
        all_pred_answer.append(response["answer"])
        all_gt_answer.append(data["answer"])
        print(f"query: {query}")
        print(f"response: {response['answer']}")
        print(f"ground truth response: {data['answer']}")
        print(f"context_str: {context_str}")
        print(f"ground truth context_str: {gt_context_sentence_list}")


**Step 7: evaluate the performance of the RAG pipeline.**
We first evaluate the performance of the retriever component of the RAG pipeline. We compute the average recall and context relevance for each query using the :class:`RetrieverEvaluator <eval.evaluators.RetrieverEvaluator>` class.

.. code-block::
    :linenos:

    retriever_evaluator = RetrieverEvaluator()
    avg_recall, recall_list = retriever_evaluator.compute_recall(
        all_retrieved_context, all_gt_context
    )
    avg_relevance, relevance_list = retriever_evaluator.compute_context_relevance(
        all_retrieved_context, all_gt_context
    )
    print(f"Average recall: {avg_recall}")
    print(f"Average relevance: {avg_relevance}")

Next, we evaluate the performance of the generator component of the RAG pipeline. We compute the average exact match accuracy for each query using the :class:`AnswerMacthEvaluator <eval.evaluators.AnswerMacthEvaluator>` class.

.. code-block::
    :linenos:

    generator_evaluator = AnswerMacthEvaluator(type="fuzzy_match")
    answer_match_acc, match_acc_list = generator_evaluator.compute_match_acc(
        all_pred_answer, all_gt_answer
    )
    print(f"Answer match accuracy: {answer_match_acc}")

Finally, we evaluate the performance of the generator component of the RAG pipeline using an LLM judge. We compute the average judgement for each query using the :class:`LLMasJudge <eval.evaluators.LLMasJudge>` class.

Note that :obj:`task_desc_str` and :obj:`judgement_query` can be customized.

.. code-block::
    :linenos:

    llm_evaluator = Generator(
        model_client=OpenAIClient(),
        prompt=Prompt(DEFAULT_LLM_EVALUATOR_PROMPT),
        output_processors=Sequential(JsonParser()),
        preset_prompt_kwargs={
            "task_desc_str": r"""
                You are a helpful assistant.
                Given the question, ground truth answer, and predicted answer, you need to answer the judgement query.
                Output True or False according to the judgement query following this JSON format:
                {
                    "judgement": True
                }
                """
        },
        model_kwargs=settings["llm_evaluator"],
    )
    llm_judge = LLMasJudge(llm_evaluator)
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    avg_judgement, judgement_list = llm_judge.compute_judgement(
        all_questions, all_pred_answer, all_gt_answer, judgement_query
    )
    print(f"Average judgement: {avg_judgement}")

**Conclusion.**
In this tutorial, we learned how to evaluate a RAG pipeline using the HotpotQA dataset. We walked through the code and explained each step of the evaluation process. You can use this tutorial as a starting point to evaluate your own RAG pipelines and customize the evaluation metrics based on your requirements.

.. admonition:: API References
   :class: highlight

.. admonition:: References
   :class: highlight

   .. [1] Finardi, Paulo, et al. "The Chronicles of RAG: The Retriever, the Chunk and the Generator." arXiv preprint arXiv:2401.07883 (2024).
