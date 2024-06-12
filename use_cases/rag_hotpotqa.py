from typing import List, Union
import dotenv
import yaml

from datasets import load_dataset

from lightrag.core.generator import Generator, GeneratorOutput
from lightrag.core.types import Document

from lightrag.core.string_parser import JsonParser
from lightrag.core.component import Sequential, Component
from lightrag.eval import (
    RetrieverRecall,
    RetrieverRelevance,
    AnswerMatchAcc,
    LLMasJudge,
    DEFAULT_LLM_EVALUATOR_PROMPT,
)
from lightrag.core.prompt_builder import Prompt

from use_cases.rag import RAG

from lightrag.components.model_client import OpenAIClient


dotenv.load_dotenv(dotenv_path=".env", override=True)


def get_supporting_sentences(
    supporting_facts: dict[str, list[Union[str, int]]], context: dict[str, list[str]]
) -> List[str]:
    """
    Extract the supporting sentences from the context based on the supporting facts. This function is specific to the HotpotQA dataset.
    """
    extracted_sentences = []
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        if title in context["title"]:
            index = context["title"].index(title)
            sentence = context["sentences"][index][sent_id]
            extracted_sentences.append(sentence)
    return extracted_sentences


LLM_EVALUATOR_PROMPT = r"""
<<SYS>>{# task desc #}
You are a helpful assistant.
Given the question, ground truth answer, and predicted answer, you need to answer the judgement query.
Output True or False according to the judgement query following this JSON format:
{
    "judgement": True
}
<</SYS>>
---------------------
{# question #}
Question: {{question_str}}
{# ground truth answer #}
Ground truth answer: {{gt_answer_str}}
{# predicted answer #}
Predicted answer: {{pred_answer_str}}
{# judgement question #}
Judgement question: {{judgement_str}}
{# assistant response #}
You:
"""


class CustomizedLLMJudge(Component):
    def __init__(self, model_kwargs: dict):
        super().__init__()
        self.model_kwargs = model_kwargs
        self.llm_evaluator = Generator(
            model_client=OpenAIClient(),
            template=LLM_EVALUATOR_PROMPT,
            output_processors=JsonParser(),
            model_kwargs=model_kwargs,
        )

    def call(
        self, question: str, pred_answer: str, gt_answer: str, judgement_query: str
    ) -> bool:
        prompt_kwargs = {
            "question_str": question,
            "gt_answer_str": gt_answer,
            "pred_answer_str": pred_answer,
            "judgement_str": judgement_query,
        }
        response: GeneratorOutput = self.llm_evaluator.call(prompt_kwargs=prompt_kwargs)
        if not response.data:
            raise ValueError("No data returned from the LLM evaluator.")
        return response.data["judgement"] if "judgement" in response.data else None


if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
    with open("./use_cases/configs/rag_hotpotqa.yaml", "r") as file:
        settings = yaml.safe_load(file)
    print(settings)

    # Load the dataset and select the first 5 as the showcase
    # 300 M.
    # More info about the HotpotQA dataset can be found at https://huggingface.co/datasets/hotpot_qa
    # where is the downloaded data saved?
    dataset = load_dataset(path="hotpot_qa", name="fullwiki")
    print(f"len of eval: {len(dataset['test'])}")
    print(f"example: {dataset['test'][1]}")
    # exit()
    dataset = dataset["train"].select(range(1))

    all_questions = []
    all_retrieved_context = []
    all_gt_context = []
    all_pred_answer = []
    all_gt_answer = []
    for data in dataset:
        # Each sample in HotpotQA has multiple documents to retrieve from. Each document has a title and a list of sentences.
        num_docs = len(data["context"]["title"])
        doc_list = [
            Document(
                meta_data={"title": data["context"]["title"][i]},
                text=" ".join(data["context"]["sentences"][i]),
            )
            for i in range(num_docs)
        ]

        # Run the RAG and validate the retrieval and generation
        rag = RAG(settings)
        print(rag)
        rag.build_index(doc_list)
        print(rag.tracking)

        query = data["question"]
        response, context_str = rag.call(query)

        # Get the ground truth context_str
        gt_context_sentence_list = get_supporting_sentences(
            data["supporting_facts"], data["context"]
        )

        all_questions.append(query)
        all_retrieved_context.append(context_str)
        all_gt_context.append(gt_context_sentence_list)
        all_pred_answer.append(response["answer"])
        all_gt_answer.append(data["answer"])
        print("====================================================")
        print(f"query: {query}")
        print(f"response: {response['answer']}")
        print(f"ground truth response: {data['answer']}")
        print(f"context_str: {context_str}")
        print(f"ground truth context_str: {gt_context_sentence_list}")
        print("====================================================")

    # Evaluate the retriever
    retriever_recall = RetrieverRecall()
    retriever_relevance = RetrieverRelevance()
    avg_recall, recall_list = retriever_recall.compute(
        all_retrieved_context, all_gt_context
    )
    avg_relevance, relevance_list = retriever_relevance.compute(
        all_retrieved_context, all_gt_context
    )
    print(f"Average recall: {avg_recall}")
    print(f"Recall for each query: {recall_list}")
    print(f"Average relevance: {avg_relevance}")
    print(f"Relevance for each query: {relevance_list}")

    # Evaluate the generator
    generator_evaluator = AnswerMatchAcc(type="fuzzy_match")
    answer_match_acc, match_acc_list = generator_evaluator.compute(
        all_pred_answer, all_gt_answer
    )
    print(f"Answer match accuracy: {answer_match_acc}")
    print(f"Match accuracy for each query: {match_acc_list}")
    # Evaluate the generator using LLM as judge.
    # The task description and the judgement query can be customized.

    llm_judge = LLMasJudge(
        llm_evaluator=CustomizedLLMJudge(model_kwargs=settings["llm_evaluator"])
    )
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    avg_judgement, judgement_list = llm_judge.compute(
        all_questions, all_pred_answer, all_gt_answer, judgement_query
    )
    print(f"Average judgement: {avg_judgement}")
    print(f"Judgement for each query: {judgement_list}")
