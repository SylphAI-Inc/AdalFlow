import datasets
from typing import List, Union

import adalflow as adal
from use_cases.rag.build.rag import (
    RAG,
)
from adalflow.eval.retriever_recall import RetrieverRecall
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.eval.llm_as_judge import LLMasJudge


def load_hotpot_qa():
    dataset = datasets.load_dataset(path="hotpot_qa", name="fullwiki")
    selected_dataset = dataset["train"].select(range(5))
    print(f"example: {selected_dataset[0]}")
    print(f"ground truth context: {selected_dataset[0]['supporting_facts']}")
    return selected_dataset


def get_supporting_sentences(
    supporting_facts: dict[str, list[Union[str, int]]], context: dict[str, list[str]]
) -> List[str]:
    """
    Extract the supporting sentences from the context based on the supporting facts.
    """
    extracted_sentences = []
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        if title in context["title"]:
            index = context["title"].index(title)
            sentence = context["sentences"][index][sent_id]
            extracted_sentences.append(sentence)
    return extracted_sentences


def prepare_documents(dataset):
    # For production use cases, you might consider batching the documents using a data loader
    docs = []
    for data in dataset:
        num_docs = len(data["context"]["title"])
        id = data["id"]
        doc_list = [
            adal.Document(
                meta_data={"title": data["context"]["title"][i]},
                text=f"title: {data['context']['title'][i]} "
                + ", sentences: "
                + " ".join(data["context"]["sentences"][i]),
                id=f"doc_{id}_{i}",
            )
            for i in range(num_docs)
        ]
        docs.extend(doc_list)
    return docs


def add_all_documents_to_rag_db(rag):
    dataset = load_hotpot_qa()
    docs = prepare_documents(dataset)
    rag.add_documents(docs)  # add all documents to the database


if __name__ == "__main__":

    rag = RAG(index_file="hotpot_qa_index.faiss")
    # add_all_documents_to_rag_db(rag)
    print(rag.transformed_docs)

    dataset = load_hotpot_qa()
    questions, retrieved_contexts, gt_contexts, pred_answers, gt_answers = (
        [],
        [],
        [],
        [],
        [],
    )
    for item in dataset:
        id = item["id"]
        doc_ids = [f"doc_{id}_{i}" for i in range(len(item["context"]["title"]))]
        # transformed_docs = rag.get_transformed_docs(
        #     filter_func=lambda x: id in x.parent_doc_id
        # )
        # print(f"id: {id}")
        # print(f"transformed_docs: {[ (doc.id, doc.order, doc.parent_doc_id)
        #                              for doc in transformed_docs]}")
        rag.prepare_retriever(filter_func=lambda x: x.parent_doc_id in doc_ids)
        response, context_str = rag.call(item["question"])
        gt_context_sentence_list = get_supporting_sentences(
            item["supporting_facts"], item["context"]
        )
        print(f"gt_context_sentence_list: {gt_context_sentence_list}")
        questions.append(item["question"])
        retrieved_contexts.append(context_str)
        gt_contexts.append(gt_context_sentence_list)
        pred_answers.append(response.data["answer"])
        gt_answers.append(item["answer"])
        print(f"question: {item['question']}")
        print(f"retrieved context: {context_str}")
        print(f"ground truth context: {gt_context_sentence_list}")
        print(f"predicted answer: {response.data['answer']}")
        print(f"ground truth answer: {item['answer']}")

    avg_recall = RetrieverRecall().compute(retrieved_contexts, gt_contexts)
    answer_match_acc = AnswerMatchAcc(type="fuzzy_match")
    acc_rslt = answer_match_acc.compute(
        pred_answers=pred_answers, gt_answers=gt_answers
    )
    llm_judge = LLMasJudge()
    judge_acc_rslt = llm_judge.compute(
        questions=questions, gt_answers=gt_answers, pred_answers=pred_answers
    )
    print(f"judge_acc_rslt: {judge_acc_rslt}")
    print(f"avg_recall: {avg_recall}")
    print(f"avg_acc: {acc_rslt}")
