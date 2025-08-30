import datasets
from typing import List, Union

import adalflow as adal
from use_cases.rag.build.rag import (
    RAG,
)
from adalflow.eval.retriever_recall import RetrieverEvaluator
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
    extract the supporting sentences from the context based on the supporting facts.
    """
    extracted_sentences = []
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        if title in context["title"]:
            index = context["title"].index(title)
            sentence = context["sentences"][index][sent_id]
            extracted_sentences.append(sentence)
    return extracted_sentences


def prepare_documents(dataset):
    # for production use cases, you might consider batching the documents using a data loader
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
    
    # debug: check if documents are loaded and show sample data
    print(f"number of transformed docs: {len(rag.transformed_docs)}")
    if rag.transformed_docs:
        print(f"sample transformed doc: {rag.transformed_docs[0]}")
        print(f"sample parent_doc_id: {rag.transformed_docs[0].parent_doc_id}")
    else:
        print("warning: no documents found in the rag system")
        print("uncomment the line below to add documents if they don't exist")
        # add_all_documents_to_rag_db(rag)
        # exit early if no documents are available
        exit(1)
    
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
        print(f"looking for doc_ids: {doc_ids}")
        
        # debug: check what documents match the filter before processing
        all_transformed_docs = rag.get_transformed_docs(filter_func=None)
        matching_docs = [doc for doc in all_transformed_docs if doc.parent_doc_id in doc_ids]
        print(f"found {len(matching_docs)} matching documents")
        
        # prepare retriever with filtered documents for this specific item
        rag.prepare_retriever(filter_func=lambda x: x.parent_doc_id in doc_ids)
        
        # handle case where no documents are found for this item
        if not rag.transformed_docs:
            print(f"warning: no documents found for item {id}. skipping...")
            continue
            
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

    # only compute metrics if we have results to evaluate
    if questions:
        avg_recall = RetrieverEvaluator().compute(retrieved_contexts, gt_contexts)
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
    else:
        print("no questions were processed. please check if documents were loaded correctly.")
