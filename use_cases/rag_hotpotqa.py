from typing import Any, List, Union
import dotenv
import yaml

from datasets import load_dataset

from core.openai_client import OpenAIClient
from core.generator import Generator
from core.embedder import Embedder
from core.documents_data_class import Document
from core.data_components import (
    ToEmbedderResponse,
    RetrieverOutputToContextStr,
    ToEmbeddings,
)

from core.document_splitter import DocumentSplitter
from core.string_parser import JsonParser
from core.component import Component, Sequential
from core.retriever import FAISSRetriever
from core.db import LocalDocumentDB
from core.evaluator import RetrieverEvaluator

dotenv.load_dotenv(dotenv_path=".env", override=True)


# TODO: RAG can potentially be a component itsefl and be provided to the users
class RAG(Component):

    def __init__(self, settings: dict):
        super().__init__()
        self.vectorizer_settings = settings["vectorizer"]
        self.retriever_settings = settings["retriever"]
        self.generator_model_kwargs = settings["generator"]
        self.text_splitter_settings = settings["text_splitter"]

        self.vectorizer = Embedder(
            model_client=OpenAIClient(),
            # batch_size=self.vectorizer_settings["batch_size"],
            model_kwargs=self.vectorizer_settings["model_kwargs"],
            output_processors=Sequential(ToEmbedderResponse()),
        )
        text_splitter = DocumentSplitter(
            split_by=self.text_splitter_settings["split_by"],
            split_length=self.text_splitter_settings["chunk_size"],
            split_overlap=self.text_splitter_settings["chunk_overlap"],
        )
        data_transformer = Sequential(
            text_splitter,
            ToEmbeddings(
                vectorizer=self.vectorizer,
                batch_size=self.vectorizer_settings["batch_size"],
            ),
        )
        self.db = LocalDocumentDB(data_transformer=data_transformer)
        # initialize retriever, which depends on the vectorizer too
        self.retriever = FAISSRetriever(
            top_k=self.retriever_settings["top_k"],
            dimensions=self.vectorizer_settings["model_kwargs"]["dimensions"],
            vectorizer=self.vectorizer,
            # db=self.db,
            output_processors=RetrieverOutputToContextStr(deduplicate=True),
        )
        # initialize generator
        self.generator = Generator(
            model_client=OpenAIClient(),
            output_processors=Sequential(JsonParser()),
            preset_prompt_kwargs={
                "task_desc_str": r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""
            },
            model_kwargs=self.generator_model_kwargs,
        )
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}

    def build_index(self, documents: List[Document]):
        self.db.load_documents(documents)
        self.db()  # transform the documents
        self.retriever.set_chunks(self.db.transformed_documents)

    def call(self, query: str) -> Any:
        context_str = self.retriever(query)
        return (
            self.generator(input=query, prompt_kwargs={"context_str": context_str}),
            context_str,
        )


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


if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
    with open("./configs/rag_hotpotqa.yaml", "r") as file:
        settings = yaml.safe_load(file)
    print(settings)

    # Load the dataset and select the first 10 as the showcase
    # More info about the HotpotQA dataset can be found at https://huggingface.co/datasets/hotpot_qa
    dataset = load_dataset("hotpot_qa", "fullwiki")
    dataset = dataset["train"].select(range(10))

    all_retrieved_context = []
    all_gt_context = []
    for data in dataset:
        # Each sample in HotpotQA has multiple documents to retrieve from. Each document has a title and a list of sentences.
        num_docs = len(data["context"]["title"])
        doc_list = [
            Document(
                meta_data=data["context"]["title"][i],
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

        all_retrieved_context.append(context_str)
        all_gt_context.append(gt_context_sentence_list)
        print("====================================================")
        print(f"query: {query}")
        print(f"response: {response['answer']}")
        print(f"ground truth response: {data['answer']}")
        print(f"context_str: {context_str}")
        print(f"ground truth context_str: {gt_context_sentence_list}")
        print("====================================================")

    retriever_evaluator = RetrieverEvaluator()
    avg_recall, recall_list = retriever_evaluator.compute_recall(
        all_retrieved_context, all_gt_context
    )
    avg_relevance, relevance_list = retriever_evaluator.compute_context_relevance(
        all_retrieved_context, all_gt_context
    )
    print(f"Average recall: {avg_recall}")
    print(f"Recall for each query: {recall_list}")
    print(f"Average relevance: {avg_relevance}")
    print(f"Relevance for each query: {relevance_list}")
