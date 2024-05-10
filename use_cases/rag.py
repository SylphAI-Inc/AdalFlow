from typing import Any, List
import dotenv
import yaml

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
        return self.generator(input=query, prompt_kwargs={"context_str": context_str})


if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
    with open("./configs/rag.yaml", "r") as file:
        settings = yaml.safe_load(file)
    print(settings)
    doc1 = Document(
        meta_data={"title": "Li Yin's profile"},
        text="My name is Li Yin, I love rock climbing" + "lots of nonsense text" * 500,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "Interviewing Li Yin"},
        text="lots of more nonsense text" * 250
        + "Li Yin is a software developer and AI researcher"
        + "lots of more nonsense text" * 250,
        id="doc2",
    )
    rag = RAG(settings)
    print(rag)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    query = "What is Li Yin's hobby and profession?"

    response = rag.call(query)
    print(f"response: {response}")
