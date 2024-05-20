from typing import List

from core.generator import Generator
from core.data_classes import Document
from core.embedder import Embedder
from core.data_components import (
    ToEmbedderResponse,
    RetrieverOutputToContextStr,
)
from core.db import LocalDocumentDB
from core.component import Component
from core.document_splitter import DocumentSplitter
from icl.retrieval_icl import RetrievalICL

from components.api_client import OpenAIClient
from components.retriever import FAISSRetriever


import dotenv

dotenv.load_dotenv(dotenv_path=".env", override=True)


class FewshotQA(Component):
    def __init__(self, task_desc: str):
        super().__init__()
        self.vectorizer_settings = {
            "batch_size": 100,
            "model_kwargs": {
                "model": "text-embedding-3-small",
                "dimensions": 256,
                "encoding_format": "float",
            },
        }
        self.retriever_settings = {
            "top_k": 2,
        }
        self.text_splitter_settings = {
            "split_by": "word",
            "chunk_size": 400,
            "chunk_overlap": 200,
        }
        model_kwargs = {"model": "gpt-3.5-turbo"}
        preset_prompt_kwargs = {"task_desc_str": task_desc}
        self.generator = Generator(
            model_client=OpenAIClient(),
            model_kwargs=model_kwargs,
            preset_prompt_kwargs=preset_prompt_kwargs,
        )

        text_splitter = DocumentSplitter(
            split_by=self.text_splitter_settings["split_by"],
            split_length=self.text_splitter_settings["chunk_size"],
            split_overlap=self.text_splitter_settings["chunk_overlap"],
        )
        vectorizer = Embedder(
            model_client=OpenAIClient(),
            model_kwargs=self.vectorizer_settings["model_kwargs"],
            output_processors=ToEmbedderResponse(),
        )
        self.retriever = FAISSRetriever(
            top_k=self.retriever_settings["top_k"],
            dimensions=self.vectorizer_settings["model_kwargs"]["dimensions"],
            vectorizer=vectorizer,
        )

        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
        self.db = LocalDocumentDB()
        self.retrieval_icl = RetrievalICL(
            retriever=self.retriever,
            retriever_output_processors=self.retriever_output_processors,
            text_splitter=text_splitter,
            vectorizer=vectorizer,
            db=self.db,
        )

    def build_icl_index(self, documents: List[Document]):
        self.retrieval_icl.build_index(documents)

    def call(self, query: str, top_k: int) -> str:
        example_str = self.retrieval_icl(query, top_k=top_k)
        return (
            self.generator.call(
                input=query,
                prompt_kwargs={"task_desc": task_desc, "example_str": example_str},
            ),
            example_str,
        )


if __name__ == "__main__":
    task_desc = "Classify the sentiment of the following reviews as either Positive or Negative."

    example1 = Document(
        text="Review: I absolutely loved the friendly staff and the welcoming atmosphere! Sentiment: Positive",
    )
    example2 = Document(
        text="Review: It was an awful experience, the food was bland and overpriced. Sentiment: Negative",
    )
    example3 = Document(
        text="Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: Positive",
    )
    example4 = Document(
        text="Review: The store is not clean and smells bad. Sentiment: Negative",
    )

    fewshot_qa = FewshotQA(task_desc)
    # build the index for the retriever-based ICL
    fewshot_qa.build_icl_index([example1, example2, example3, example4])
    print(fewshot_qa)
    query = (
        "Review: The concert was a lot of fun and the band was energetic and engaging."
    )
    # tok_k: how many examples you want retrieve to show to the model
    response, example_str = fewshot_qa(query, top_k=2)
    print(f"response: {response}")
    print(f"example_str: {example_str}")
