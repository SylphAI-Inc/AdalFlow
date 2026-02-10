from typing import Any, List, Optional
import os
import logging
from adalflow.core import Component, Generator, Embedder, Sequential
from adalflow.core.types import Document, ModelClientType
from adalflow.core.string_parser import JsonParser
from adalflow.core.db import LocalDB
from adalflow.utils import setup_env

from adalflow.components.retriever.faiss_retriever import FAISSRetriever

from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)
from adalflow.utils.global_config import get_adalflow_default_root_path

setup_env()
# TODO: RAG can potentially be a component itsefl and be provided to the users

log = logging.getLogger(__name__)

configs = {
    "embedder": {
        "batch_size": 100,
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 5,
    },
    "generator": {
        "model_client": ModelClientType.OPENAI(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "stream": False,
        },
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    },
}


# use data process complete that will transform on Document structure
def prepare_data_pipeline():
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = Embedder(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = Sequential(splitter, embedder_transformer)
    return data_transformer


def prepare_database_with_index(
    docs: List[Document],
    index_file: str = "index.faiss",
    index_path: Optional[str] = None,
):
    index_path = index_path or get_adalflow_default_root_path()
    index_path = os.path.join(index_path, index_file)
    if os.path.exists(index_path):
        return None
    db = LocalDB()
    db.load(docs)
    data_transformer = prepare_data_pipeline()
    db.transform(data_transformer, key="data_transformer")
    # store
    db.save_state(index_path)


RAG_PROMPT_TEMPLATE = r"""<START_OF_SYSTEM_MESSAGE>
{{task_desc}}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER>
{{input_str}}
{{context_str}}
<END_OF_USER>
"""

rag_prompt_task_desc = r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""


class RAG(Component):

    def __init__(
        self,
        index_file: str = "index.faiss",
        index_path: Optional[str] = None,
        # model_client: ModelClient = ModelClientType.OPENAI(),
        # model_kwargs: dict = None,
        configs: dict = configs,
    ):
        super().__init__()

        # 1. it can restore data from existing storage
        index_path = index_path or get_adalflow_default_root_path()
        index_path = os.path.join(index_path, index_file)
        self.index_path = index_path
        if not os.path.exists(index_path):
            self.db = LocalDB()
            self.register_data_transformer()
            self.transformed_docs = []
        else:
            self.db = LocalDB.load_state(index_path)
            self.transformed_docs: List[Document] = self.db.get_transformed_data(
                "data_transformer"
            )
        embedder = Embedder(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )
        # map the documents to embeddings
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)

        self.generator = Generator(
            **configs["generator"],
            prompt_kwargs={
                "task_desc_str": rag_prompt_task_desc,
            },
            output_processors=JsonParser(),
        )

    def register_data_transformer(self):
        if "data_transformer" not in self.db.get_transformer_keys():
            data_transformer = prepare_data_pipeline()
            self.db.register_transformer(data_transformer, key="data_transformer")
            print("Data transformer registered")

    def add_documents(self, docs: List[Document]):
        self.db.extend(docs, apply_transformer=True)
        # save the state
        self.db.save_state(self.index_path)

    def get_transformed_docs(self, filter_func=None):
        # fix: use keyword arguments to match the expected method signature
        return self.db.get_transformed_data(key="data_transformer", filter_fn=filter_func)

    def prepare_retriever(self, filter_func=None):
        # get filtered documents for this specific query
        self.transformed_docs = self.get_transformed_docs(filter_func)
        
        # handle case where no documents match the filter
        if not self.transformed_docs:
            log.warning("no documents found matching the filter criteria")
            return
            
        # build the retriever index from the filtered documents
        self.retriever.build_index_from_documents(
            self.transformed_docs, document_map_func=lambda doc: doc.vector
        )

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        prompt_kwargs = {
            "context_str": context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        return response, context

    def call(self, query: str, verbose: bool = False) -> Any:
        retrieved_documents = self.retriever(query)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]
        if verbose:
            print(f"retrieved_documents: \n {retrieved_documents}")
        context_str = self.retriever_output_processors(retrieved_documents)

        if verbose:
            print(f"context_str: \n {context_str}")

        return self.generate(query, context=context_str)


if __name__ == "__main__":
    doc1 = Document(
        meta_data={"title": "Li Yin's profile"},
        text="My name is Li Yin, I love rock climbing" + "lots of nonsense text" * 500,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "Interviewing Li Yin"},
        text="lots of more nonsense text" * 250
        + "Li Yin is an AI researcher and a software engineer"
        + "lots of more nonsense text" * 250,
        id="doc2",
    )
    # only run it once to prepare the data, if index exists, it will not run
    prepare_database_with_index([doc1, doc2], index_file="index.faiss")
    rag = RAG(index_file="index.faiss")
    print(rag)
    query = "What is Li Yin's hobby and profession?"

    response = rag.call(query)

    print(f"response: {response}")

    doc3 = Document(
        meta_data={"title": "Apple's profile"},
        text="Apple is a cute dog with black and tan fur"
        + "lots of nonsense text" * 500,
        id="doc3",
    )
    doc4 = Document(
        meta_data={"title": "Apple's characteristics"},
        text="lots of more nonsense text" * 250
        + "Apple is energetic, loves to play with her monkey toy"
        + "lots of more nonsense text" * 250,
        id="doc4",
    )
    # Add more documents to the database at runtime
    rag.add_documents([doc3, doc4])
    rag.prepare_retriever()
    response = rag.call("What is Apple's favorite toy?")
    print(f"response: {response}")
    print(rag.db.items)

    # If you want to run a section time, please delete the index file or else there will be redundant data
