from typing import Any, Dict, List, Union, Optional

from core.component import (
    OpenAIEmbedder,
    Component,
    RetrieverOutput,
    FAISSRetriever,
    EmbedderOutput,
)
from core.openai_llm import OpenAIGenerator
from core.data_classes import Document, Chunk

# TODO: rewrite SentenceSplitter and other splitter classes
from llama_index.core.node_parser import SentenceSplitter


class Query(Component):
    """
    Allow to define query.
    #TODO: support batch processing
    """

    def __init__(self, query: str, session_id: str = None):
        self.query = query
        self.session_id = session_id


##############################################
# Main RAG class
# Configs and combines functional modules
# One settings per RAG instance instead of global settings
##############################################
class RAG(Component):
    """
    TODO: design a base class later
    inputs: A list of documents [Can potentially use pandas dataframe for more complex data]
    Focus on retrieving for now
    """

    def __init__(self, settings: dict = {}):
        super().__init__()
        self.settings = settings
        # TODO: it could be more consistent to call vectorizer as embedder
        self.vectorizer = OpenAIEmbedder(
            provider=self.settings["vectorizer"]["provider"],
            model=self.settings["vectorizer"]["model"],
            encoding_format=self.settings["vectorizer"]["encoding_format"],
            dimensions=self.settings["vectorizer"]["embedding_size"],
        )
        self.retriever = FAISSRetriever(
            top_k=self.settings["retriever"]["top_k"],
            d=self.settings["vectorizer"]["embedding_size"],
            vectorizer=self.vectorizer,
        )

        self.generator = OpenAIGenerator(
            model=self.settings["generator"]["model"], temperature=0.3
        )
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}
        self.llm_task_desc = """
            You are a helpful assistant.

            Your task is to answer the query that may or may not come with context information.
            When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
            """

    def _load_documents(self, documents: List[Document]):
        self.documents = documents

    def _chunk_documents(self):
        self.chunks: List[Chunk] = []
        text_splitter = None
        text_splitter_settings = self.settings["text_splitter"]
        # TODO: wrap up text splitter into a class with __call__ method
        if text_splitter_settings["type"] == "sentence_splitter":
            text_splitter: SentenceSplitter = SentenceSplitter(
                chunk_size=text_splitter_settings["chunk_size"],
                chunk_overlap=text_splitter_settings["chunk_overlap"],
            )

        for doc in self.documents:
            chunks = text_splitter.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                self.chunks.append(Chunk(vector=[], text=chunk, order=i, doc_id=doc.id))

    def _vectorize_chunks(self):
        """
        TODO: what happens when the text is None or too long
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer is not set")
        batch_size = (
            50
            if "batch_size" not in self.settings["vectorizer"]
            else self.settings["vectorizer"]["batch_size"]
        )

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            embedder_output: EmbedderOutput = self.vectorizer(
                [chunk.text for chunk in batch]
            )
            vectors = embedder_output.embeddings
            for j, vector in enumerate(vectors):
                self.chunks[i + j].vector = vector

            # update tracking
            self.tracking["vectorizer"]["num_calls"] += 1
            self.tracking["vectorizer"][
                "num_tokens"
            ] += embedder_output.usage.total_tokens

    def build_index(self, documents: List[Document]):
        self._load_documents(documents)
        self._chunk_documents()
        self._vectorize_chunks()
        self.retriever.set_chunks(self.chunks)
        print(
            f"Index built with {len(self.chunks)} chunks, {len(self.documents)} documents, ready for retrieval"
        )

    def retrieve(
        self, query_or_queries: Union[str, List[str]]
    ) -> Union[RetrieverOutput, List[RetrieverOutput]]:
        if not self.retriever:
            raise ValueError("Retriever is not set")
        retrieved = self.retriever(query_or_queries)
        if isinstance(query_or_queries, str):
            return retrieved[0] if retrieved else None
        return retrieved

    @staticmethod
    def retriever_output_to_context_str(
        retriever_output: Union[RetrieverOutput, List[RetrieverOutput]],
        deduplicate: bool = False,
    ) -> str:
        """
        How to combine your retrieved chunks into the context is highly dependent on your use case.
        If you used query expansion, you might want to deduplicate the chunks.
        """
        chunks_to_use: List[Chunk] = []
        context_str = ""
        sep = " "
        if isinstance(retriever_output, RetrieverOutput):
            chunks_to_use = retriever_output.chunks
        else:
            for output in retriever_output:
                chunks_to_use.extend(output.chunks)
        if deduplicate:
            unique_chunks_ids = set([chunk.id for chunk in chunks_to_use])
            # id and if it is used, it will be True
            used_chunk_in_context_str: Dict[Any, bool] = {
                id: False for id in unique_chunks_ids
            }
            for chunk in chunks_to_use:
                if not used_chunk_in_context_str[chunk.id]:
                    context_str += sep + chunk.text
                    used_chunk_in_context_str[chunk.id] = True
        else:
            context_str = sep.join([chunk.text for chunk in chunks_to_use])
        return context_str

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        # system_prompt_template = Template(DEFAULT_QA_PROMPT)
        # context_str = context if context else ""
        # query_str = query
        # system_prompt_content = system_prompt_template.render(
        #     context_str=context_str, query_str=query_str
        # )
        # messages = [
        #     {"role": "system", "content": system_prompt_content},
        # ]
        # print(f"messages: {messages}")
        response = self.generator.call(
            input=query, context_str=context, task_desc_str=self.llm_task_desc
        )
        return response

    # The key RAG flow
    def call(self, query: str, context: str) -> str:
        pass


import yaml

if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
    with open("/Users/mengliu/Dropbox/Startup/LightRAG/core/config.yaml", "r") as file:
        settings = yaml.safe_load(file)
    print(settings)
    doc1 = Document(
        meta_data={"title": "Li Yin's profile"},
        text="My name is Li Yin, I love rock climbing" + "lots of nonsense text" * 1000,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "Interviewing Li Yin"},
        text="lots of more nonsense text" * 500
        + "Li Yin is a software developer and AI researcher"
        + "lots of more nonsense text" * 500,
        id="doc2",
    )
    rag = RAG(settings=settings)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    query = "What is Li Yin's hobby and profession?"
    # in this simple case, query expansion is not necessary, this is only for demonstration the list input of queries
    # the anaswer will only be right if each expended query has the right relevant chunk as set top_k to 1
    expanded_queries = ["Li Yin's hobby", "Li Yin's profession"]
    outputs = rag.retrieve(expanded_queries)
    print(f"retrieved: {outputs}")
    context_str = rag.retriever_output_to_context_str(outputs)
    response = rag.generate(query, context_str)
    print(f"response: {response}")
    # now try to set top_k to 2, and see if the answer is still correct
    # or set chunk_size to 20, chunk_overlap to 10, and see if the answer is still correct
    # you can try to fit all the context into the prompt, use long-context LLM.
