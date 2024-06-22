from lightrag.components.model_client import OpenAIClient
from lightrag.core.types import Document
from lightrag.components.retriever import LLMRetriever
from lightrag.core.string_parser import ListParser
from lightrag.components.data_process import (
    DocumentSplitter,
)

import dotenv

dotenv.load_dotenv(dotenv_path=".env", override=True)

# Document preparation and splitting
splitter_settings = {"split_by": "word", "split_length": 200, "split_overlap": 100}
text_splitter = DocumentSplitter(**splitter_settings)
documents = [
    Document(
        id="doc1",
        meta_data={"title": "Luna's Profile"},
        text="lots of more nonsense text." * 50
        + "Luna is a domestic shorthair."
        + "lots of nonsense text." * 50
        + "Luna loves to eat Tuna."
        + "lots of nonsense text." * 50,
    ),
    Document(
        id="doc2",
        meta_data={"title": "Luna's Hobbies"},
        text="lots of more nonsense text." * 50
        + "Luna loves to eat lickable treats."
        + "lots of more nonsense text." * 50
        + "Luna loves to play cat wand."
        + "lots of more nonsense text." * 50
        + "Luna likes to sleep all the afternoon",
    ),
]

# split the documents
splitted_docs = text_splitter.call(documents)

# configure the model
gpt_model_kwargs = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
}
# set up the retriever
llm_retriever = LLMRetriever(
    top_k=1,
    model_client=OpenAIClient(),
    model_kwargs=gpt_model_kwargs,
    output_processors=ListParser(),
)

# build indexes for the splitted documents
llm_retriever.build_index_from_documents(documents=splitted_docs)

# set up queries
queries = ["what does luna like to eat?", "what does Luna look like?"]


# get the retrieved list of GeneratorOutput, each contains list of indices
llm_query_output = llm_retriever.retrieve(query_or_queries=queries)
# print(llm_query_indices)
print("*" * 50)
for query, result in zip(queries, llm_query_output):
    result = result.data  # get list of indices from generatoroutput
    print(f"Query: {query}")
    if result:
        # Retrieve the indices from the result
        document_indices = result
        for idx in document_indices:
            # Ensure the index is within the range of splitted_docs
            if idx < len(splitted_docs):
                doc = splitted_docs[idx]
                print(f"Document ID: {doc.id} - Title: {doc.meta_data['title']}")
                print(f"Text: {doc.text}")  # Print the first 200 characters
            else:
                print(f"Index {idx} out of range.")
    else:
        print("No documents retrieved for this query.")
    print("*" * 50)
