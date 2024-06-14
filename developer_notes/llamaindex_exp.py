# from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever


# from llama_index.core.data_structs import Node
# from llama_index.core.schema import NodeWithScore

# nodes = [
#     NodeWithScore(node=Node(text="text1"), score=0.7),
#     NodeWithScore(node=Node(text="text2"), score=0.8),
# ]

# nodes = [
#     Node(text="Li"),
#     Node(text="text2 Li"),
# ]
# retriever = LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=1)

# # from llama_index.core.response.notebook_utils import display_source_node

# # will retrieve context from specific companies
# nodes = retriever.retrieve("Li?")
# print(nodes)


# from rank_bm25 import BM25Okapi

# corpus = [
#     "text1",
#     "Li",
# ]
# tokenized_corpus = [doc.split(" ") for doc in corpus]

# bm25 = BM25Okapi(tokenized_corpus)
# query = "Li"
# tokenized_query = query.split(" ")

# doc_scores = bm25.get_scores(tokenized_query)
# print(doc_scores)

# from lightrag.components.retriever import InMemoryBM25Retriever, split_text_by_word_fn
# from lightrag.utils import save_json, load_json
# from lightrag.utils import enable_library_logging

# retriever = InMemoryBM25Retriever(top_k=2, split_function=split_text_by_word_fn)


# # retriever.build_index_from_documents(documents=corpus)
# # index = retriever.get_index()
# # print(index)
# # save_path = "developer_notes/data/in_memory_bm25_index.json"
# # save_json(index, save_path)
# index = load_json("developer_notes/data/in_memory_bm25_index.json")
# retriever.load_index(index)
# query = "Li"
# output = retriever.retrieve(query)
# print(output)
# # output = retriever.retrieve(query)
# # print(output)

# # index = retriever.get_index()
# # print(index)
# from lightrag.components.model_client import OpenAIClient
# from lightrag.utils import setup_env
# from lightrag.components.retriever import LLMRetriever
# from lightrag.tracing import trace_generator_call


# @trace_generator_call(save_dir="developer_notes/traces")
# class LoggedLLMRetriever(LLMRetriever):
#     pass


# # LLMRetriever = trace_generator_call(LLMRetriever)

# # print(f"LLMRetriever: {LLMRetriever}")


# retriever = LoggedLLMRetriever(
#     top_k=1, model_client=OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
# )

# retriever.build_index_from_documents(documents=corpus)  # index are the documents

# print(retriever.generator())
# output = retriever.retrieve(query)
# print(output)
# # import os

from lightrag.components.retriever import RerankerRetriever

query = "Li"
documents = ["Li", "text2"]

retriever = RerankerRetriever(top_k=1)
print(retriever)
retriever.build_index_from_documents(documents=documents)
print(retriever.documents)
output = retriever.retrieve(query)
print(output)


# enable_library_logging()
# path = os.path.join(
#     get_script_dir(), "data", f"{retriever.__class__.__name__}_index.json"
# )
# print(path)
# save_json(index, path)
