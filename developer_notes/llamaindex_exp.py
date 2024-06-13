from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever


from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

nodes = [
    NodeWithScore(node=Node(text="text1"), score=0.7),
    NodeWithScore(node=Node(text="text2"), score=0.8),
]

nodes = [
    Node(text="Li"),
    Node(text="text2 Li"),
]
retriever = LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=1)

# from llama_index.core.response.notebook_utils import display_source_node

# will retrieve context from specific companies
nodes = retriever.retrieve("Li?")
print(nodes)


from rank_bm25 import BM25Okapi

corpus = [
    "Li",
    "text2 Li",
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
query = "Li"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

from lightrag.components.retriever import InMemoryBM25Retriever, split_text_by_word_fn

retriever = InMemoryBM25Retriever(top_k=1, split_function=split_text_by_word_fn)


retriever.build_index_from_documents(documents=corpus)
output = retriever.retrieve(query)
print(output)
