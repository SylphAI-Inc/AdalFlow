from lightrag.core.types import Document

from lightrag.components.retriever import LLMRetriever
from lightrag.components.model_client import OpenAIClient

import utils.setup_env


def test_llm_retriever():
    # TODO: directly pass Generator class is more intuitive than the generator_kwargs

    retriever = LLMRetriever(
        top_k=1, model_client=OpenAIClient, model_kwargs={"model": "gpt-3.5-turbo"}
    )
    print(retriever)
    documents = [
        Document(text="Paris is the capital of France."),
        Document(text="Berlin is the capital of Germany."),
    ]
    retriever.build_index_from_documents(documents)
    retriever.print_prompt()

    response = retriever("What do you know about Paris?")
    print(response)


if __name__ == "__main__":
    test_llm_retriever()
