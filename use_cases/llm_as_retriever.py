from components.retriever.llm_retriever import LLMRetriever
from core.generator import Generator
from core.openai_client import OpenAIClient
from core.data_classes import Document
import dotenv

dotenv.load_dotenv(dotenv_path=".env", override=True)


def test_llm_retriever():
    generator_kwargs = {
        "model_client": OpenAIClient(),
        "model_kwargs": {"model": "gpt-3.5-turbo"},
    }

    retriever = LLMRetriever(generator_kwargs=generator_kwargs, top_k=1)
    print(retriever)
    # build index
    documents = [
        Document(text="Paris is the capital of France."),
        Document(text="Berlin is the capital of Germany."),
    ]
    retriever.build_index_from_documents(documents)

    response = retriever("What can you about Paris?")
    print(response)


if __name__ == "__main__":
    test_llm_retriever()
