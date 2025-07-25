from adalflow.utils import setup_env
from adalflow.components.agent import Agent, Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.retriever import Retriever

setup_env()

class DocumentRetriever(Retriever):
    def __init__(self, documents: list):
        super().__init__()
        self.documents = documents

    def call(self, input: str, top_k: int = 3):
        # Simple similarity search implementation
        # In practice, use vector embeddings
        relevant_docs = [doc for doc in self.documents if input.lower() in doc.lower()]
        return relevant_docs[:top_k]

# Setup
documents = [
    "Python is a programming language.",
    "Machine learning uses algorithms to learn patterns.",
    "AdalFlow is a framework for building AI applications."
]

retriever = DocumentRetriever(documents)

agent = Agent(
    name="RAGAgent",
    tools=[FunctionTool(retriever.call)],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o"},
    max_steps=5
)

runner = Runner(agent=agent)

result = runner.call(
    prompt_kwargs={"input_str": "What is AdalFlow?"}
)