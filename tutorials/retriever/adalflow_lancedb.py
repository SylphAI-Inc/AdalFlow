from adalflow.components.retriever.lancedb_retriver import LanceDBRetriever

from adalflow.core.embedder import Embedder
from adalflow.core.types import ModelClientType

model_kwargs = {
    "model": "text-embedding-3-small",
    "dimensions": 256,
    "encoding_format": "float",
}

documents = [
    {
        "title": "The Impact of Renewable Energy on the Economy",
        "content": "Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs.",
    },
    {
        "title": "Understanding Solar Panels",
        "content": "Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms.",
    },
    {
        "title": "Pros and Cons of Solar Energy",
        "content": "While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides.",
    },
    {
        "title": "Renewable Energy and Its Effects",
        "content": "Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change.",
    },
]


def init_retriever():
    embedder = Embedder(
        model_client=ModelClientType.OPENAI(), model_kwargs=model_kwargs
    )
    retriever = LanceDBRetriever(
        embedder=embedder, dimensions=256, db_uri="/tmp/lancedb", top_k=2
    )
    retriever.add_documents(documents)
    return retriever


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()

    retriever = init_retriever()
    output = retriever.retrieve(query="What are the benefits of renewable energy?")
    print(output)
