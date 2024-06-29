r"""We let users install the required SDKs conditionally for our integrated model providers."""

from lightrag.utils.registry import EntityMapping
from lightrag.utils import LazyImport, OptionalPackages

CohereAPIClient = LazyImport(
    "lightrag.components.model_client.cohere_client.CohereAPIClient",
    OptionalPackages.COHERE,
)
TransformerReranker = LazyImport(
    "lightrag.components.model_client.transformers_client.TransformerReranker",
    OptionalPackages.TRANSFORMERS,
)
TransformerEmbedder = LazyImport(
    "lightrag.components.model_client.transformers_client.TransformerEmbedder",
    OptionalPackages.TRANSFORMERS,
)
TransformerLLM = LazyImport(
    "lightrag.components.model_client.transformers_client.TransformerLLM",
    OptionalPackages.TRANSFORMERS,
)
TransformersClient = LazyImport(
    "lightrag.components.model_client.transformers_client.TransformersClient",
    OptionalPackages.TRANSFORMERS,
)
AnthropicAPIClient = LazyImport(
    "lightrag.components.model_client.anthropic_client.AnthropicAPIClient",
    OptionalPackages.ANTHROPIC,
)
GroqAPIClient = LazyImport(
    "lightrag.components.model_client.groq_client.GroqAPIClient",
    OptionalPackages.GROQ,
)
OpenAIClient = LazyImport(
    "lightrag.components.model_client.openai_client.OpenAIClient",
    OptionalPackages.OPENAI,
)
get_first_message_content = LazyImport(
    "lightrag.components.model_client.openai_client.get_first_message_content",
    OptionalPackages.OPENAI,
)
get_all_messages_content = LazyImport(
    "lightrag.components.model_client.openai_client.get_all_messages_content",
    OptionalPackages.OPENAI,
)
get_probabilities = LazyImport(
    "lightrag.components.model_client.openai_client.get_probabilities",
    OptionalPackages.OPENAI,
)


__all__ = [
    "CohereAPIClient",
    "TransformerReranker",
    "TransformerEmbedder",
    "TransformerLLM",
    "TransformersClient",
    "AnthropicAPIClient",
    "GroqAPIClient",
    "OpenAIClient",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
