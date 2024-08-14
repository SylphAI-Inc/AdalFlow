r"""We let users install the required SDKs conditionally for our integrated model providers."""

from adalflow.utils.registry import EntityMapping
from adalflow.utils.lazy_import import (
    LazyImport,
    OptionalPackages,
)

# NOTE: Do not subclass lazy imported classes, it will cause issues with the lazy import mechanism.
# Instead, directly import the class from its specif module and use it.
CohereAPIClient = LazyImport(
    "adalflow.components.model_client.cohere_client.CohereAPIClient",
    OptionalPackages.COHERE,
)
TransformerReranker = LazyImport(
    "adalflow.components.model_client.transformers_client.TransformerReranker",
    OptionalPackages.TRANSFORMERS,
)
TransformerEmbedder = LazyImport(
    "adalflow.components.model_client.transformers_client.TransformerEmbedder",
    OptionalPackages.TRANSFORMERS,
)
TransformerLLM = LazyImport(
    "adalflow.components.model_client.transformers_client.TransformerLLM",
    OptionalPackages.TRANSFORMERS,
)
TransformersClient = LazyImport(
    "adalflow.components.model_client.transformers_client.TransformersClient",
    OptionalPackages.TRANSFORMERS,
)
AnthropicAPIClient = LazyImport(
    "adalflow.components.model_client.anthropic_client.AnthropicAPIClient",
    OptionalPackages.ANTHROPIC,
)
GroqAPIClient = LazyImport(
    "adalflow.components.model_client.groq_client.GroqAPIClient",
    OptionalPackages.GROQ,
)
OpenAIClient = LazyImport(
    "adalflow.components.model_client.openai_client.OpenAIClient",
    OptionalPackages.OPENAI,
)
GoogleGenAIClient = LazyImport(
    "adalflow.components.model_client.google_client.GoogleGenAIClient",
    OptionalPackages.GOOGLE_GENERATIVEAI,
)
OllamaClient = LazyImport(
    "adalflow.components.model_client.ollama_client.OllamaClient",
    OptionalPackages.OLLAMA,
)
get_first_message_content = LazyImport(
    "adalflow.components.model_client.openai_client.get_first_message_content",
    OptionalPackages.OPENAI,
)
get_all_messages_content = LazyImport(
    "adalflow.components.model_client.openai_client.get_all_messages_content",
    OptionalPackages.OPENAI,
)
get_probabilities = LazyImport(
    "adalflow.components.model_client.openai_client.get_probabilities",
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
    "GoogleGenAIClient",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
