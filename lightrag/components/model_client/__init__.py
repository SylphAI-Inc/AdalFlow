r"""We let users install the required SDKs conditionally for our integrated model providers."""

from lightrag.utils.registry import EntityMapping
import logging

log = logging.getLogger(__name__)

__all__ = []

try:
    from .openai_client import OpenAIClient

    __all__.append("OpenAIClient")
except ImportError as e:
    log.info(f"Optional module not installed: {e}")

try:
    from .groq_client import GroqAPIClient

    __all__.append("GroqAPIClient")
except ImportError as e:
    log.info(f"Optional module not installed: {e}")

try:
    from .anthropic_client import AnthropicAPIClient

    __all__.append("AnthropicAPIClient")

except ImportError as e:
    log.info(f"Optional module not installed: {e}")

try:
    from .transformers_client import (
        TransformersClient,
        TransformerEmbedder,
        TransformerReranker,
    )

    __all__.extend(["TransformersClient", "TransformerEmbedder", "TransformerReranker"])
except ImportError as e:
    log.info(f"Optional module not installed: {e}")

try:
    from .cohere_client import CohereAPIClient

    __all__.append("CohereAPIClient")
except ImportError as e:
    log.info(f"Optional module not installed: {e}")


for name in __all__:
    EntityMapping.register(name, globals()[name])
