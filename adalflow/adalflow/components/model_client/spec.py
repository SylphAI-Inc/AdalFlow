"""Model specs: ``"provider/model"`` strings in place of a client plus model_kwargs.

    client, kwargs = resolve_model("anthropic/claude-haiku-5-5")
    Generator(model_client=client, model_kwargs=kwargs, ...)

The client is created with its default settings, which read API keys from the usual
environment variables. Pass a ``ModelClient`` instead of a string to configure it
yourself.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

__all__ = ["PROVIDERS", "resolve_model"]

# provider prefix -> client class name in adalflow.components.model_client
PROVIDERS: Dict[str, str] = {
    "openai": "OpenAIClient",
    "anthropic": "AnthropicAPIClient",
    "google": "GoogleGenAIClient",
    "groq": "GroqAPIClient",
    "ollama": "OllamaClient",
    "together": "TogetherClient",
    "deepseek": "DeepSeekClient",
    "mistral": "MistralClient",
    "xai": "XAIClient",
    "fireworks": "FireworksClient",
    "sambanova": "SambaNovaClient",
    "bedrock": "BedrockAPIClient",
    "azure": "AzureAIClient",
    "cohere": "CohereAPIClient",
}


def resolve_model(
    model: Union[str, Any], model_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Returns ``(model_client, model_kwargs)`` for a spec or an existing client.

    Args:
        model: ``"provider/model-name"`` (the name may itself contain ``/``, as in
            ``"together/meta-llama/Llama-3-8b"``), or a ``ModelClient`` instance.
        model_kwargs: extra settings (``temperature``...). For a string spec,
            ``model`` is filled in from the spec.
    """
    kwargs = dict(model_kwargs or {})
    if not isinstance(model, str):
        from adalflow.core.model_client import ModelClient

        if not isinstance(model, ModelClient):
            raise TypeError(
                f"model must be a 'provider/model' string or a ModelClient, "
                f"got {type(model).__name__}"
            )
        return model, kwargs
    provider, sep, name = model.partition("/")
    if not sep or not provider or not name:
        raise ValueError(
            f"model spec {model!r} must look like 'provider/model', "
            f"e.g. 'openai/gpt-4o-mini'"
        )
    cls_name = PROVIDERS.get(provider.lower())
    if cls_name is None:
        raise ValueError(
            f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}. "
            "Pass a ModelClient instance for others."
        )
    from adalflow.components import model_client as clients

    client = getattr(clients, cls_name)()
    return client, {"model": name, **kwargs}
