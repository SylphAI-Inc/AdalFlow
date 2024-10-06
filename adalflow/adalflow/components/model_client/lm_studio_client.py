import os
import re
from typing import Dict, Optional, Any, List, Union, Sequence
import logging
import backoff
import requests
from adalflow.core.types import ModelType, GeneratorOutput, EmbedderOutput, Embedding, Usage
from adalflow.core.model_client import ModelClient

log = logging.getLogger(__name__)

class LMStudioClient(ModelClient):
    """A component wrapper for the LM Studio API client."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        super().__init__()
        self._host = host or os.getenv("LMSTUDIO_HOST", "http://localhost")
        self._port = port or int(os.getenv("LMSTUDIO_PORT", "1234"))
        self._base_url = f"{self._host}:{self._port}/v1"
        self.init_sync_client()
        self.async_client = None  # To be added

    def init_sync_client(self):
        """Create the synchronous client"""
        self.sync_client = requests.Session()

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert the input and model_kwargs to api_kwargs for the LM Studio API."""
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            assert isinstance(input, Sequence), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            messages = []
            if input is not None and input != "":
                messages.append({"role": "system", "content": "You are a helpful assistant. Provide a direct and concise answer to the user's question. Do not include any URLs or references in your response."})
                messages.append({"role": "user", "content": input})
            assert isinstance(messages, Sequence), "input must be a sequence of messages"
            final_model_kwargs["messages"] = messages
            
            # Set default values for controlling response length if not provided
            final_model_kwargs.setdefault("max_tokens", 50)
            final_model_kwargs.setdefault("temperature", 0.1)
            final_model_kwargs.setdefault("top_p", 0.9)
            final_model_kwargs.setdefault("frequency_penalty", 0.0)
            final_model_kwargs.setdefault("presence_penalty", 0.0)
            final_model_kwargs.setdefault("stop", ["\n", "###", "://"])
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @backoff.on_exception(backoff.expo, requests.RequestException, max_time=10)
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type == ModelType.EMBEDDER:
            response = self.sync_client.post(f"{self._base_url}/embeddings", json=api_kwargs)
        elif model_type == ModelType.LLM:
            response = self.sync_client.post(f"{self._base_url}/chat/completions", json=api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        
        response.raise_for_status()
        return response.json()

    def parse_chat_completion(self, completion: Dict) -> GeneratorOutput:
        """Parse the completion to a GeneratorOutput."""
        if "choices" in completion and len(completion["choices"]) > 0:
            content = completion["choices"][0]["message"]["content"]
            
            # Clean up the content
            content = self._clean_response(content)
            
            return GeneratorOutput(data=None, raw_response=content)
        else:
            log.error(f"Error parsing the completion: {completion}")
            return GeneratorOutput(data=None, error="Error parsing the completion", raw_response=completion)

    def _clean_response(self, content: str) -> str:
        """Clean up the response content."""
        # Remove any URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Remove any content after "###" or "://"
        content = re.split(r'###|://', content)[0]
        
        # Remove any remaining HTML-like tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove any repeated information
        sentences = content.split('.')
        unique_sentences = []
        for sentence in sentences:
            if sentence.strip() and sentence.strip() not in unique_sentences:
                unique_sentences.append(sentence.strip())
        content = '. '.join(unique_sentences)
        
        return content.strip()

    def parse_embedding_response(self, response: Dict) -> EmbedderOutput:
        """Parse the embedding response to an EmbedderOutput."""
        try:
            embeddings = [Embedding(embedding=data["embedding"], index=i) for i, data in enumerate(response["data"])]
            usage = Usage(
                prompt_tokens=response["usage"]["prompt_tokens"],
                total_tokens=response["usage"]["total_tokens"]
            )
            return EmbedderOutput(data=embeddings, model=response["model"], usage=usage)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """LM Studio doesn't support async calls natively, so we use the sync method."""
        return self.call(api_kwargs, model_type)
