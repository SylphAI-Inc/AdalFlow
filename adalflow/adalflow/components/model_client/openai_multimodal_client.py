"""OpenAI multimodal client for handling image and text inputs."""

import base64
from typing import Any, Dict, List, Optional, Union
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])
from openai import OpenAI

from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput


class OpenAIMultimodalClient(ModelClient):
    """OpenAI client for multimodal models."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI multimodal client.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded image string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _prepare_image_content(
        self, image_source: Union[str, Dict[str, Any]], detail: str = "auto"
    ) -> Dict[str, Any]:
        """Prepare image content for API request.

        Args:
            image_source: Either a path to local image or a URL.
            detail: Image detail level ('auto', 'low', or 'high').

        Returns:
            Formatted image content for API request.
        """
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                return {
                    "type": "image_url",
                    "image_url": {"url": image_source, "detail": detail},
                }
            else:
                base64_image = self._encode_image(image_source)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                }
        return image_source

    def generate(
        self,
        prompt: str,
        images: Optional[
            Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
        ] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """Generate text response for given prompt and images.

        Args:
            prompt: Text prompt.
            images: Image source(s) - can be path(s), URL(s), or formatted dict(s).
            model_kwargs: Additional model parameters.

        Returns:
            GeneratorOutput containing the model's response.
        """
        model_kwargs = model_kwargs or {}
        model = model_kwargs.get("model", "gpt-4o-mini")
        max_tokens = model_kwargs.get("max_tokens", 300)
        detail = model_kwargs.get("detail", "auto")

        # Prepare message content
        content = [{"type": "text", "text": prompt}]

        if images:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                content.append(self._prepare_image_content(img, detail))

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            return GeneratorOutput(
                id=response.id,
                data=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,
                raw_response=response.model_dump(),
            )
        except Exception as e:
            return GeneratorOutput(error=str(e))
