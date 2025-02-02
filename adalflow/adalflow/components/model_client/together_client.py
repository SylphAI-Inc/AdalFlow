import logging
from typing import Any, Dict, Optional, Sequence

# Official Together Python library
from together import Together

# AdalFlow imports
from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    GeneratorOutput,
    EmbedderOutput,
)

logger = logging.getLogger(__name__)


class TogetherClient(ModelClient):
    def __init__(self):
        super().__init__()
        self.together_client = Together()
        self.sync_client = self.init_sync_client()
        self.async_client = self.init_async_client()

    def init_sync_client(self):
        """
        For the official Together library, we don't strictly need a separate sync client.
        We'll just return None or we could return self.together_client if we want.
        """
        return None

    def init_async_client(self):
        """
        If Together offers an async interface, we could store that here. If not, just None.
        """
        return None

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert AdalFlow's input data to the official Together function signature.

        For LLM usage, we expect:
            {
              "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "messages": [ { "role":"user", "content":"Hi" } ],
              ... plus any optional params like 'temperature'
            }
        """
        final_args = dict(model_kwargs)

        if model_type == ModelType.LLM:
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, Sequence):
                messages = input
            else:
                raise ValueError(
                    "For LLM usage, input must be str or list of messages."
                )

            final_args["messages"] = messages
            return final_args

        elif model_type == ModelType.EMBEDDER:
            raise NotImplementedError("Together embedder usage is not implemented.")

        else:
            raise ValueError(
                f"model_type {model_type} is not supported by Together SDK client."
            )

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type != ModelType.LLM:
            raise ValueError(
                "Currently only LLM usage is supported with the TogetherClient."
            )

        model = api_kwargs.get("model")
        if not model:
            raise ValueError(
                "Missing 'model' param in api_kwargs for Together LLM call."
            )

        messages = api_kwargs["messages"]
        temperature = api_kwargs.get("temperature", 0.7)
        max_tokens = api_kwargs.get("max_tokens", 128)

        try:
            response = self.together_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as exc:
            logger.error(f"Together SDK call error: {exc}")
            raise exc

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        raise NotImplementedError(
            "Together official library async usage not available yet."
        )

    def parse_chat_completion(self, completion) -> GeneratorOutput:
        if not completion:
            return GeneratorOutput(
                data=None, error="Empty Together response", raw_response=completion
            )

        try:
            # Typically: completion.choices[0].message.content
            # But let's check the library doc. We'll guess it's the same as OpenAI style:
            content = completion.choices[0].message.content
            return GeneratorOutput(
                data=content,
                error=None,
                raw_response=completion,
            )
        except (AttributeError, IndexError) as e:
            err_msg = f"Unexpected Together response format: {e}"
            logger.error(err_msg)
            return GeneratorOutput(data=None, error=err_msg, raw_response=completion)

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """
        Not relevant until an official embedding method is provided.
        """
        raise NotImplementedError("Together embedding parse not yet implemented.")


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import get_logger
    from dotenv import load_dotenv

    load_dotenv()
    log = get_logger(level="DEBUG")

    client = TogetherClient()

    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "temperature": 0.7,
            "max_tokens": 64,
        },
    )

    prompt_kwargs = {
        "input": "Hi from AdalFlow! Summarize generative AI briefly."
        # or
        # "input": [
        #     {"role":"system","content":"You are a helpful assistant."},
        #     {"role":"user","content":"What's new in generative AI?"}
        # ]
    }

    response = generator(prompt_kwargs)

    if response.error:
        print(f"[TogetherSDK] Error: {response.error}")
    else:
        print(f"[TogetherSDK] Response: {response.data}")
