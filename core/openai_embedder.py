from typing import Dict, List, Union
from openai import OpenAI
import os
import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
from core.data_classes import EmbedderOutput
from core.embedder import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(
        self, *, provider: str = "OpenAI", batch_size: int = 1, model_kwargs: Dict = {}
    ) -> None:
        super().__init__(provider=provider, model_kwargs=model_kwargs)
        self.batch_size = batch_size  # TODO: check the scope of the embedder

        self._init_sync_client()

    def _init_sync_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.sync_client = OpenAI()

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance
        """
        text = text.replace("\n", " ")
        return text

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    def call(
        self,
        input: Union[str, List[str]],
        **model_kwargs,  # overwrites the default kwargs
    ) -> EmbedderOutput:
        """
        Automatically handles retries for the above exceptions
        TODO: support async calls
        """
        formulated_inputs = []
        if isinstance(input, str):
            formulated_inputs.append(self._process_text(input))
        else:
            for query in input:
                formulated_inputs.append(self._process_text(query))

        num_queries = len(formulated_inputs)

        # check overrides for kwargs
        pass_model_kwargs = self.compose_model_kwargs(**model_kwargs)

        print(f"kwargs: {pass_model_kwargs}")

        response = self.sync_client.embeddings.create(
            input=formulated_inputs, **pass_model_kwargs
        )
        usage = response.usage
        embeddings = [data.embedding for data in response.data]
        assert (
            len(embeddings) == num_queries
        ), f"Number of embeddings {len(embeddings)} is not equal to the number of queries {num_queries}"
        return EmbedderOutput(embeddings=embeddings, usage=usage)
