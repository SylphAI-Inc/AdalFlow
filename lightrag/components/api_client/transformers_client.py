"""
There are two ways to run transformers:
(1) model and then run model inference
(2) Pipeline and then run pipeline inference

This file demonstrates how to 
(1) create a torch model inference component:  TransformerEmbedder which equalize to OpenAI(), the SyncAPIClient
(2) Convert this model inference component to LightRAG API client: TransformersClient
"""

from typing import Any, Dict, Union, List
from functools import lru_cache
import torch.nn.functional as F

try:
    import torch
except ImportError:
    raise ImportError("Please install torch with: pip install torch")
from torch import Tensor

try:
    import transformers
except ImportError:
    raise ImportError("Please install transformers with: pip install transformers")
from transformers import AutoTokenizer, AutoModel

from lightrag.core.api_client import APIClient
from lightrag.core.types import ModelType

from lightrag.core.component import Component


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TransformerEmbedder(Component):
    """
    The is the real client, either sync or async
    """

    def __init__(self):
        super().__init__()

    @lru_cache(None)
    def _init_model(self, model_name: str):
        print(f"loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"done loading tokenizer {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        print(f"done loading model {model_name}")

    def _infer_gte_base_embedding(
        self, input=Union[str, List[str]], tolist: bool = True
    ):
        if isinstance(input, str):
            input = [input]
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # convert to list
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings

    def __call__(self, **kwargs):
        if "model" not in kwargs:
            raise ValueError("model is required")

        if "mock" in kwargs and kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings
        # load files and models, cache it for the next inference
        model_name = kwargs["model"]
        self._init_model(model_name)
        # inference the model
        if model_name == "thenlper/gte-base":
            return self._infer_gte_base_embedding(kwargs["input"])
        else:
            raise ValueError(f"model {model_name} is not supported")


class TransformersClient(APIClient):
    def __init__(self) -> None:
        super().__init__()
        self.provider = "Transformers"
        self.sync_client = self._init_sync_client()
        self.async_client = None
        support_model_list = {
            "thenlper/gte-base": {
                "type": ModelType.EMBEDDER,
            }
        }

    def _init_sync_client(self):
        return TransformerEmbedder()

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        return self.sync_client(**api_kwargs)

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            final_model_kwargs["input"] = input
            return final_model_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")


if __name__ == "__main__":

    def test_transformer_embedder():
        transformer_embedder_model = "thenlper/gte-base"
        transformer_embedder_model_component = TransformerEmbedder()
        print("Testing transformer embedder")
        output = transformer_embedder_model_component(
            model=transformer_embedder_model, input="Hello world"
        )
        print(output)

    def test_transformer_client():
        transformer_client = TransformersClient()
        print("Testing transformer client")
        # run the model
        kwargs = {
            "model": "thenlper/gte-base",
            "mock": False,
        }
        api_kwargs = transformer_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            combined_model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        )
        print(api_kwargs)
        output = transformer_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        print(transformer_client)
        print(output)

    # test transfomer embedding
    from transformers import file_utils

    # print(file_utils.default_cache_path)
    # from transformers import TRANSFORMERS_CACHE

    # print(TRANSFORMERS_CACHE)

    # import shutil

    # shutil.rmtree(TRANSFORMERS_CACHE)
    test_transformer_client()
