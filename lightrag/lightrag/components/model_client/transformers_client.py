"""Huggingface transformers ModelClient integration."""

from typing import Any, Dict, Union, List, Optional
import logging
from functools import lru_cache

import torch.nn.functional as F
import torch
from torch import Tensor


from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, Embedding, EmbedderOutput
from lightrag.core.functional import get_top_k_indices_scores
from lightrag.utils.lazy_import import safe_import, OptionalPackages


safe_import(
    OptionalPackages.TRANSFORMERS.value[0], OptionalPackages.TRANSFORMERS.value[1]
)
safe_import(OptionalPackages.TORCH.value[0], OptionalPackages.TORCH.value[1])

log = logging.getLogger(__name__)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# TODO: provide a standard api for embedding and chat models used in local model SDKs
class TransformerEmbedder:
    """Local model SDK for transformers.


    There are two ways to run transformers:
    (1) model and then run model inference
    (2) Pipeline and then run pipeline inference

    This file demonstrates how to
    (1) create a torch model inference component:  TransformerEmbedder which equalize to OpenAI(), the SyncAPIClient
    (2) Convert this model inference component to LightRAG API client: TransformersClient

    The is now just an exmplary component that initialize a certain model from transformers and run inference on it.
    It is not tested on all transformer models yet. It might be necessary to write one for each model.

    References:
    - transformers: https://huggingface.co/docs/transformers/en/index
    - thenlper/gte-base model:https://huggingface.co/thenlper/gte-base
    """

    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = "thenlper/gte-base"):
        super().__init__()

        if model_name is not None:
            self.init_model(model_name=model_name)

    @lru_cache(None)
    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            # register the model
            self.models[model_name] = self.model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def infer_gte_base_embedding(
        self,
        input=Union[str, List[str]],
        tolist: bool = True,
    ):
        model = self.models.get("thenlper/gte-base", None)
        if model is None:
            # initialize the model
            self.init_model("thenlper/gte-base")

        if isinstance(input, str):
            input = [input]
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
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
        # inference the model
        if model_name == "thenlper/gte-base":
            return self.infer_gte_base_embedding(kwargs["input"])
        else:
            raise ValueError(f"model {model_name} is not supported")


class TransformerReranker:
    __doc__ = r"""Local model SDK for a reranker model using transformers.

    References:
    - model: https://huggingface.co/BAAI/bge-reranker-base
    - paper: https://arxiv.org/abs/2309.07597

    note:
    If you are using Macbook M1 series chips, you need to ensure ``torch.device("mps")`` is set.
    """
    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = "BAAI/bge-reranker-base"):
        self.model_name = model_name or "BAAI/bge-reranker-base"
        if model_name is not None:
            self.init_model(model_name=model_name)

    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Check device availability and set the device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                log.info("Using CUDA (GPU) for inference.")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                log.info("Using MPS (Apple Silicon) for inference.")
            else:
                device = torch.device("cpu")
                log.info("Using CPU for inference.")

            # Move model to the selected device
            self.device = device
            self.model.to(device)
            self.model.eval()
            # register the model
            self.models[model_name] = self.model  # TODO: better model registration
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def infer_bge_reranker_base(
        self,
        # input=List[Tuple[str, str]],  # list of pairs of the query and the candidate
        query: str,
        documents: List[str],
    ) -> List[float]:
        model = self.models.get(self.model_name, None)
        if model is None:
            # initialize the model
            self.init_model(self.model_name)

        # convert the query and documents to pair input
        input = [(query, doc) for doc in documents]

        with torch.no_grad():

            inputs = self.tokenizer(
                input,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            # apply sigmoid to get the scores
            scores = F.sigmoid(scores)

        scores = scores.tolist()
        return scores

    def __call__(self, **kwargs):
        r"""Ensure "model" and "input" are in the kwargs."""
        if "model" not in kwargs:
            raise ValueError("model is required")

        # if "mock" in kwargs and kwargs["mock"]:
        #     import numpy as np

        #     scores = np.array([np.random.rand(1).tolist()])
        #     return scores
        # load files and models, cache it for the next inference
        model_name = kwargs["model"]
        # inference the model
        if model_name == self.model_name:
            assert "query" in kwargs, "query is required"
            assert "documents" in kwargs, "documents is required"
            scores = self.infer_bge_reranker_base(kwargs["query"], kwargs["documents"])
            return scores
        else:
            raise ValueError(f"model {model_name} is not supported")


class TransformerLLM:
    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = "HuggingFaceH4/zephyr-7b-beta"):
        super().__init__()

        if model_name is not None:
            self.init_model(model_name=model_name)

    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            # register the model
            self.models[model_name] = self.model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Done loading model {model_name}")
            # Set pad token if it's not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
                self.model.config.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # ensure consistency in the model config
        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def parse_chat_completion(self, input_text: str, response: str):
        parsed_response = response.replace(
            input_text, ""
        ).strip()  # Safely handle cases where input_text might not be in response

        return parsed_response if parsed_response else response

    def call(
        self,
        input_text: str,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        max_length: int = 150,
    ):
        if not self.model:
            log.error("Model is not initialized.")
            raise ValueError("Model is not initialized.")

        # Ensure tokenizer has pad token; set it if not
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )  # Sync model config pad token id

        # Process inputs with attention mask and padding
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(
            self.device
        )
        # inputs = self.tokenizer(input_text, return_tensors="pt", padding="longest", truncation=True).to(self.device)

        with torch.no_grad():  # Ensures no gradients are calculated to save memory and computations
            generate_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,  # Control the output length more precisely
            )
        response = self.tokenizer.decode(
            generate_ids[0],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        parsed_response = self.parse_chat_completion(input_text, response)
        return parsed_response

    def __call__(
        self,
        input_text: str,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        max_length: int = 150,
    ):
        return self.call(
            input_text,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            max_length=max_length,
        )

    # def call(self, input_text: str, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):
    #     if not self.model:
    #         log.error("Model is not initialized.")
    #         raise ValueError("Model is not initialized.")

    #     inputs = self.tokenizer(input_text, return_tensors="pt")
    #     generate_ids = self.model.generate(inputs.input_ids, max_length=30)
    #     response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)[0]
    #     return response

    # def __call__(self, input_text: str, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):
    #     return self.call(input_text, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


class TransformersClient(ModelClient):
    __doc__ = r"""LightRAG API client for transformers.

    Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.
    """

    support_models = {
        "thenlper/gte-base": {
            "type": ModelType.EMBEDDER,
        },
        "BAAI/bge-reranker-base": {
            "type": ModelType.RERANKER,
        },
        "HuggingFaceH4/zephyr-7b-beta": {"type": ModelType.LLM},
    }

    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__()
        self._model_name = model_name
        if self._model_name:
            assert (
                self._model_name in self.support_models
            ), f"model {self._model_name} is not supported"
        if self._model_name == "thenlper/gte-base":
            self.sync_client = self.init_sync_client()
        elif self._model_name == "BAAI/bge-reranker-base":
            self.reranker_client = self.init_reranker_client()
        elif self._model_name == "HuggingFaceH4/zephyr-7b-beta":
            self.llm_client = self.init_llm_client()
        self.async_client = None

    def init_sync_client(self):
        return TransformerEmbedder()

    def init_reranker_client(self):
        return TransformerReranker()

    def init_llm_client(self):
        return TransformerLLM()

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert "model" in api_kwargs, "model is required"
        assert (
            api_kwargs["model"] in self.support_models
        ), f"model {api_kwargs['model']} is not supported"
        if (
            model_type == ModelType.EMBEDDER
            and "model" in api_kwargs
            and api_kwargs["model"] == "thenlper/gte-base"
        ):
            if self.sync_client is None:
                self.sync_client = self.init_sync_client()
            return self.sync_client(**api_kwargs)
        elif (  # reranker
            model_type == ModelType.RERANKER
            and "model" in api_kwargs
            and api_kwargs["model"] == "BAAI/bge-reranker-base"
        ):
            if not hasattr(self, "reranker_client") or self.reranker_client is None:
                self.reranker_client = self.init_reranker_client()
            scores = self.reranker_client(**api_kwargs)
            top_k_indices, top_k_scores = get_top_k_indices_scores(
                scores, api_kwargs["top_k"]
            )
            return top_k_indices, top_k_scores
        elif (  # LLM
            model_type == ModelType.LLM
            and "model" in api_kwargs
            and api_kwargs["model"] == "HuggingFaceH4/zephyr-7b-beta"
        ):
            if not hasattr(self, "llm_client") or self.llm_client is None:
                self.llm_client = self.init_llm_client()
            response = self.llm_client(**api_kwargs)
            return response

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            final_model_kwargs["input"] = input
            return final_model_kwargs
        elif model_type == ModelType.RERANKER:
            assert "model" in final_model_kwargs, "model must be specified"
            assert "documents" in final_model_kwargs, "documents must be specified"
            assert "top_k" in final_model_kwargs, "top_k must be specified"
            final_model_kwargs["query"] = input
            return final_model_kwargs
        elif model_type == ModelType.LLM:
            assert "model" in final_model_kwargs, "model must be specified"
            final_model_kwargs["input"] = input
            return final_model_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")
