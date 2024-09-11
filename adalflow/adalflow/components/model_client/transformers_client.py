"""Huggingface transformers ModelClient integration."""

from typing import Any, Dict, Union, List, Optional, Sequence
import logging
from functools import lru_cache
import re
import warnings

from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput, ModelType, Embedding, EmbedderOutput
from adalflow.core.functional import get_top_k_indices_scores

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

import torch.nn.functional as F
from torch import Tensor
import torch

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
from os import getenv as get_env_variable

transformers = safe_import(
    OptionalPackages.TRANSFORMERS.value[0],
    OptionalPackages.TRANSFORMERS.value[1]
)
torch = safe_import(OptionalPackages.TORCH.value[0], OptionalPackages.TORCH.value[1])


log = logging.getLogger(__name__)


def average_pool(last_hidden_states: Tensor, attention_mask: list) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def mean_pooling(model_output: dict, attention_mask) -> Tensor:
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_device():
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

    return device


def clean_device_cache():
    import torch

    if torch.backends.mps.is_built():
        torch.mps.empty_cache()

        torch.mps.set_per_process_memory_fraction(1.0)


class TransformerEmbeddingModelClient(ModelClient):
    __doc__ = r"""LightRAG API client for embedding models using HuggingFace's transformers library.

    Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.

    Some modeles are gated, you will need to their page to get the access token.
    Find how to apply tokens here: https://huggingface.co/docs/hub/security-tokens
    Once you have a token and have access, put the token in the environment variable HF_TOKEN.
    """
    #
    #   Model initialisation
    #
    def __init__(
            self,
            model_name: Optional[str] = None,
            tokenizer_kwargs: Optional[dict] = None,
            auto_model_kwargs: Optional[dict] = None,
            auto_tokenizer_kwargs: Optional[dict] = None,
            auto_model: Optional[type] = AutoModel,
            auto_tokenizer: Optional[type] = AutoTokenizer,
            local_files_only: Optional[bool] = False,
            custom_model: Optional[PreTrainedModel] = None,
            custom_tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
            ):

        super().__init__()
        self.model_name = model_name
        self.tokenizer_kwargs = tokenizer_kwargs or dict()
        self.auto_model_kwargs = auto_model_kwargs or dict()
        self.auto_tokenizer_kwargs = auto_tokenizer_kwargs or dict()
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.auto_model=auto_model
        self.auto_tokenizer=auto_tokenizer
        self.local_files_only = local_files_only
        self.custom_model=custom_model
        self.custom_tokenizer=custom_tokenizer

        # Check if there is conflicting arguments
        self.use_auto_model = auto_model is not None
        self.use_auto_tokenizer = auto_tokenizer is not None
        self.use_cusom_model = custom_model is not None
        self.use_cusom_tokenizer = custom_tokenizer is not None
        self.model_name_exit = model_name is not None

        ## arguments related to model
        if self.use_auto_model and self.use_cusom_model:
            raise ValueError("Cannot specify 'auto_model' and 'custom_model'.")
        elif (not self.use_auto_model) and (not self.use_cusom_model):
            raise ValueError("Need to specify either 'auto_model' or 'custom_model'.")
        elif self.use_auto_model and (not self.model_name_exit):
            raise ValueError("When 'auto_model' is specified 'model_name' must be specified too.")
        
        ## arguments related to tokenizer
        if self.use_auto_tokenizer and self.use_cusom_tokenizer:
            raise Exception("Cannot specify 'auto_tokenizer' and 'custom_tokenizer'.")
        elif (not self.use_auto_tokenizer) and (not self.use_cusom_tokenizer):
            raise Exception("Need to specify either'auto_tokenizer' and 'custom_tokenizer'.")
        elif self.use_auto_tokenizer and (not self.model_name_exit):
            raise ValueError("When 'auto_tokenizer' is specified 'model_name' must be specified too.")

        self.init_sync_client()


    def init_sync_client(self):
        self.init_model(
            model_name=self.model_name,
            auto_model=self.auto_model,
            auto_tokenizer=self.auto_tokenizer,
            custom_model=self.custom_model,
            custom_tokenizer=self.custom_tokenizer
            )


    @lru_cache(None)
    def init_model(
        self,
        model_name: Optional[str] = None,
        auto_model: Optional[type] = AutoModel,
        auto_tokenizer: Optional[type] = AutoTokenizer,
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        ):

        try:
            if self.use_auto_model:
                self.model = auto_model.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                    **self.auto_model_kwargs
                    )
            else:
                self.model = custom_model

            if self.use_auto_tokenizer:
                self.tokenizer = auto_tokenizer.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                    **self.auto_tokenizer_kwargs
                    )
            else:
                self.tokenizer = custom_tokenizer

            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #
    def infer_embedding(
        self,
        input=Union[str, List[str], List[List[str]]],
        tolist: bool = True,
    ) -> Union[List, Tensor]:
        model = self.model

        self.handle_input(input)
        batch_dict = self.tokenize_inputs(input, kwargs=self.tokenizer_kwargs)
        outputs = self.compute_model_outputs(batch_dict, model)
        embeddings = self.compute_embeddings(outputs, batch_dict)

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings


    def handle_input(self, input: Union[str, List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        if isinstance(input, str):
            input = [input]
        return input
     

    def tokenize_inputs(self, input: Union[str, List[str], List[List[str]]], kwargs: Optional[dict] = None) -> dict:
        kwargs = kwargs or dict()
        batch_dict = self.tokenizer(input, **kwargs)
        return batch_dict


    def compute_model_outputs(self, batch_dict: dict, model: PreTrainedModel) -> dict:
        with torch.no_grad():
            outputs = model(**batch_dict)
        return outputs


    def compute_embeddings(self, outputs: dict, batch_dict: dict):
        embeddings = mean_pooling(
            outputs, batch_dict["attention_mask"]
        )
        return embeddings

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = None, model_type: Optional[ModelType]= ModelType.UNDEFINED) -> Union[List, Tensor]:

        api_kwargs = api_kwargs or dict()
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")
        # I don't think it is useful anymore
        # if (
        #     model_type == ModelType.EMBEDDER
        #     # and "model" in api_kwargs
        # ):
        if "mock" in api_kwargs and api_kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings

        # inference the model
        return self.infer_embedding(api_kwargs["input"])


    def parse_embedding_response(self, response: Union[List, Tensor]) -> EmbedderOutput:
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response


    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: Optional[ModelType]= ModelType.UNDEFINED
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        # if model_type == ModelType.EMBEDDER:
        final_model_kwargs["input"] = input
        return final_model_kwargs


class TransformerLLMModelClient(ModelClient):
    __doc__ = r"""LightRAG API client for text generation models using HuggingFace's transformers library.

    Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.

    Some modeles are gated, you will need to their page to get the access token.
    Find how to apply tokens here: https://huggingface.co/docs/hub/security-tokens
    Once you have a token and have access, put the token in the environment variable HF_TOKEN.
    """
    #
    #   Model initialisation
    #
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_decode_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        auto_model_kwargs: Optional[dict] = None,
        auto_tokenizer_kwargs: Optional[dict] = None,
        init_from: Optional[str] = "autoclass",
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        use_token: bool = False,
        torch_dtype: Optional[Any] = torch.bfloat16,
        local_files_only: Optional[bool] = False
    ):
        super().__init__()

        self.model_name = model_name  # current model to use
        self.tokenizer_decode_kwargs = tokenizer_decode_kwargs or dict()
        self.tokenizer_kwargs = tokenizer_kwargs or dict()
        self.auto_model_kwargs = auto_model_kwargs or dict()
        self.auto_tokenizer_kwargs = auto_tokenizer_kwargs or dict()
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.use_token = use_token
        self.torch_dtype = torch_dtype
        self.init_from = init_from
        self.apply_chat_template = apply_chat_template
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or dict(tokenize=False, add_generation_prompt=True)
        self.local_files_only = local_files_only
        self.model = None
        if model_name is not None:
            self.init_model(model_name=model_name)


    def _check_token(self, token: str):
        if get_env_variable(token) is None:
            warnings.warn(
                f"{token} is not set. You may not be able to access the model."
            )


    def _get_token_if_relevant(self) -> Union[str, bool]:
        if self.use_token:
            self._check_token("HF_TOKEN")
            token = get_env_variable("HF_TOKEN")
        else:
            token = False      
        return token


    def _init_from_pipeline(self):

        clean_device_cache()
        token = self._get_token_if_relevant() # return a token string or False
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device=get_device(),
            token=token
        )


    def _init_from_automodelcasual_lm(self):

        token = self._get_token_if_relevant() # return a token str or False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token,
            local_files_only=self.local_files_only,
            **self.auto_tokenizer_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            token=token,
            local_files_only=self.local_files_only,
            **self.auto_model_kwargs
        )
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )  # ensure consistency in the model config


    @lru_cache(None)
    def init_model(self, model_name: str):

        log.debug(f"Loading model {model_name}") 
        try:
            if self.init_from == "autoclass":
                self._init_from_automodelcasual_lm()
            elif self.init_from == "pipeline":
                self._init_from_pipeline()
            else:
                raise ValueError("argument 'init_from' must be one of 'autoclass' or 'pipeline'.")
        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #
    def _infer_from_pipeline(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        **kwargs,
    ):

        if not self.model:
            self.init_model(model_name=model)

        log.info(
            f"Start to infer model {model}, messages: {messages}, kwargs: {kwargs}"
        )
        #  TO DO: add default values in doc
        final_kwargs = {
            "max_new_tokens": max_tokens or 256,
            "do_sample": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.95),
        }
        if apply_chat_template:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self._get_token_if_relevant(),
                local_files_only=self.local_files_only,
            )
            # Set pad token if it's not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
                self.model.config.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # ensure consistency in the model config

            model_input = self._handle_input(
                messages,
                apply_chat_template=True,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                )
        else:
            model_input = self._handle_input(messages)

        outputs = self.model(
            model_input,
            **final_kwargs,
        )
        log.info(f"Outputs: {outputs}")
        return outputs


    def _infer_from_automodelcasual_lm(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        max_length: Optional[int] = 8192,  # model-agnostic
        apply_chat_template: bool = False,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if not self.model:
            self.init_model(model_name=model)

        if apply_chat_template:
            model_input = self._handle_input(
                messages,
                apply_chat_template=True,
                chat_template_kwargs=chat_template_kwargs,
                chat_template=chat_template
                )
        else:
           model_input = self._handle_input(messages) 
        input_ids = self.tokenizer(model_input, **self.tokenizer_kwargs).to(
            get_device()
        )
        outputs_tokens = self.model.generate(**input_ids, max_length=max_length, max_new_tokens=max_tokens, **kwargs)
        outputs = []
        for output in outputs_tokens:
            outputs.append(self.tokenizer.decode(output))
        return outputs


    def _handle_input(
            self,
            messages: Sequence[Dict[str, str]],
            apply_chat_template: bool = False,
            chat_template_kwargs: dict = None,
            chat_template: Optional[str] = None,
            ) -> str:

        if apply_chat_template:
            if chat_template is not None:
                self.tokenizer.chat_template = chat_template
            prompt = self.tokenizer.apply_chat_template(
                messages, **chat_template_kwargs
            )
            if ("tokenize" in chat_template_kwargs) and (chat_template_kwargs["tokenize"] == True):
                prompt = self.tokenizer.decode(prompt, **self.tokenizer_decode_kwargs)
                return prompt
            else:
                return prompt
        else:
            text = messages[-1]["content"]
            return text


    def infer_llm(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        max_tokens: Optional[int] = None,
        **kwargs,
    ):

        if self.init_from == "pipeline":
            return self._infer_from_pipeline(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                apply_chat_template=self.apply_chat_template,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                **kwargs
            )
        else:
            return self._infer_from_automodelcasual_lm(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                apply_chat_template=self.apply_chat_template,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                **kwargs
            )

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = None, model_type: Optional[ModelType]= ModelType.UNDEFINED):
        api_kwargs = api_kwargs or dict()
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")

        model_name = api_kwargs["model"]
        if (model_name != self.model_name) and (self.model_name is not None):
            # need to update the model_name
            log.warning(f"The model passed in 'model_kwargs' is different that the one that has been previously initialised: Updating model from {self.model_name} to {model_name}.")
            self.model_name = model_name
            self.init_model(model_name=model_name)
        elif (model_name != self.model_name) and (self.model_name is None):
            # need to initialize the model for the first time
            self.model_name = model_name
            self.init_model(model_name=model_name)


        output = self.infer_llm(**api_kwargs)
        return output


    def _parse_chat_completion_from_pipeline(self, completion: Any) -> str:

        text = completion[0]["generated_text"]
        pattern = r"(?<=\|assistant\|>).*"
        match = re.search(pattern, text)

        if match:
            text = match.group().strip().lstrip("\\n")
            return text
        else:
            return ""


    def _parse_chat_completion_from_automodelcasual_lm(self, completion: Any) -> GeneratorOutput:
        print(f"completion: {completion}")
        return completion[0]


    def parse_chat_completion(self, completion: Any) -> str:
        try:
            if self.init_from == "pipeline":
                output = self._parse_chat_completion_from_pipeline(completion)
            else:
                output = self._parse_chat_completion_from_automodelcasual_lm(completion)
            return GeneratorOutput(data=output, raw_response=str(completion))
        except Exception as e:
            log.error(f"Error parsing chat completion: {e}")
            return GeneratorOutput(data=None, raw_response=str(completion), error=e)


    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = None,
        model_type: Optional[ModelType]= ModelType.UNDEFINED
    ) -> dict:
        model_kwargs = model_kwargs or dict()
        final_model_kwargs = model_kwargs.copy()
        assert "model" in final_model_kwargs, "model must be specified"
        #messages = [{"role": "system", "content": input}]
        messages = [{"role": "user", "content": input}] # Not sure, but it seems to make more sense
        final_model_kwargs["messages"] = messages
        return final_model_kwargs


class TransformerRerankerModelClient(ModelClient):
    __doc__ = r"""LightRAG API client for reranker (cross-encoder) models using HuggingFace's transformers library.

    Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.

    Some modeles are gated, you will need to their page to get the access token.
    Find how to apply tokens here: https://huggingface.co/docs/hub/security-tokens
    Once you have a token and have access, put the token in the environment variable HF_TOKEN.
    """
    #
    #   Model initialisation
    #
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        auto_model_kwargs: Optional[dict] = None,
        auto_tokenizer_kwargs: Optional[dict] = None,
        auto_model: Optional[type] = AutoModelForSequenceClassification,
        auto_tokenizer: Optional[type] = AutoTokenizer,
        local_files_only: Optional[bool] = False
    ):
        self.auto_model = auto_model
        self.auto_model_kwargs = auto_model_kwargs or dict()
        self.auto_tokenizer_kwargs = auto_tokenizer_kwargs or dict()
        self.auto_tokenizer= auto_tokenizer
        self.model_name = model_name
        self.tokenizer_kwargs = tokenizer_kwargs or dict()
        if "return_tensors" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["return_tensors"]= "pt"
        self.local_files_only = local_files_only
        if model_name is not None:
            self.init_model(model_name=model_name)


    def init_model(self, model_name: str):
        try:
            self.tokenizer = self.auto_tokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            **self.auto_tokenizer_kwargs
            )
            self.model = self.auto_model.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            **self.auto_model_kwargs
            )
            # Check device availability and set the device
            device = get_device()

            # Move model to the selected device
            self.device = device
            self.model.to(device)
            self.model.eval()
            # register the model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    #
    #   Inference code
    #

    def infer_reranker(
        self,
        model: str,
        query: str,
        documents: List[str],
    ) -> List[float]:
        if not self.model:
            self.init_model(model_name=model)
        # convert the query and documents to pair input
        input = [(query, doc) for doc in documents]

        with torch.no_grad():

            inputs = self.tokenizer(
                input,
                **self.tokenizer_kwargs
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            # apply sigmoid to get the scores
            scores = F.sigmoid(scores)

        scores = scores.tolist()
        return scores

    #
    # Preprocessing, postprocessing and call for inference code
    #
    def call(self, api_kwargs: Dict = None):
        api_kwargs = api_kwargs or dict()
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")

        model_name = api_kwargs["model"]
        if (model_name != self.model_name) and (self.model_name is not None):
            # need to update the model_name
            log.warning(f"The model passed in 'model_kwargs' is different that the one that has been previously initialised: Updating model from {self.model_name} to {model_name}.")
            self.model_name = model_name
            self.init_model(model_name=model_name)
        elif (model_name != self.model_name) and (self.model_name is None):
            # need to initialize the model for the first time
            self.model_name = model_name
            self.init_model(model_name=model_name)

        assert "query" in api_kwargs, "query is required"
        assert "documents" in api_kwargs, "documents is required"
        assert "top_k" in api_kwargs, "top_k is required"

        top_k = api_kwargs.pop("top_k")
        scores = self.infer_reranker(**api_kwargs)
        top_k_indices, top_k_scores = get_top_k_indices_scores(
            scores, top_k
        )
        log.warning(f"output: ({top_k_indices}, {top_k_scores})")
        return top_k_indices, top_k_scores


    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = None,
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        model_kwargs = model_kwargs or dict()
        final_model_kwargs = model_kwargs.copy()

        assert "model" in final_model_kwargs, "model must be specified"
        assert "documents" in final_model_kwargs, "documents must be specified"
        assert "top_k" in final_model_kwargs, "top_k must be specified"
        final_model_kwargs["query"] = input
        return final_model_kwargs
