from typing import List, Optional, Union, Dict, Any
from uuid import UUID
import uuid
import os
import dotenv
import numpy as np
from abc import ABC, abstractmethod
import jinja2
from jinja2 import Template

import backoff
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from copy import deepcopy

import tiktoken


dotenv.load_dotenv(dotenv_path=".env", override=True)


# TODO: (1) improve logging
##############################################
# Key data structures for RAG
# TODO: visualize the data structures
##############################################
class Document:
    meta_data: dict  # can save data for filtering at retrieval time too
    text: str
    id: Optional[Union[str, UUID]] = (
        None  # if the file name is unique, its better to use it as id instead of UUID
    )
    estimated_num_tokens: Optional[int] = (
        None  # useful for cost and chunking estimation
    )

    def __init__(
        self,
        meta_data: dict,
        text: str,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
    ):
        self.meta_data = meta_data
        self.text = text
        self.id = id
        self.estimated_num_tokens = estimated_num_tokens

    @staticmethod
    def from_dict(doc: Dict):
        assert "meta_data" in doc, "meta_data is required"
        assert "text" in doc, "text is required"
        if "estimated_num_tokens" not in doc:
            tokenizer = Tokenizer()
            doc["estimated_num_tokens"] = tokenizer.count_tokens(doc["text"])
        if "id" not in doc:
            doc["id"] = uuid.uuid4()

        return Document(**doc)

    def __repr__(self) -> str:
        return f"Document(id={self.id}, meta_data={self.meta_data}, text={self.text[0:50]}, estimated_num_tokens={self.estimated_num_tokens})"

    def __str__(self):
        return self.__repr__()


class Chunk:
    vector: List[float]
    text: str
    order: Optional[int] = (
        None  # order of the chunk in the document. Llama index uses RelatedNodeInfo which is an overkill
    )

    doc_id: Optional[Union[str, UUID]] = (
        None  # id of the Document where the chunk is from
    )
    id: Optional[Union[str, UUID]] = None
    estimated_num_tokens: Optional[int] = None
    score: Optional[float] = None  # used in retrieved output
    meta_data: Optional[Dict] = (
        None  # only when the above fields are not enough or be used for metadata filtering
    )

    def __init__(
        self,
        vector: List[float],
        text: str,
        order: Optional[int] = None,
        doc_id: Optional[Union[str, UUID]] = None,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
        meta_data: Optional[Dict] = None,
    ):
        self.vector = vector if vector else []
        self.text = text
        self.order = order
        self.doc_id = doc_id
        self.id = id if id else uuid.uuid4()
        self.meta_data = meta_data

        self.estimated_num_tokens = estimated_num_tokens if estimated_num_tokens else 0
        # estimate the number of tokens
        if not self.estimated_num_tokens:
            tokenizer = Tokenizer()
            self.estimated_num_tokens = tokenizer.count_tokens(self.text)

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, doc_id={self.doc_id}, order={self.order}, text={self.text}, vector={self.vector[0:5]}, estimated_num_tokens={self.estimated_num_tokens}, score={self.score})"

    def __str__(self):
        return self.__repr__()


##############################################
# Helper modules for RAG
##############################################
class Tokenizer:
    def __init__(self, name: str = "cl100k_base"):
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


##############################################
# Basic prompts example
# load it with jinja2
# {# #} is for comments
# {{ }} is for variables
# Write it as if you are writing a document
# 1. we like to put all content of the prompt into a single jinja2 template, all in the system role
# 2. Even though the whole prompt is a system role, we differentiate our own system and user prompt in the template as User and You
# 3. system prompts section include: role, task desc, requirements, few-shot examples [Requirements or few-shots can be removed if you fine-tune the model]
# 4. user prompts section include: context, query. Answer is left blank.
##############################################
DEFAULT_QA_PROMPT = r"""
    <START_OF_SYSTEM_PROMPT>
    You are a helpful assistant.

    Your task is to answer the query that may or may not come with context information.
    When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
    {# you can add requirements and few-shot examples here #}
    <END_OF_SYSTEM_PROMPT>
    <START_OF_USER_PROMPT>
    Context information:
    ---------------------
    {{context_str}}
    ---------------------
    User: {{query_str}}
    You:
    """


##############################################
# Key functional modules for RAG
##############################################


class Retriever(ABC):
    name = "Retriever"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    # def reset(self):
    #     pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


##############################################
# Generator Runner
##############################################
