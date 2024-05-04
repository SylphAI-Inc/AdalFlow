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

from core.data_classes import Chunk, EmbedderOutput, RetrieverOutput
from core.tokenizer import Tokenizer

dotenv.load_dotenv(dotenv_path=".env", override=True)


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
