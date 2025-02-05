from adalflow.components.model_client.groq_client import GroqAPIClient
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.together_client import TogetherClient

from adalflow.utils import setup_env

setup_env()


llama3_model = {
    "model_client": GroqAPIClient(),
    "model_kwargs": {
        "model": "llama-3.1-8b-instant",
    },
}
gpt_3_model = {
    "model_client": OpenAIClient(input_type="text"),
    "model_kwargs": {
        "model": "gpt-3.5-turbo-0125",
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 0.99,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    },
}

gpt_o1_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "o1",
        "temperature": 1,
        # "top_p": 0.99,
    },
}

deepseek_r1_model = {
    "model_client": TogetherClient(),
    "model_kwargs": {
        "model": "deepseek-ai/DeepSeek-R1",
        "temperature": 1,
        "top_p": 0.99,
    },
}


gpt_3_1106_model = {
    "model_client": OpenAIClient(input_type="text"),
    "model_kwargs": {
        "model": "gpt-3.5-turbo-1106",
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 0.99,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    },
}

# https://openai.com/api/pricing/
# use this for evaluation
gpt_4o_mini_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4o-mini",
        "temperature": 1,
        "top_p": 0.99,
        "max_tokens": 1000,
        # "frequency_penalty": 1,  # high for nto repeating prompt
    },
}

gpt_4_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4-turbo",
        "temperature": 1,
        "top_p": 0.99,
        "max_tokens": 1000,
        # "frequency_penalty": 1,  # high for nto repeating prompt
    },
}

gpt_4o_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4o",  # gpt-4o-realtime-preview-2024-12-17
        "temperature": 1,
        "top_p": 0.99,
        # "max_tokens": 1000,
        # "frequency_penalty": 0.8,  # high for nto repeating prompt
    },
}


dataset_path = "cache_datasets"
