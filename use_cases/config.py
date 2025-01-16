from adalflow.components.model_client.groq_client import GroqAPIClient
from adalflow.components.model_client.openai_client import OpenAIClient

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
        "model": "gpt-3.5-turbo",
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

gpt_4o_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4o-mini",
        "temperature": 1,
        "top_p": 0.99,
        "max_tokens": 1000,
        # "frequency_penalty": 1,  # high for nto repeating prompt
    },
}


dataset_path = "cache_datasets"
