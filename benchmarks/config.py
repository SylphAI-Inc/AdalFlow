from adalflow.components.model_client import GroqAPIClient
from adalflow.components.model_client import OpenAIClient

dspy_save_path = "benchmarks/BHH_object_count/models/dspy"
text_grad_save_path = "benchmarks/BHH_object_count/models/text_grad"
adal_save_path = "benchmarks/BHH_object_count/models/adal"

dspy_hotpot_qa_save_path = "benchmarks/hotpot_qa/models/dspy"
text_grad_hotpot_qa_save_path = "benchmarks/hotpot_qa/models/text_grad"
adal_hotpot_qa_save_path = "benchmarks/hotpot_qa/models/adal"


llama3_model = {
    "model_client": GroqAPIClient,
    "model_kwargs": {
        "model": "llama-3.1-8b-instant",
    },
}
gpt_3_model = {
    "model_client": OpenAIClient,
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

gpt_4o_model = {
    "model_client": OpenAIClient,
    "model_kwargs": {
        "model": "gpt-4o",
        "temperature": 0.9,
        "top_p": 0.99,
    },
}


def load_model(**kwargs):
    if "model_client" in kwargs:
        kwargs["model_client"] = kwargs["model_client"]()
    return kwargs
