from transformers import LlamaModel, LlamaConfig

# Initializing a LLaMA llama-7b style configuration
# configuration = LlamaConfig()

# # Initializing a model from the llama-7b style configuration
# model = LlamaModel(configuration)

# # Accessing the model configuration
# configuration = model.config


# https://huggingface.co/LiteLLMs/Meta-Llama-3-8B-GGUF
import transformers
import torch
import dotenv
import os

dotenv.load_dotenv(dotenv_path=".env", override=True)
# get access here https://huggingface.co/meta-llama/Meta-Llama-3-8B
# get access to mistral https://huggingface.co/mistralai/Mistral-7B-v0.1
# model_id = "meta-llama/Meta-Llama-3-8B"
from transformers import BitsAndBytesConfig

# quantize to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     token=os.environ.get("HF_TOKEN"),
# )

from transformers import pipeline

model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer_name = "HuggingFaceH4/zephyr-7b-beta"
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer_name,
    # model_kwargs={"quantization_config": quantization_config},
)
# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
print(pipe)
output = pipe("Hey how are you doing today?")
print(output)
