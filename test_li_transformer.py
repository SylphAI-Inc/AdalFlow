from transformers import LlamaModel, LlamaConfig

# Initializing a LLaMA llama-7b style configuration
configuration = LlamaConfig()

# Initializing a model from the llama-7b style configuration
model = LlamaModel(configuration)

# Accessing the model configuration
configuration = model.config


import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
pipeline("Hey how are you doing today?")
