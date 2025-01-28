from adalflow.components.model_client import DeepSeekClient
from adalflow.core.types import ModelType

from adalflow.utils import setup_env

# Initialize the DeepSeekClient
setup_env() 
deepseek_client = DeepSeekClient()

# Example query for the DeepSeek reasoning model
query = "What is the capital of France?"

# === Example 1: Using DeepSeek LLM Model ===
print("=== Example 1: Using DeepSeek LLM Model ===")

# Set the model type for LLM
model_type = ModelType.LLM

# Define the system prompt and user query
system_prompt = "You are a helpful assistant."
prompt = f"<START_OF_SYSTEM_PROMPT>\n{system_prompt}\n<END_OF_SYSTEM_PROMPT>\n<START_OF_USER_PROMPT>\n{query}\n<END_OF_USER_PROMPT>"

# Define model-specific parameters
model_kwargs = {"model": "deepseek-reasoner", "temperature": 0.7, "max_tokens": 100, "stream": False}

# Convert the inputs into API-compatible arguments
api_kwargs = deepseek_client.convert_inputs_to_api_kwargs(
    input=prompt, model_kwargs=model_kwargs, model_type=model_type
)
print(f"api_kwargs: {api_kwargs}")

# Call the DeepSeek reasoning model
response = deepseek_client.call(api_kwargs=api_kwargs, model_type=model_type)

# Parse the response
response_text = deepseek_client.parse_chat_completion(response)
print(f"response_text: {response_text}")

# === Example 2: Using DeepSeek Embedder Model ===
print("\n=== Example 2: Using DeepSeek Embedder Model ===")

# Set the model type for embedding
model_type = ModelType.EMBEDDER

# Define the input for embedding
input = [query] * 2  # Batch embedding

# Define model-specific parameters for embedding
model_kwargs = {
    "model": "deepseek-embedder",  # Replace with the actual embedding model name from DeepSeek
    "dimensions": 512,  # Example dimension size
    "encoding_format": "float",
}

# Convert the inputs into API-compatible arguments
api_kwargs = deepseek_client.convert_inputs_to_api_kwargs(
    input=input, model_kwargs=model_kwargs, model_type=model_type
)
print(f"api_kwargs: {api_kwargs}")

# Call the DeepSeek embedding model
response = deepseek_client.call(api_kwargs=api_kwargs, model_type=model_type)

# Parse the embedding response
response_embedder_output = deepseek_client.parse_embedding_response(response)
print(f"response_embedder_output: {response_embedder_output}")
