from adalflow.components.model_client import BedrockAPIClient
from adalflow.core.types import ModelType
from adalflow.utils import setup_env


def list_models():
    # For list of models
    model_client = BedrockAPIClient()
    model_client.list_models(byProvider="meta")


def bedrock_chat_conversation():
    # Initialize the Bedrock client for API interactions
    awsbedrock_client = BedrockAPIClient()
    query = "What is the capital of France?"

    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {query}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    # Set the model type to Large Language Model (LLM)
    model_type = ModelType.LLM

    # Configure model parameters:
    # - model: Specifies Llama-3-2 1B as the model to use
    # - temperature: Controls randomness (0.5 = balanced between deterministic and creative)
    # - max_tokens: Limits the response length to 100 tokens

    # Using Model ARN since its has inference_profile in us-east-1 region
    # https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers?model=meta.llama3-2-1b-instruct-v1:0
    model_id = "arn:aws:bedrock:us-east-1:306093656765:inference-profile/us.meta.llama3-2-1b-instruct-v1:0"
    model_kwargs = {"model": model_id, "temperature": 0.5, "max_tokens": 100}

    # Convert the inputs into the format required by BedRock's API
    api_kwargs = awsbedrock_client.convert_inputs_to_api_kwargs(
        input=formatted_prompt, model_kwargs=model_kwargs, model_type=model_type
    )
    print(f"api_kwargs: {api_kwargs}")

    response = awsbedrock_client.call(api_kwargs=api_kwargs, model_type=model_type)

    # Extract the text from the chat completion response
    response_text = awsbedrock_client.parse_chat_completion(response)
    print(f"response_text: {response_text}")


if __name__ == "__main__":
    setup_env()
    list_models()
    bedrock_chat_conversation()
