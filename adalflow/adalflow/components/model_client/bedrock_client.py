"""AWS Bedrock ModelClient integration."""

import json
import os
from typing import Dict, Optional, Any, Callable, Generator as GeneratorType
import backoff
import logging

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput

from adalflow.utils.lazy_import import safe_import, OptionalPackages

boto3 = safe_import(OptionalPackages.BOTO3.value[0], OptionalPackages.BOTO3.value[1])

from botocore.config import Config

log = logging.getLogger(__name__)

bedrock_runtime_exceptions = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION_NAME", "us-east-1"),
).exceptions


def get_first_message_content(completion: Dict) -> str:
    r"""When we only need the content of the first message.
    It is the default parser for chat completion."""
    return completion["output"]["message"]["content"][0]["text"]


__all__ = [
    "BedrockAPIClient",
    "get_first_message_content",
    "bedrock_runtime_exceptions",
]


class BedrockAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Bedrock API client.

    Note:

    This api is in experimental and is not fully tested and validated yet.

    Support:
    1. AWS Titan
    2. Claude
    3. Cohere
    4. LLama
    5. Mistral
    6. Jamba

    Setup:
    1. Install boto3: `pip install boto3`
    2. Ensure you have the AWS credentials set up. There are four variables you can optionally set:
        - AWS_PROFILE_NAME: The name of the AWS profile to use.
        - AWS_REGION_NAME: The name of the AWS region to use.
        - AWS_ACCESS_KEY_ID: The AWS access key ID.
        - AWS_SECRET_ACCESS_KEY: The AWS secret access key.




    Example:

    .. code-block:: python

        from adalflow.components.model_client import BedrockAPIClient

        template = "<SYS>
        You are a helpful assistant.
        </SYS>
        User: {{input_str}}
        You:
        "

        # use AWS_PROFILE_NAME and AWS_REGION_NAME from the environment variables in this case
        # ensure you request the modelId from the AWS team
        self.generator = Generator(
            model_client=BedrockAPIClient(),
            model_kwargs={
                "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                "inferenceConfig": {
                    "temperature": 0.8
                }
            }, template=template
        )

    Relevant API docs:
    1. https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html
    2. https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-ex-python.html
    3. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestParameters
    4. To setup the AWS credentials, follow the instructions here:
    https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html
    5. Additionally, this medium article is a good reference:
    https://medium.com/@harangpeter/setting-up-aws-bedrock-for-api-based-text-inference-dc25ab2b216b
    """

    def __init__(
        self,
        aws_profile_name=None,
        aws_region_name=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        aws_connection_timeout=None,
        aws_read_timeout=None,
        chat_completion_parser: Callable = None,
    ):
        super().__init__()
        self._aws_profile_name = aws_profile_name
        self._aws_region_name = aws_region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._aws_connection_timeout = aws_connection_timeout
        self._aws_read_timeout = aws_read_timeout

        self._client = None
        self.session = None
        self.sync_client = self.init_sync_client()
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )
        self.inference_parameters = [
            "maxTokens",
            "temperature",
            "topP",
            "stopSequences",
        ]

    def init_sync_client(self):
        """
        There is no need to pass both profile and secret key and access key. Path one of them.
        if the compute power assume a role that have access to bedrock, no need to pass anything.
        """
        aws_profile_name = self._aws_profile_name or os.getenv("AWS_PROFILE_NAME")
        aws_region_name = self._aws_region_name or os.getenv("AWS_REGION_NAME")
        aws_access_key_id = self._aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = self._aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        aws_session_token = self._aws_session_token or os.getenv("AWS_SESSION_TOKEN")

        config = None
        if self._aws_connection_timeout or self._aws_read_timeout:
            config = Config(
                connect_timeout=self._aws_connection_timeout,  # Connection timeout in seconds
                read_timeout=self._aws_read_timeout,  # Read timeout in seconds
            )

        session = boto3.Session(
            profile_name=aws_profile_name,
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        bedrock_runtime = session.client(service_name="bedrock-runtime", config=config)

        self._client = session.client(service_name="bedrock")
        return bedrock_runtime

    def init_async_client(self):
        raise NotImplementedError("Async call not implemented yet.")

    def handle_stream_response(self, stream: dict) -> GeneratorType:
        r"""Handle the stream response from bedrock. Yield the chunks.

        Args:
            stream (dict): The stream response generator from bedrock.

        Returns:
            GeneratorType: A generator that yields the chunks from bedrock stream.
        """
        try:
            stream: GeneratorType = stream["stream"]
            for chunk in stream:
                log.debug(f"Raw chunk: {chunk}")
                yield chunk
        except Exception as e:
            log.debug(f"Error in handle_stream_response: {e}")  # Debug print
            raise

    def parse_chat_completion(self, completion: dict) -> "GeneratorOutput":
        r"""Parse the completion, and assign it into the raw_response attribute.

        If the completion is a stream, it will be handled by the handle_stream_response
        method that returns a Generator. Otherwise, the completion will be parsed using
        the get_first_message_content method.

        Args:
            completion (dict): The completion response from bedrock API call.

        Returns:
            GeneratorOutput: A generator output object with the parsed completion. May
                return a generator if the completion is a stream.
        """
        try:
            usage = None
            data = self.chat_completion_parser(completion)
            if not isinstance(data, GeneratorType):
                # Streaming completion usage tracking is not implemented.
                usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(
                data=None, error=str(e), raw_response=json.dumps(completion)
            )

    def track_completion_usage(self, completion: Dict) -> CompletionUsage:
        r"""Track the completion usage."""
        usage = completion["usage"]
        return CompletionUsage(
            completion_tokens=usage["outputTokens"],
            prompt_tokens=usage["inputTokens"],
            total_tokens=usage["totalTokens"],
        )

    def list_models(self, **kwargs):
        # Initialize Bedrock client (not runtime)

        try:
            response = self._client.list_foundation_models(**kwargs)
            models = response.get("modelSummaries", [])
            for model in models:
                print(f"Model ID: {model['modelId']}")
                print(f"  Name: {model['modelName']}")
                print(f"  Model ARN: {model['modelArn']}")
                print(f"  Provider: {model['providerName']}")
                print(f"  Input: {model['inputModalities']}")
                print(f"  Output: {model['outputModalities']}")
                print(f"  InferenceTypesSupported: {model['inferenceTypesSupported']}")
                print("")

        except Exception as e:
            print(f"Error listing models: {e}")

    def _validate_and_process_config_keys(self, api_kwargs: Dict):
        """
        Validate and process the model ID in API kwargs.

        :param api_kwargs: Dictionary of API keyword arguments
        :raises KeyError: If 'model' key is missing
        """
        if "model" in api_kwargs:
            api_kwargs["modelId"] = api_kwargs.pop("model")
        else:
            raise KeyError("The required key 'model' is missing in model_kwargs.")

        # In .converse() `maxTokens`` is the key for maximum tokens limit
        if "max_tokens" in api_kwargs:
            api_kwargs["maxTokens"] = api_kwargs.pop("max_tokens")

        return api_kwargs

    def _separate_parameters(self, api_kwargs: Dict) -> tuple:
        """
        Separate inference configuration and additional model request fields.

        :param api_kwargs: Dictionary of API keyword arguments
        :return: Tuple of (inference_config, additional_model_request_fields)
        """
        inference_config = {}
        additional_model_request_fields = {}
        keys_to_remove = set()
        excluded_keys = {"modelId"}

        # Categorize parameters
        for key, value in list(api_kwargs.items()):
            if key in self.inference_parameters:
                inference_config[key] = value
                keys_to_remove.add(key)
            elif key not in excluded_keys:
                additional_model_request_fields[key] = value
                keys_to_remove.add(key)

        # Remove categorized keys from api_kwargs
        for key in keys_to_remove:
            api_kwargs.pop(key, None)

        return api_kwargs, inference_config, additional_model_request_fields

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ):
        """
        check the converse api doc here:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        """
        api_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            # Validate and process model ID
            api_kwargs = self._validate_and_process_config_keys(api_kwargs)

            # Separate inference config and additional model request fields
            api_kwargs, inference_config, additional_model_request_fields = self._separate_parameters(api_kwargs)

            api_kwargs["messages"] = [
                {"role": "user", "content": [{"text": input}]},
            ]
            api_kwargs["inferenceConfig"] = inference_config
            api_kwargs["additionalModelRequestFields"] = additional_model_request_fields
        else:
            raise ValueError(f"Model type {model_type} not supported")
        return api_kwargs

    @backoff.on_exception(
        backoff.expo,
        (
            bedrock_runtime_exceptions.ThrottlingException,
            bedrock_runtime_exceptions.ModelTimeoutException,
            bedrock_runtime_exceptions.InternalServerException,
            bedrock_runtime_exceptions.ModelErrorException,
            bedrock_runtime_exceptions.ValidationException,
        ),
        max_time=2,
    )
    def call(
        self,
        api_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        """
        kwargs is the combined input and model_kwargs
        """
        if model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("Streaming call")
                api_kwargs.pop(
                    "stream", None
                )  # stream is not a valid parameter for bedrock
                self.chat_completion_parser = self.handle_stream_response
                return self.sync_client.converse_stream(**api_kwargs)
            else:
                api_kwargs.pop("stream", None)
                return self.sync_client.converse(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    async def acall(self):
        raise NotImplementedError("Async call not implemented yet.")
