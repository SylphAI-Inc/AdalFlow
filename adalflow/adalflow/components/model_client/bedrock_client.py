"""AWS Bedrock ModelClient integration."""

import os
from typing import Dict, Optional, Any, Callable
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
        aws_profile_name="default",
        aws_region_name="us-west-2",  # Use a supported default region
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

        self.session = None
        self.sync_client = self.init_sync_client()
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )

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

    def parse_chat_completion(self, completion):
        log.debug(f"completion: {completion}")
        try:
            data = completion["output"]["message"]["content"][0]["text"]
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(
                data=None, error=str(e), raw_response=str(completion)
            )

    def track_completion_usage(self, completion: Dict) -> CompletionUsage:
        r"""Track the completion usage."""
        usage = completion["usage"]
        return CompletionUsage(
            completion_tokens=usage["outputTokens"],
            prompt_tokens=usage["inputTokens"],
            total_tokens=usage["totalTokens"],
        )

    def list_models(self):
        # Initialize Bedrock client (not runtime)

        try:
            response = self._client.list_foundation_models()
            models = response.get("models", [])
            for model in models:
                print(f"Model ID: {model['modelId']}")
                print(f"  Name: {model['name']}")
                print(f"  Description: {model['description']}")
                print(f"  Provider: {model['provider']}")
                print("")
        except Exception as e:
            print(f"Error listing models: {e}")

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
            api_kwargs["messages"] = [
                {"role": "user", "content": [{"text": input}]},
            ]
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
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs
        """
        if model_type == ModelType.LLM:
            return self.sync_client.converse(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    async def acall(self):
        raise NotImplementedError("Async call not implemented yet.")
