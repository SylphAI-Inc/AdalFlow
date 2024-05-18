"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

from core.generator import Generator
from core.openai_client import OpenAIClient

from core.component import Component

# TODO: make the environment variable loading more robust, and let users specify the .env path
import dotenv
import os

dotenv.load_dotenv()

class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "gpt-3.5-turbo"}
        self.generator = Generator(
            model_client=OpenAIClient(), model_kwargs=model_kwargs
        )
        self.generator.print_prompt()

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    from utils.logging_config import LogSettings
    # configure the logging once and use it
    log_settings = LogSettings(file_name="./tests/log_test/app.log")
    logger = log_settings.get_logger()  # Retrieve the configured logger
    
    simple_qa = SimpleQA()
    logger.info(simple_qa)
    response = simple_qa.call("What is the capital of France?")
    logger.info(f'response: {response}')