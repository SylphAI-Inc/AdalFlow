from core.generator import Generator
from core.component import Component

from components.api_client import GroqAPIClient
from tracing.decorators import trace_generator_states, trace_generator_call

import utils.setup_env


@trace_generator_states()
@trace_generator_call(error_only=False)
class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": "You are a helpful assistant and with a great sense of humor. changes",
            },
            trainable_params=[
                "task_desc_str"
            ],  # 1 we need to clearly define which is trainable.
        )
        self.generator2 = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": "You are the second generator.",
            },
        )

    def call(self, query: str) -> str:
        return self.generator2.call(prompt_kwargs={"input_str": query})


if __name__ == "__main__":
    from utils import enable_library_logging, get_logger

    enable_library_logging(enable_file=False, level="DEBUG")
    log = get_logger(__name__, level="INFO")
    simple_qa = SimpleQA()
    log.info(simple_qa)
    states = simple_qa.state_dict()  # only prameters
    print(f"states: {states}")

    print("show the system prompt")
    simple_qa.generator.print_prompt()
    records = simple_qa.generator_call_logger.load("generator")
    print(f"records: {records}")
    print("Answer:")
    print(simple_qa.call("What is the capital of France?"))
