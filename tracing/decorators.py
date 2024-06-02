import functools
import os
import warnings
from typing import List, Dict, Optional
import logging

from core.generator import Generator
from tracing.generator_logger import GeneratorLogger

logger = logging.getLogger(__name__)


def trace_generator(
    # names: List[str] = ["default_generator"],  # unique generator name
    config: Dict[str, str] = {},  # attribute: generator name
    # attribute: str = "generator",  # the attribute who points to the target generator
    filepath: str = "./traces/",
    filename: Optional[str] = None,
):
    __doc__ = r"""Decorator to trace generators in a task component.

    It dynamically attaches a logger to the target generator attribute and logs the prompt states of the generator.
    You can use it on any component that has attributes pointing to a generator object.

    Args:
        config (Dict[str, str]): A dictionary of attribute names and generator names.
        filepath (str): The path to the directory where the trace file will be saved.
        filename (str): The name of the trace file. If not provided, it will be "{class_name}_generator_trace.json".

    """
    assert issubclass(Generator, Generator), "Currently only support Generator class"

    # ensure generator names are unique
    generator_names = list(config.values())
    assert len(set(generator_names)) == len(
        generator_names
    ), "Generator names should be unique."

    def decorator(cls):
        nonlocal filename  # Declare filename as nonlocal to modify it within this scope
        original_init = cls.__init__
        class_name = cls.__name__
        filename = filename or f"{class_name}_generator_trace.json"
        final_filename = os.path.join(filepath, filename)
        logger.info(f"Tracing generator in {class_name} to {final_filename}")

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call the original constructor.
            original_init(self, *args, **kwargs)

            # create the logger in the current component
            if not hasattr(self, "generator_logger"):
                self.generator_logger = GeneratorLogger(filename=final_filename)

            # Dynamically get the attribute to be logged if it exists.
            for attribute, name in config.items():
                target = getattr(self, attribute, None)

                if target is None:
                    warnings.warn(
                        f"Attribute {attribute} not found in {class_name}. Skipping tracing."
                    )
                    continue

                if not isinstance(target, Generator):
                    warnings.warn(
                        f"Attribute {attribute} is not a Generator instance. Skipping tracing."
                    )
                    continue

                # log the prompt states of the target generator
                self.generator_logger.log_prompt(target, name)

        # Replace the original constructor with the new one.
        cls.__init__ = new_init
        return cls

    return decorator


if __name__ == "__main__":
    template = "Hello, {{ input_str }}!"
    from components.api_client import OpenAIClient
    import utils.setup_env

    @trace_generator(
        config={"generator": "TestGenerator", "generator1": "TestGenerator1"}
    )
    class TestGenerator:
        def __init__(self):
            preset_prompt_kwargs = {"input_str": "world"}
            self.generator = Generator(
                model_client=OpenAIClient(),
                template=template,
                preset_prompt_kwargs=preset_prompt_kwargs,
            )
            self.generator1 = Generator(
                model_client=OpenAIClient(),
                template=r"Second generator: {{ input_str }}!",
                preset_prompt_kwargs=preset_prompt_kwargs,
            )

        def call(self):
            self.generator.call()

    test_generator = TestGenerator()
