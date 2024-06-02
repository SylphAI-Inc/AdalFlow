r"""
Decorators for task component to track its generator's prompt changes.
"""

import functools
import os
import warnings

from core.generator import Generator
from tracing.generator_logger import GeneratorLogger


def trace_generator(
    name: str = "default_generator",  # generator name
    attribute: str = "generator",  # the attribute who points to the target generator
    filepath: str = "./traces/",
):
    # ensure the attribute is generator exists and is of type Generator
    assert issubclass(Generator, Generator), "Currently only support Generator class"

    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call the original constructor.
            original_init(self, *args, **kwargs)

            # Dynamically get the attribute to be logged if it exists.
            target = getattr(self, attribute, None)

            if target is None:
                warnings.warn(
                    f"Attribute {attribute} not found in {name}. No tracing will be done."
                )
                return

            if not isinstance(target, Generator):
                warnings.warn(
                    f"Attribute {attribute} is not of type Generator. No tracing will be done."
                )
                return

            # check if the target attribute
            if target and not hasattr(target, "logger"):
                target.logger = GeneratorLogger(
                    filename=os.path.join(filepath, f"{attribute}_{name}_trace.json")
                )

            # Log the initial state if the logger has been attached.
            if hasattr(target, "logger"):
                # This assumes log_states method is capable of logging the state of the target object.
                target.logger.log_prompt(target, name)

        # Replace the original constructor with the new one.
        cls.__init__ = new_init
        return cls

    return decorator


if __name__ == "__main__":
    template = "Hello, {{ input_str }}!"
    from components.api_client import OpenAIClient
    import utils.setup_env

    @trace_generator(name="TestGenerator", attribute="generator")
    class TestGenerator:
        def __init__(self):
            preset_prompt_kwargs = {"input_str": "world"}
            self.generator = Generator(
                model_client=OpenAIClient(),
                template=template,
                preset_prompt_kwargs=preset_prompt_kwargs,
            )
            self.generator1 = 12

        def call(self):
            self.generator.call()

    test_generator = TestGenerator()
