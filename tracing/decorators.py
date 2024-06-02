import functools
import os
import warnings
from typing import List, Optional
import logging

from core.generator import Generator
from tracing.generator_logger import GeneratorLogger

logger = logging.getLogger(__name__)


def trace_generator(
    attributes: Optional[List[str]] = None,  # list of attributes
    filepath: Optional[str] = "./traces/",
    filename: Optional[str] = None,
):
    __doc__ = r"""Decorator to trace generators in a task component.

    It dynamically attaches a GeneratorLogger to the target generator attribute and logs the prompt states of the generator.
    You can use it on any component that has attributes pointing to a generator object.

    Args:
        attributes (List[str]): The list of attributes that point to the generator objects.
        filepath (str): The path to the directory where the trace file will be saved.
        filename (str): The name of the trace file. If not provided, it will be "{class_name}_generator_trace.json".

    Examples:
        >>> @trace_generator()
        >>> class TestGenerator:
        >>>     def __init__(self):
        >>>         preset_prompt_kwargs = {"input_str": "world"}
        >>>         self.generator = Generator(
        >>>             model_client=OpenAIClient(),
        >>>             template=template,
        >>>             preset_prompt_kwargs=preset_prompt_kwargs,
        >>>         )
        >>> # now you will see log files in the ./traces/ with a filename like TestGenerator_generator_trace.json
        >>> # If you update the template or the preset_prompt_kwargs, it will be logged in the file.
    """
    assert issubclass(Generator, Generator), "Currently only support Generator class"

    def decorator(cls):
        original_init = cls.__init__
        class_name = cls.__name__
        final_filename = filename or f"{class_name}_generator_trace.json"
        final_file = os.path.join(filepath, final_filename)

        logger.info(f"Tracing generator in {class_name} to {final_file}")

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # enable automatic detection of generator attributes
            effective_attributes = attributes or [
                attr
                for attr in dir(self)
                if isinstance(getattr(self, attr, None), Generator)
            ]

            # create the logger in the current component
            if not hasattr(self, "generator_logger"):
                print("Creating generator logger")
                self.generator_logger = GeneratorLogger(filename=final_file)

            # Dynamically get the attribute to be logged if it exists.
            for attribute in effective_attributes:
                print(f"Tracing generator in {class_name}")
                target = getattr(self, attribute, None)
                generator_name = attribute

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
                print(f"Logging prompt states of {generator_name}")
                self.generator_logger.log_prompt(target, generator_name)

        cls.__init__ = new_init
        return cls

    return decorator
