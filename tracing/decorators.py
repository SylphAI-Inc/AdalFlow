import functools
import os
import warnings
from typing import List, Optional, Dict, Any
import logging

from core.generator import Generator, GeneratorOutputType
from tracing import GeneratorStatesLogger, GeneratorCallLogger

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
                self.generator_logger = GeneratorStatesLogger(filename=final_file)

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


def trace_generator_error_call(
    attributes: Optional[List[str]] = None, filepath: Optional[str] = "./traces/"
):
    __doc__ = r"""Decorator to trace failed generator predictions in a task component.

    This decorator is a wrapper around the generator call method. It logs the generator call by
    reading its GeneratorOutput and logs the call if the output is an error.
    """

    def decorator(cls):
        original_init = cls.__init__
        class_name = cls.__name__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Ensure directory exists
            if not os.path.exists(filepath):
                os.makedirs(filepath, exist_ok=True)

            # Find generator attributes
            effective_attributes = attributes or [
                attr
                for attr in dir(self)
                if isinstance(getattr(self, attr, None), Generator)
            ]
            generator_names_to_files: Dict[str, str] = {}
            # create the logger in the current component
            if not hasattr(self, "generator_call_logger"):
                self.generator_call_logger = GeneratorCallLogger(dir=filepath)

            generator_names_to_files = (
                self.generator_call_logger.generator_names_to_files
            )

            for attr_name in effective_attributes:
                generator = getattr(self, attr_name, None)

                # handle the file registration
                if attr_name not in generator_names_to_files:
                    self.generator_call_logger.register_generator(attr_name)
                if generator and hasattr(generator, "call"):
                    original_call = generator.call  # TODO: support acall

                    @functools.wraps(original_call)
                    def wrapped_call(*args, **kwargs):
                        output: GeneratorOutputType = original_call(*args, **kwargs)
                        try:
                            if output.error_message is not None:
                                self.generator_call_logger.log_call(
                                    name=attr_name,
                                    model_kwargs=kwargs.get("model_kwargs", {}),
                                    prompt_kwargs=kwargs.get("prompt_kwargs", {}),
                                    output=output,
                                )
                        except Exception as e:
                            logger.error(f"Error logging generator call: {e}")

                    setattr(generator, "call", wrapped_call)

        cls.__init__ = new_init
        return cls

    return decorator
