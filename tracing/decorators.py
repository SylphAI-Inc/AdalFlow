import functools
import os
import warnings
from typing import List, Optional, Dict, Any
import logging

from core.generator import Generator, GeneratorOutputType
from tracing import GeneratorStatesLogger, GeneratorCallLogger

log = logging.getLogger(__name__)


def trace_generator_states(
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

        log.info(f"Tracing generator in {class_name} to {final_file}")

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
                log.debug(f"Creating generator states logger for {class_name}")
                self.generator_logger = GeneratorStatesLogger(filename=final_file)

            # Dynamically get the attribute to be logged if it exists.
            for attribute in effective_attributes:
                log.debug(f"Tracing generator in {class_name}")
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
                self.generator_logger.log_prompt(target, generator_name)

        cls.__init__ = new_init
        return cls

    return decorator


def trace_generator_call(
    attributes: Optional[List[str]] = None,
    filepath: Optional[str] = "./traces/",
    error_only: bool = True,
):
    __doc__ = r"""Decorator to trace generator predictions in a task component, especially failed ones.

    This decorator is a wrapper around the generator call method. It logs the generator call by
    reading its GeneratorOutput and logs the call if the output is an error.

    Args:
        attributes (List[str]): The list of attributes that point to the generator objects.
        filepath (str): The path to the directory where the trace file will be saved.
        error_only (bool): If True, only log the calls that have an error. Default is True.

    Examples:
        >>> @trace_generator_call()
        >>> class TestGenerator:
        >>>     def __init__(self):
        >>>         preset_prompt_kwargs = {"input_str": "world"}
        >>>         self.generator = Generator(
        >>>             model_client=OpenAIClient(),
        >>>             template=template,
        >>>             preset_prompt_kwargs=preset_prompt_kwargs,
        >>>         )
        >>> # now you will see ./traces/TestGenerator dir being created.
        >>> # If the generator call has an error, it will be logged in the error file generator_call.jsonl

    You can access the logger via TestGenerator.generator_call_logger if you want to access call records in the code.
    """

    def decorator(cls):
        original_init = cls.__init__
        class_name = cls.__name__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
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
                self.generator_call_logger = GeneratorCallLogger(
                    dir=filepath, project_name=class_name
                )

            generator_names_to_files = (
                self.generator_call_logger.generator_names_to_files
            )

            for attr_name in effective_attributes:
                target_generator = getattr(self, attr_name, None)
                if target_generator is None:
                    warnings.warn(
                        f"Attribute {attr_name} not found in {class_name}. Skipping tracing."
                    )
                    continue

                # handle the file registration
                if attr_name not in generator_names_to_files:

                    self.generator_call_logger.register_generator(attr_name)
                    filename = self.generator_call_logger.get_location(attr_name)
                    log.info(f"Registered generator {attr_name} with file {filename}")
                if target_generator and hasattr(target_generator, "call"):
                    original_call = target_generator.call  # TODO: support acall

                    @functools.wraps(original_call)
                    def wrapped_call(*args, **kwargs):
                        output: GeneratorOutputType = original_call(*args, **kwargs)
                        try:
                            if error_only and output.error_message is not None:
                                self.generator_call_logger.log_call(
                                    name=attr_name,
                                    model_kwargs=kwargs.get("model_kwargs", {}),
                                    prompt_kwargs=kwargs.get("prompt_kwargs", {}),
                                    output=output,
                                )
                            if not error_only:
                                self.generator_call_logger.log_call(
                                    name=attr_name,
                                    model_kwargs=kwargs.get("model_kwargs", {}),
                                    prompt_kwargs=kwargs.get("prompt_kwargs", {}),
                                    output=output,
                                )
                        except Exception as e:
                            log.error(f"Error logging generator call: {e}")
                        return output

                    setattr(target_generator, "call", wrapped_call)

        cls.__init__ = new_init
        return cls

    return decorator
