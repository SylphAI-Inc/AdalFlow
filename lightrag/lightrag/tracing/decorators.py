import functools
import warnings
from typing import List, Optional, Dict
import logging

from lightrag.core.generator import Generator
from lightrag.tracing import GeneratorStateLogger, GeneratorCallLogger

log = logging.getLogger(__name__)


def trace_generator_states(
    attributes: Optional[List[str]] = None,  # list of attributes of type Generator
    save_dir: Optional[str] = "./traces/",
    project_name: Optional[str] = None,
    filename: Optional[str] = None,
):
    r"""Decorator to trace generators in a task component.

    It dynamically attaches a GeneratorLogger to the target generator attribute and logs the prompt states of the generator.
    You can use it on any component that has attributes pointing to a generator object.

    Args:
        attributes (List[str], Optional): The list of attributes that point to the generator objects.
            If not provided, it will automatically detect the attributes that are instances of Generator.
        filepath (str, Optional): The path to the directory where the trace file will be saved. Default is "./traces/".
        filename (str, Optional): The name of the trace file. If not provided, it will be "{class_name}_generator_trace.json".

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
        logger_project_name = project_name or class_name
        # final_filename = filename or f"{class_name}_generator_trace.json"
        # final_file = os.path.join(filepath, final_filename)

        # log.info(f"Tracing generator in {class_name} to {final_file}")

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
                self.generator_logger = GeneratorStateLogger(
                    save_dir=save_dir,
                    project_name=logger_project_name,
                    filename=filename,
                )

            # Dynamically get the attribute to be logged if it exists.
            for attribute in effective_attributes:
                log.debug(f"Tracing generator states in {class_name}")
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
    save_dir: Optional[str] = "./traces/",
    error_only: bool = True,
):
    r"""Decorator to trace generator predictions in a task component, especially failed ones.

    This decorator is a wrapper around the generator call method. It logs the generator call by
    reading its GeneratorOutput and logs the call if the output is an error.

    Args:
        attributes (List[str]): The list of attributes that point to the generator objects.
        save_dir (str): The path to the directory where the trace file will be saved.
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

    If you want to decorate a component(such as LLMRetriever) from the library where you do not have access to the source code, you can do it like this:

    .. code-block:: python
        from lightrag.components.retriever import LLMRetriever

        # Define a subclass and apply the decorator
        @trace_generator_call(save_dir=...)
        class LoggedLLMRetriever(LLMRetriever):
            pass
        retriever = LoggedLLMRetriever(...)


    You can access the logger via TestGenerator.generator_call_logger if you want to access call records in the code.
    """

    def decorator(cls):
        original_init = cls.__init__
        class_name = cls.__name__

        def _wrap_generator(
            generator_name: str,
            generator: Generator,
            error_only: bool,
            logger: GeneratorCallLogger,
        ):
            r"""Wrap the call method of the generator to log the call."""
            original_call = generator.call

            @functools.wraps(original_call)
            def wrapped_call(*args, **kwargs):
                output = original_call(*args, **kwargs)
                try:
                    if (error_only and output.error is not None) or not error_only:
                        log.debug(f"Logging generator call for {generator_name}")
                        logger.log_call(
                            name=generator_name,
                            model_kwargs=kwargs.get("model_kwargs", {}),
                            prompt_kwargs=kwargs.get("prompt_kwargs", {}),
                            output=output,
                        )
                except Exception as e:
                    log.error(f"Error logging generator call for {generator_name}: {e}")
                return output

            return wrapped_call

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

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
                    save_dir=save_dir, project_name=class_name
                )

            generator_names_to_files = (
                self.generator_call_logger.generator_names_to_files
            )

            # Wrap each generator (with attr_name as the generator name)
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
                    filename = self.generator_call_logger.get_log_location(attr_name)
                    log.info(f"Registered generator {attr_name} with file {filename}")
                # Wrap the call method of the target generator
                if target_generator and hasattr(target_generator, "call"):
                    setattr(
                        target_generator,
                        "call",
                        _wrap_generator(
                            generator_name=attr_name,
                            generator=target_generator,
                            error_only=error_only,
                            logger=self.generator_call_logger,
                        ),
                    )

        cls.__init__ = new_init
        return cls

    return decorator
