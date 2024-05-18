"""
LLM application log is unique, as developers pay much attention to the input text, intermediate data flow and final output.
This logging system aims to facilitate the developers check the application in a flexible way.
It is easy to use while highly configurable.

Reference: https://github.com/stanfordnlp/dspy/blob/main/dspy/utils/logging.py
"""

import logging
import os
import sys
from typing import Optional
import structlog


# TODO:
# 1. Add colors to the log console output: https://www.structlog.org/en/stable/console-output.html
# 2. Wrap it with other functions/components


def setup_handler(handler: logging.Handler, log_level: str):
    """
    Setup the logging handler with the specified log level.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )  # add time and filename
    handler.setFormatter(formatter)  # configure the formatter
    handler.setLevel(log_level)  # set log levels to output


def configure_logging(
    logger: logging.Logger, log_level: str, method: str, file_name: str
):
    """
    Configure the logging system with structured and output.
    """
    if method == "console":
        renderer = structlog.dev.ConsoleRenderer()
    elif method == "file":
        # check if the path exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        renderer = structlog.processors.JSONRenderer()
    else:
        raise ValueError(f"Invalid method: {method}")

    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

    logger.setLevel(logging.getLevelName(log_level))
    handler = (
        logging.FileHandler(file_name)
        if method == "file"
        else logging.StreamHandler(sys.stdout)
    )
    setup_handler(handler, log_level)
    logger.addHandler(handler)


def get_logger(
    logger_name: str = "AppLog",
    method: str = "file",
    file_name: Optional[str] = "./logs/app.log",
    log_level: str = "INFO",
):
    """
    Configures the logging system with structured and output.
    method: str, default="file", options=["console", "file"]
    file_name: str, default=None, the file name to store the logs
    log_level: str, default="INFO", options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    Example:

    from utils.logger import get_logger
    logger = get_logger(file_name="./logs/app.log")  # Retrieve the configured logger
    simple_qa = SimpleQA()
    logger.info(simple_qa)
    response = simple_qa.call("What is the capital of France?")
    logger.info(f'response: {response}')

    Then you can open ./logs/app.log and view the content.
    """
    logger = logging.getLogger(logger_name)
    # create the logs directory if it does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    configure_logging(logger, log_level, method, file_name)
    return logger


if __name__ == "__main__":
    from utils.logger import get_logger
    from use_cases.simple_qa_groq import SimpleQA

    logger = get_logger(method="console")  # Retrieve the configured logger
    simple_qa = SimpleQA()
    logger.info(simple_qa)
    response = simple_qa.call("What is the capital of France?")
    logger.info(f"response: {response}")
    # log_settings = LogSettings(
    #     output_type="str", method="file", file_name="./utils/test.log", log_level="INFO"
    # )
    # logger = (
    #     log_settings.get_logger()
    # )  # Explicitly get the configured logger from log_settings
    # logger.info("hello this is logger")  # Log message to 'test.log'

    # check ./utils/test.log to view the log
