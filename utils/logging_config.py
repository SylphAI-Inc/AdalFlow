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
class LogSettings:
    """
    Configures the logging system with structured and output.
    
    Example:

    from utils.logging_config import LogSettings
    # configure the logging once and use it in the code
    log_settings = LogSettings(file_name="./tests/log_test/app.log")
    logger = log_settings.get_logger()  # Retrieve the configured logger
    simple_qa = SimpleQA()
    logger.info(simple_qa)
    response = simple_qa.call("What is the capital of France?")
    logger.info(f'response: {response}')
    
    Then you can open ./tests/log_test/app.log and view the content.
    """
    def __init__(self, logger_name: str = "AppLog", output_type: str = "str", method: str = "file", file_name: Optional[str] = None, log_level: str = "INFO") -> None:
        self.output_type = output_type
        self.method = method
        self.file_name = file_name
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.configure_logging()

    def configure_logging(self):
        """
        Configures logging with structlog and standard logging based on settings.
        """
        renderer = structlog.dev.ConsoleRenderer() if self.method == "console" and self.output_type == "str" else structlog.processors.JSONRenderer()
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

        self.logger.setLevel(logging.getLevelName(self.log_level))
        self.setup_handlers()

    def setup_handlers(self):
        """
        Setup logging handlers based on the specified method.
        """
        self.logger.handlers.clear()  # Clear existing handlers to avoid duplication
        handler = logging.FileHandler(self.file_name) if self.method == "file" else logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # add time and filename
        handler.setFormatter(formatter) # configure the formatter
        handler.setLevel(logging.getLevelName(self.log_level)) # set log levels to output
        self.logger.addHandler(handler)
        
    
    def get_logger(self):
        """
        Returns the configured logger.
        """
        return self.logger
    

if __name__ == "__main__":
    log_settings = LogSettings(output_type="str", method="file", file_name="./utils/test.log", log_level="INFO")
    logger = log_settings.get_logger()  # Explicitly get the configured logger from log_settings
    logger.info('hello this is logger')  # Log message to 'test.log'
    
    # check ./utils/test.log to view the log