import logging
import os
import sys
from typing import Optional
import colorama
import structlog


# TODO: 
# 1. Add colors to the log: https://www.structlog.org/en/stable/console-output.html
# 2. Create a documentation
# 3. Think about how to set up the log once and reduce the user's manual work to use logger.info
class LogSettings:
    """
    Configures the logging system with structured and output.
    """
    def __init__(self, logger_name: str = "AppLog", output_type: str = "str", method: str = "console", file_name: Optional[str] = None, log_level: str = "DEBUG") -> None:
        self.output_type = output_type
        self.method = method
        self.file_name = file_name
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.configure_logging()
        # self.setup_structlog()
        # self.set_log_output()

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
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # format the message
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