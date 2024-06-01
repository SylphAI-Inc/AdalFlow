"""
LightRAG's easy-to-customize logging system with colorized console output.

On Windows versions prior to Windows 10, the Command Prompt and PowerShell do not support ANSI color codes. You would need to use third-party libraries like colorama to enable ANSI color support.
Please add the following colorama setting at the beginning of your code:

Installation:

.. code-block:: bash

    pip install colorama

Initialization:

.. code-block:: python

    import colorama
    colorama.init(autoreset=True)
"""

import logging
import sys
from typing import List, Optional
import inspect
import os
from datetime import datetime

# create default configuration for the logging, and still allow users to provide more of their own configuration on top of this
# (1) enable both console and file output
# (2) set up the default format: "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
# The format is: time, log level, filename(where the log is from), line number, function name, message
# set up default date format: "%Y-%m-%d %H:%M:%S"
# (3) set up the default log level: INFO
# (4) offer a colored print to help debug with timestamp and path

# Define log levels within the class
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
    
# color map, from color string to colorama.Fore colors, easier to use, reduce friction
COLOR_MAP = {
    "black": "\x1b[30m", "blue": "\x1b[34m", "cyan": "\x1b[36m", "green": "\x1b[32m", 
    "magenta": "\x1b[35m", "red": "\x1b[31m",  "white": "\x1b[37m", "yellow": "\x1b[33m"
}

# Helper function to get log level from string
def get_level(level: str) -> int:
    """Return the logging level constant based on a string."""
    return LOG_LEVELS.get(level.upper(), logging.INFO)

def get_default_logger(
    name: str = "default",
    filename: str = "./logs/app.log",
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    r"""LightRAG's easy-to-customize logging system.

    LightRAG's logging features one-line configuration. We wrapped the format, the log level and handlers. 
    Users only need least effort to set up the logger but can still do their own configuration on top of the logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "default".
        filename (str, optional): Name of the output log file. Defaults to "./logs/app.log".
        level (str, optional): Log level. Defaults to "INFO".
        enable_console (bool, optional): Control the console output. Defaults to True.
        enable_file (bool, optional): Control the file output. Defaults to True.
        
    Example:
        .. code-block:: python

            from utils.logger import get_default_logger
            logger = get_default_logger(level="DEBUG") # set level = debug
            logger.info("This is an info message")
            logger.warning("This is a warning message")
            
    Returns:
        logging.Logger: A configured logger, following the same logic with Python default logger, you can add more configuration such as add Handler to it.
    """
    logger = logging.getLogger(name=name)
    level = get_level(level)
    logger.setLevel(level)

    # Enable default formatter
    format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")

    # Set up the handler
    handlers: List[logging.Handler] = []
    if enable_console:
        handler = logging.StreamHandler(sys.stdout)
        handlers.append(handler)
    if enable_file:
        file_path: str = os.path.dirname(filename)
        os.makedirs(file_path, exist_ok=True)
        handler = logging.FileHandler(filename=filename, mode="a")
        handlers.append(handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # prevent duplication in the root logger
    logger.propagate = False

    return logger


def get_current_script_and_line():
    """helper function to get function, script, line

    Returns:
        function_name (str): the name of the function that is calling.
        script_name (str): the name of the script that is calling.
        line_number (str): the line number of the code that is calling.
    """
    caller_frame = inspect.stack()[2]
    file_path, line_number, function_name = caller_frame.filename, caller_frame.lineno, caller_frame.function
    script_name = os.path.basename(file_path)
    
    return function_name, script_name, line_number


def colored_print(text: str, color: Optional[str] = "cyan") -> None:
    """LightRAG's colored print for debugging.
    
    LightRAG's customized print with colored text, position of code block the print is set, and current timestamp.

    Args:
        text (str): Text to be printed.
        color (Optional[str], optional): Supported colors to apply with texts:
        'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow'. Defaults to "cyan".
    
    Example:
        .. code-block:: python

            from utils.logger import colored_print
            colored_print("hello", color="green")
            colored_print("hello")
            
    Returns:
        None
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    function_name, script_name, line_number = get_current_script_and_line()
    color_code = COLOR_MAP.get(color, COLOR_MAP["cyan"])
    print(f"{color_code}{timestamp} - [{script_name}:{line_number}:{function_name}] - {text}\033[0m") 
    # \033[0m means reset, not impacting the next print texts

if __name__ == "__main__":
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    baselogger = logging.getLogger(__name__)
    baselogger.info("this is base logger")
    logger = get_default_logger(level="DEBUG")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    colored_print("hello world", color="green")
    colored_print("hello world")