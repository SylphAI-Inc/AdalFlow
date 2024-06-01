"""
LightRAG's provides default logger and colored print function.

NOTE: If you are on Windows versions prior to Windows 10, the Command Prompt and PowerShell do not support ANSI color codes. You would need to use third-party libraries like colorama to enable ANSI color support.
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
from typing import List, Tuple
import inspect
import os
from datetime import datetime

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# color map, from color string to colorama.Fore colors, easier to use, reduce friction
COLOR_MAP = {
    "black": "\x1b[30m",
    "blue": "\x1b[34m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "magenta": "\x1b[35m",
    "red": "\x1b[31m",
    "white": "\x1b[37m",
    "yellow": "\x1b[33m",
}


# TODO: what if users set up two logger in the same script with the same name, will it cause any issue?
# TODO: does it make sense when both enable_console and enable_file are False?
def get_default_logger(
    name: str = "default",
    filename: str = "./logs/app.log",
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    r"""Default logger to simplify the logging configuration.

    We config logging with the following default settings:
    1. Enable both console and file output.
    2. Set up the default format: "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s".
    The format is: time, log level, filename(where the log is from), line number, function name, message.
    3. Set up the default date format: "%Y-%m-%d %H:%M:%S".

    Args:
        name str: Name of the logger. Defaults to "default". Users should pass '__name__' to get the logger name as the module name.
        filename str: Name of the output log file. Defaults to "./logs/app.log".
        level str: Log level. Defaults to "INFO". Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        enable_console bool: Control the console output. Defaults to True.
        enable_file bool: Control the file output. Defaults to True.

    Example:
        .. code-block:: python

            from utils.logger import get_default_logger
            logger = get_default_logger(level="DEBUG") # set level = debug
            logger.info("This is an info message")
            logger.warning("This is a warning message")

    Returns:
        logging.Logger: A configured logger, following the same logic with Python default logger, you can add more configuration such as add Handler to it.
    """

    def get_level(level: str) -> int:
        """Return the logging level constant based on a string."""
        return LOG_LEVELS.get(level.upper(), logging.INFO)

    logger = logging.getLogger(name=name)
    # follow the default logger behavior, if the logger already has handlers, we skip the configuration
    if logger.handlers:

        return logger

    logger.setLevel(get_level(level))

    # Enable default formatter
    format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")

    # Set up the handler
    handlers: List[logging.Handler] = []
    if enable_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if enable_file:
        file_path: str = os.path.dirname(filename)
        os.makedirs(file_path, exist_ok=True)
        handler = logging.FileHandler(filename=filename, mode="a")
        handlers.append(handler)

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    # prevent duplication in the root logger
    logger.propagate = False

    return logger


def get_current_script_and_line() -> Tuple[str, str, int]:
    """Get the function name, script name, and line of where the print is called.

    Returns:
        function_name (str): the name of the function that is calling.
        script_name (str): the name of the script that is calling.
        line_number (str): the line number of the code that is calling.
    """
    caller_frame = inspect.stack()[2]
    file_path, line_number, function_name = (
        caller_frame.filename,
        caller_frame.lineno,
        caller_frame.function,
    )
    script_name = os.path.basename(file_path)

    return function_name, script_name, line_number


def printc(text: str, color: str = "cyan"):
    """Color enhanced print function along with logger-like format.

    LightRAG's customized print with colored text, position of code block the print is set, and current timestamp.

    Args:
        text (str): Text to be printed.
        color (Optional[str], optional): Color of the text. Options:
        'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow'. Defaults to "cyan".

    Example:
        .. code-block:: python

            from utils.logger import colored_print
            colored_print("hello", color="green")
            colored_print("hello")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    function_name, script_name, line_number = get_current_script_and_line()
    color_code = COLOR_MAP.get(color, COLOR_MAP["cyan"])
    print(
        f"{color_code}{timestamp} - [{script_name}:{line_number}:{function_name}] - {text}\033[0m"
    )
    # \033[0m means reset, not impacting the next print texts


if __name__ == "__main__":

    print(f"logger name: {__name__}")
    logger = get_default_logger(name=__name__, level="DEBUG")

    def test_logger(logger):
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        printc("hello world", color="green")
        printc("hello world")
        printc(logger.name)

    test_logger(logger)
