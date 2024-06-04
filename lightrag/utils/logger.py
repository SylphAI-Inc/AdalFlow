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
from typing import List, Tuple, Optional
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

log = logging.getLogger(__name__)


# NOTE: what if users set up two logger in the same script with the same name, we dont config the logger again, but we add the handler again, is it a problem?
# NOTE: When both console and file are false, the logger should not have any handlers.
def get_default_log_config(
    level: str = "INFO",
    filepath: str = "./logs/app.log",
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
            logger = get_default_logger(name=__name__, level="DEBUG") # set level = debug
            logger.info("This is an info message")
            logger.warning("This is a warning message")

    Returns:
        logging.Logger: A configured logger, following the same logic with Python default logger, you can add more configuration such as add Handler to it.
    """

    def get_level(level: str) -> int:
        """Return the logging level constant based on a string."""
        return LOG_LEVELS.get(level.upper(), logging.INFO)

    # 1. create the logger the library developers want to use
    # logger = logging.getLogger(name=name)

    # if logger.handlers:
    # log.info(f"Logger {name} already has handlers, skip the configuration.")

    # return logger

    # 2. Config the default format and style
    format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")

    # Set up the handler
    handlers: List[logging.Handler] = []
    if enable_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if enable_file:
        file_path: str = os.path.dirname(filepath)
        os.makedirs(file_path, exist_ok=True)
        handler = logging.FileHandler(filename=filepath, mode="a")
        handlers.append(handler)

    for h in handlers:
        h.setFormatter(formatter)
        # logger.addHandler(h)

    # prevent duplication in the root logger
    # logger.propagate = False
    # print(
    #     f"Logger {name} is configured with level {level}, console: {enable_console}, file: {enable_file}"
    # )

    # 3. Enable a global config that will enable the library logs to be shown
    # logging.basicConfig(level=get_level(level), handlers=handlers)
    return get_level(level), handlers  # logger


def enable_library_logging(
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = False,
    # filename: str = "./logs/app.log",
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    r"""Config the library logging.

    We config logging with the following default settings:
    1. Enable console output, and disable file output.
    2. Set up the default format: "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s".
    The format is: time, log level, filename(where the log is from), line number, function name, message.
    3. Set up the default date format: "%Y-%m-%d %H:%M:%S".

    Args:
        level str: Log level. Defaults to "INFO". Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        enable_console bool: Control the console output. Defaults to True.
        enable_file bool: Control the file output. Defaults to False.
        filename str: Name of the output log file. Defaults to "./logs/app.log".

    """
    save_dir = save_dir or "./logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or f"app.log"
    filepath = os.path.join(save_dir, filename)
    default_config = get_default_log_config(
        level=level,
        enable_console=enable_console,
        enable_file=enable_file,
        filepath=filepath,
    )
    logging.basicConfig(level=default_config[0], handlers=default_config[1])


def get_logger(
    name: str,
    level: str = "INFO",
    # filename: str = "./logs/app.log",
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:

    save_dir = save_dir or "./logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or f"app.log"
    filepath = os.path.join(save_dir, filename)

    config = get_default_log_config(
        level=level,
        filepath=filepath,
        enable_console=enable_console,
        enable_file=enable_file,
    )
    logger = logging.getLogger(name)
    logger.setLevel(config[0])
    for handler in config[1]:
        logger.addHandler(handler)
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
        color (str): Color of the text. Defaults to "cyan". Options:
        'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow'. Defaults to "cyan".

    Example:
        .. code-block:: python

            from utils.logger import colored_print
            printc("hello", color="green")
            printc("hello")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    function_name, script_name, line_number = get_current_script_and_line()
    color_code = COLOR_MAP.get(color, COLOR_MAP["cyan"])
    print(
        f"{color_code}{timestamp} - [{script_name}:{line_number}:{function_name}] - {text}\033[0m"
    )
    # \033[0m means reset, not impacting the next print texts
