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
from typing import List, Tuple, Optional, Literal
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


def _get_log_config(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    filepath: str = "./logs/app.log",
    enable_console: bool = True,
    enable_file: bool = True,
) -> Tuple[int, List[logging.Handler]]:
    r"""Helper function to get the default log configuration.

    We config logging with the following default settings:
    1. Enable both console and file output.
    2. Set up the default format: "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s".
    The format is: time, log level, filename(where the log is from), line number, function name, message.
    3. Set up the default date format: "%Y-%m-%d %H:%M:%S".

    Args:
        level (str): Log level. Defaults to "INFO". Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        filepath (str): Name of the output log file. Defaults to "./logs/app.log".
        enable_console (bool): Control the console output. Defaults to True.
        enable_file (bool): Control the file output. Defaults to True.

    Returns:
        Tuple[int, List[logging.Handler]]: The logging level and the list of handlers.
    """

    def get_level(level: str) -> int:
        """Return the logging level constant based on a string."""
        return LOG_LEVELS.get(level.upper(), logging.INFO)

    # 2. Config the default format and style
    format = "%(asctime)s - %(module)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
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

    return get_level(level), handlers


def enable_library_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    enable_console: bool = True,
    enable_file: bool = False,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> logging.Logger:
    r"""Enable the library logging which is the root logger.

    Root logger has no name or '' as the name. It is the ancestor of all loggers.

    The default config follows :func:`get_default_log_config`.

    Example:

    1. Enable the library logging with default settings which outputs library logs to console:

    .. code-block:: python

        from lightrag.utils.logger import enable_library_logging

        enable_library_logging(level="DEBUG", enable_console=True, enable_file=False)

    2. Enable the library logging with default settings which outputs library logs to file:

    .. code-block:: python

        from lightrag.utils.logger import enable_library_logging

        enable_library_logging(level="DEBUG", enable_console=False, enable_file=True, filename="library.log")

    3. Enable the library logging along with getting a logger of the same configuration:

    .. code-block:: python

        from lightrag.utils.logger import enable_library_logging

        logger = enable_library_logging(level="DEBUG", enable_console=True, enable_file=False, return_logger=True)



    Args:
        level (str): Log level. Defaults to "INFO". Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        enable_console (bool): Control the console output. Defaults to True.
        enable_file (bool): Control the file output. Defaults to False.
        save_dir (Optional[str]): Directory to save log files. Defaults to "./logs".
        filename (Optional[str]): Name of the output log file. Defaults to "lib.log".
        return_logger (bool): Return the logger with the same configuration. Defaults to False.
    """
    # reset the past logging configuration
    save_dir = save_dir or "./logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or "lib.log"
    filepath = os.path.join(save_dir, filename)
    default_config = _get_log_config(
        level=level,
        enable_console=enable_console,
        enable_file=enable_file,
        filepath=filepath,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(default_config[0])
    root_logger.handlers = []  # Clear existing handlers
    for handler in default_config[1]:
        root_logger.addHandler(handler)

    return root_logger


def get_logger(
    name: str,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    # filename: str = "./logs/app.log",
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    r"""Get a named logger without changing or writing to the root logger.

    Args:
        name (str): Name of the logger.
        level (str): Log level. Defaults to "INFO". Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        save_dir (Optional[str]): Directory to save log files. Defaults to "./logs".
        filename (Optional[str]): Name of the output log file. Defaults to "app.log".
        enable_console (bool): Control the console output. Defaults to True.
        enable_file (bool): Control the file output. Defaults to True.

    Example:

    .. code-block:: python

        from lightrag.utils.logger import get_logger, enable_library_logging

        filename = "lib.log"

        enable_library_logging(level="DEBUG", enable_console=True, enable_file=True, filename=filename)

        app_log_filename = "app.log"

        logger = get_logger(name="app", level="DEBUG", enable_console=True, enable_file=True, filename=app_log_filename)

        # This will result two log files, one for library and one for app
    """
    assert name, "Name of the logger cannot be None or empty"

    save_dir = save_dir or "./logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or "app.log"
    filepath = os.path.join(save_dir, filename)

    config = _get_log_config(
        level=level,
        filepath=filepath,
        enable_console=enable_console,
        enable_file=enable_file,
    )
    logger = logging.getLogger(name)
    logger.setLevel(config[0])
    logger.handlers = []  # Clear existing handlers
    for handler in config[1]:
        logger.addHandler(handler)
    logger.propagate = False  # Do not propagate to the root logger
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
