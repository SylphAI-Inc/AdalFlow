"""
LightRAG's log with colored console output.

Windows users should set up colorama for colored console output.
"""

import logging
import sys

# from colorama import Fore, Style, init
from typing import List
import os

# Initialize Colorama, especially for windows users
# init(autoreset=True)


# # color map, from color string to colorama.Fore colors, easier to use, reduce friction
# COLOR_MAP = {
#     color.lower(): getattr(Fore, color)
#     for color in dir(Fore)
#     if not color.startswith("_")
# }


# create default configuration for the logging, and still allow users to provide more of their own configuration on top of this
# (1) enable both console and file output
# (2) set up the default format: "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
# The format is: time, log level, filename(where the log is from), line number, function name, message
# set up default date format: "%Y-%m-%d %H:%M:%S"
# (3) set up the default log level: INFO
class DefaultLogging:
    r"""LightRAG's easy-to-customize logging system with colorized console output.

    LightRAG's logging features one-line configuration after importing. You can set up:

    * colored console output that can be easily distinguished from printed strings.
    * file output in target directory.

    **Allowed colors:** ['black', 'blue', 'cyan', 'green', 'lightblack_ex', 'lightblue_ex', 'lightcyan_ex', 'lightgreen_ex', 'lightmagenta_ex', 'lightred_ex', 'lightwhite_ex', 'lightyellow_ex', 'magenta', 'red', 'reset', 'white', 'yellow']
    You may use different colors in different code component.

    Example:

        Console output setting

        .. code-block:: python

            from utils.logger import PatchedLogging

            logger = PatchedLogging.getLogger(log_level=logging.DEBUG)
            logger.info("This is an info message", color="Blue")
            logger.info("This is an info message", color="green")
            do_func()
            print('something')
            logger.warning("This is a warning message")
            logger.error("This is an error message")
            logger.debug("This is a debug message")


        File output setting

        .. code-block:: python

            from utils.logger import PatchedLogging

            logger = PatchedLogging.getLogger(output_type="file")
            logger.info("This is an info message")
            do_func()
            print('something')
            logger.warning("This is a warning message")
            logger.error("This is an error message")
            logger.debug("This is a debug message")

    You can use it to show system logs, execution states, and application input/output data flow.

    .. note::
        * You can either use console output or file.
        * Log file doesn't support colors. Please don't add color parameters when your output type is file.
        * Please set up your own directory and filename for log files, the default path is ``./logs/app.log``.
        * OSError will be raised if your configured directory can't be created.
    """

    # default color for messages at different level
    # LOG_COLORS = {
    #     "DEBUG": Fore.CYAN,
    #     "INFO": Fore.GREEN,
    #     "WARNING": Fore.YELLOW,
    #     "ERROR": Fore.RED,
    #     "CRITICAL": Fore.MAGENTA,
    # }

    # Define log levels within the class
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    @staticmethod
    def get_level(level_name: str) -> int:
        """Return the logging level constant based on a string."""
        return DefaultLogging.LOG_LEVELS.get(level_name.upper(), logging.INFO)

    @staticmethod
    def getLogger(
        name: str = "default",
        filename: str = "./logs/app.log",  # path and filename
        level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
    ) -> logging.Logger:
        logger = logging.getLogger(name=name)
        level = DefaultLogging.get_level(level)
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
        logger.propagate = False

        # def create_custom_log_method(level):
        #     def custom_log_method(message, *args, color=None, **kwargs):
        #         caller_frame = inspect.stack()[
        #             1
        #         ]  # Adjust index based on your wrapper's depth
        #         caller_info = {
        #             "filename": caller_frame.filename,
        #             "lineno": caller_frame.lineno,
        #             "funcName": caller_frame.function,
        #         }
        #         if color and output_type == "console":
        #             color_code = COLOR_MAP.get(
        #                 color.lower(),
        #                 PatchedLogging.LOG_COLORS[logging.getLevelName(level).upper()],
        #             )
        #             message = color_code + message + Style.RESET_ALL
        #         kwargs["extra"] = caller_info
        #         logger.log(level, message, *args, **kwargs)

        #     return custom_log_method

        # for levelname, level in PatchedLogging.LOG_LEVELS.items():
        #     setattr(logger, levelname.lower(), create_custom_log_method(level))

        return logger


if __name__ == "__main__":
    # console output:
    # print(f"Colors you can choose: {list(COLOR_MAP.keys())}")
    logger = DefaultLogging.getLogger(level="DEBUG")
    logger.info("This is an info message")
    logger.info("This is an info message")  # , color="Blue")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")

    # # file output
    # logger = PatchedLogging.getLogger(output_type="file")
    # logger.info("This is an info message")
    # logger.warning("This is a warning message")
    # logger.error("This is an error message")
    # logger.debug("This is a debug message")