"""
Colorize console logs to separate logs from prints 

Windows users should use colorama for colored console output 
"""

import logging
import sys
from colorama import Fore, Style, init
from typing import Dict, Union
import os

# Initialize Colorama, especially for windows users
init(autoreset=True)

# default color for messages at different level
LOG_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.MAGENTA
}

# color map, from color string to colorama.Fore colors, easier to use, reduce friction
COLOR_MAP = {color.lower(): getattr(Fore, color) for color in dir(Fore) if not color.startswith('_')}

class PatchedLogging:
    r"""LightRAG's easy-to-customize logging system with colorized console output.
    
    Example:
        .. code-block:: python
        
            from utils.logger import PatchedLogging
            
            # console output
            # The colors you can choose: ['black', 'blue', 'cyan', 'green', 'lightblack_ex', 'lightblue_ex', 'lightcyan_ex', 'lightgreen_ex', 'lightmagenta_ex', 'lightred_ex', 'lightwhite_ex', 'lightyellow_ex', 'magenta', 'red', 'reset', 'white', 'yellow']
            # log_level you can set: https://docs.python.org/3/library/logging.html#levels
            logger = PatchedLogging.getLogger(log_level=logging.DEBUG)
            logger.info('This is an info message', color="Blue")
            
            # file output
            logger = PatchedLogging.getLogger(output_type="file")
            logger.info("This is an info message")
            logger.warning("This is a warning message")
            logger.error("This is an error message")
            logger.debug("This is a debug message")
    
    .. note::
        You can either use console output or file. Log file doesn't support colors. Please don't add color parameters when your output type is file. 
        Please set up your own directory and filename for log files, the default path is ./logs/app.log
        OSError will be raised if your configured directory can't be created.
    """
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def getLogger(output_type:str = "console", dir:str = "./logs", filename:str = "app.log", log_level:int = logging.INFO, format: Union[str|None] = None) -> logging.Logger:
        """Configure logger in one method

        Args:
            output_type (str, optional): _description_. Defaults to "console".
            dir (str, optional): _description_. Defaults to "./logs".
            filename (str, optional): _description_. Defaults to "app.log".
            log_level (int, optional): _description_. Defaults to logging.INFO.
            format (Union[str | None], optional): _description_. Defaults to None.

        Returns:
            logging.Logger: the configured logger
        """
        # set up logger with log level
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)  # default log level is INFO

        # if output in the console, then use stream handler, else check the file path(create if not exists) and use the file handler
        if output_type == "console":
            handler = logging.StreamHandler(sys.stdout)
        else:
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir, exist_ok=True)
            except:
                raise OSError(f"Directory {dir} can't be created")
            
            file_path = os.path.join(dir, filename)
            handler = logging.FileHandler(file_path)
        
        # reset format
        if not format:
            format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        
        # set up format
        formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        def create_custom_log_method(level):
            def custom_log_method(message, *args, color=None, **kwargs):
                if output_type == "file" and color is not None:
                    raise ValueError("Color is not supported in log files")
                    
                if output_type == "console":
                    # Use default color if not provided, when the user input wrong color, the default is yellow
                    color = COLOR_MAP.get(color.lower(), "\x1b[33m") if color is not None else LOG_COLORS.get(logging.getLevelName(level))
                    message = color + message + Style.RESET_ALL
                
                logger.log(level, message, *args, **kwargs)
            return custom_log_method

        # Assign custom log methods with color support
        for levelname, _ in LOG_COLORS.items():
            level = getattr(logging, levelname)
            setattr(logger, levelname.lower(), create_custom_log_method(level))

        return logger


if __name__ == "__main__":
    ## console output:
    # print(f"Colors you can choose: {list(COLOR_MAP.keys())}")
    # logger = PatchedLogging.getLogger(log_level=logging.DEBUG)
    # logger.info("This is an info message", color="Blue")
    # logger.info("This is an info message", color="green")
    # logger.warning("This is a warning message")
    # logger.error("This is an error message")
    # logger.debug("This is a debug message")

    # file output
    logger = PatchedLogging.getLogger(output_type="file")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")