"""
Colorize console logs to separate logs from prints.
Windows users should set up colorama for colored console output.
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
    
    # def __init__(self) -> None:
    #     pass
    
    @staticmethod
    def getLogger(output_type:str = "console", dir:str = "./logs", filename:Union[str, None] = None, log_level:int = logging.INFO, format: Union[str, None] = None) -> logging.Logger:
        """Configure logger in one method

        Args:
            output_type (str, optional): output type, "console" or "file". Defaults to "console".
            dir (str, optional): directory for the log file. Defaults to "./logs".
            filename (str, optional): log file name. Defaults to "app.log".
            log_level (int, optional): log_level, refer to: https://docs.python.org/3/library/logging.html#levels. Defaults to logging.INFO.
            format (Union[str | None], optional): the format of each log message. 

        Returns:
            logging.Logger: the configured logger
        """
        # set up logger with log level
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)  # default log level is INFO
        
        # Define default log format if none provided
        if not format:
            format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        # set up format
        formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
        
        # if the user input a file name, then switch to file output
        if filename is not None:
            output_type = "file"
        else:
            filename = "app.log"
        
        try:
            # if handler not existing then set new
            if output_type == "console" and not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
                handler = logging.StreamHandler(sys.stdout)
                
            # if output in the console, then use stream handler, else check the file path(create if not exists) and use the file handler
            elif output_type == "file" and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.join(dir, filename) for h in logger.handlers):
                try:
                    if not os.path.exists(dir):
                        os.makedirs(dir, exist_ok=True)
                except:
                    raise OSError(f"Directory {dir} can't be created")
                file_path = os.path.join(dir, filename)
                handler = logging.FileHandler(file_path)

        except Exception as e:
            print(f"Failed to set up logging handlers: {e}")
            raise
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent logger from propagating messages to higher-level loggers -> prevent duplication
        logger.propagate = False

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
    # console output:
    # print(f"Colors you can choose: {list(COLOR_MAP.keys())}")
    logger = PatchedLogging.getLogger(log_level=logging.DEBUG)
    logger.info("This is an info message")
    logger.info("This is an info message", color="Blue")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")

    # # file output
    # logger = PatchedLogging.getLogger(output_type="file")
    # logger.info("This is an info message")
    # logger.warning("This is a warning message")
    # logger.error("This is an error message")
    # logger.debug("This is a debug message")