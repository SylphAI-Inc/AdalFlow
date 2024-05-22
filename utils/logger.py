import logging
import os

# # Get or create a logger at the module level
# logger = logging.getLogger(__name__)

class BaseLogger:
    r"""
    Base class for loggers.
    Logger setup that configures file and console handlers, ensuring no duplicates.
    
    Example:
    from utils import BaseLogger
    import logging
    base_logger = BaseLogger(filename='myapp.log', log_level=logging.DEBUG).logger
    base_logger.debug('This is a debug message')
    
    Args:
        directory (str, optional): It is the directory that stores the log file. Defaults to './logs'.
        filename (str, optional): It is the name of the log file. Defaults to 'app.log'.
        log_level (int, optional): It is the level of log to show. Defaults to logging.INFO.
        Reference: https://docs.python.org/3/library/logging.html#levels
    """
    def __init__(self, directory: str = './logs', filename: str = 'app.log', log_level: int = logging.INFO):
        """
        Initialize the logger, the log file directory, file name, and level of the log.
        """
        # self.logger = logger
        # use unique logger name for each instance
        unique_logger_name = f"{__name__}_{filename.replace('.', '_')}"
        self.logger = logging.getLogger(unique_logger_name)
        self.file_path = os.path.join(directory, filename)
        self.log_level = log_level
        self.logger.setLevel(self.log_level)  # Set the default logging level

        # Ensure the log directory exists
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Failed to create log directory {directory}: {e}")
            raise

        # Configure handlers of the initialized logger only if they haven't been added yet
        if not self.logger.handlers:
            self.configure_handlers()

    def configure_handlers(self):
        """Configure file and console handlers for the logger."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        try:
            # # File handler, defines the behavior of the log file
            # file_handler = logging.FileHandler(self.file_path)
            # file_handler.setFormatter(formatter)
            # self.logger.addHandler(file_handler)

            # If there are no such handlers, it then proceeds to create a new FileHandler
            if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
                file_handler = logging.FileHandler(self.file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
            # Ensure only one console handler exists
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        except Exception as e:
            print(f"Failed to set up logging handlers: {e}")
            raise


# Usage Example
if __name__ == "__main__":
    # base_logger = BaseLogger().logger
    # base_logger.info("This is a test log message.")
    
    base_logger = BaseLogger(log_level=logging.DEBUG).logger
    base_logger.debug('This is a debug message')
    base_logger.info('This is an info message')
    base_logger.warning('This is a warning message')
    base_logger.error('This is an error message')
    base_logger.critical('This is a critical message')
