from .logger import BaseLogger

# Initialize and expose the logger instance
default_logger = BaseLogger().logger

# Expose the base_logger and BaseLogger for easier import
__all__ = ['default_logger', 'BaseLogger']
