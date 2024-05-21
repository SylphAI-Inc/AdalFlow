from .logger import BaseLogger

# Initialize and expose the logger instance
base_logger = BaseLogger().logger

# Expose the base_logger and BaseLogger for easier import
__all__ = ['base_logger', 'BaseLogger']
