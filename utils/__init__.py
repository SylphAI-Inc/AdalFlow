from .logger import BaseLogger

# Initialize and expose the logger instance
base_logger = BaseLogger().logger

# Optionally expose the logger setup function if users need to reconfigure or create additional loggers
__all__ = ['base_logger', 'BaseLogger']
