from utils.logger import PatchedLogging
import sys

logger = PatchedLogging.getLogger(log_level="DEBUG")

logger.info("This is an info message", color="Blue")

# import logging

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# defaul_logger = logging.getLogger(__name__)
# # defaul_logger.setLevel(logging.INFO)
# defaul_logger.info("This is an info message")

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# Create a logger object
logger = logging.getLogger("example_logger")
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")

# Set levels for handlers
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
# # Setup basic configuration for logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
c_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
)
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Log messages
logger.warning("This is a warning")
logger.error("This is an error")
