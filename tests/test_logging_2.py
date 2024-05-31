# from utils.logger import PatchedLogging

# # Example usage
# if __name__ == "__main__":
#     logger = PatchedLogging.get_logger(log_level="DEBUG")
#     logger.info("This is an info message", color="green")

import logging
import inspect


class CustomLogger(logging.Logger):
    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        """
        Create a log record, allowing overriding of the filename, line number, and function name.
        """
        rv = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        if extra:
            for key, value in extra.items():
                if hasattr(rv, key):
                    setattr(rv, key, value)
        return rv


# Set the custom logger class in the logging system
logging.setLoggerClass(CustomLogger)
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("This is an info message", extra={"color": "green"})
