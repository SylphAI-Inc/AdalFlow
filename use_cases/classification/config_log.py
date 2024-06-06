import os

from lightrag.utils import enable_library_logging, get_logger

from use_cases.classification.utils import get_script_dir


# Enable library logging in logs/library.log
# only save the logs in the file
enable_library_logging(
    save_dir=os.path.join(get_script_dir(), "logs"),
    level="DEBUG",
    enable_file=True,
    enable_console=False,
    filename="library.log",
)

# get the app logger and enable both console and file logging
log = get_logger(
    __name__,
    level="INFO",
    save_dir=os.path.join(get_script_dir(), "logs"),
    filename="app.log",
)
