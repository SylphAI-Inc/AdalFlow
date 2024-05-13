import logging
import os
import sys
from typing import Optional
import colorama
import structlog


# formatting console output, with colors in the console
# reference: https://www.structlog.org/en/stable/console-output.html
columns=[
    # Render the timestamp without the key name in yellow.
    structlog.dev.Column(
        "timestamp",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=colorama.Fore.YELLOW,
            reset_style=colorama.Style.RESET_ALL,
            value_repr=str,
        ),
    ),
    # Render the event without the key name in bright magenta.
    structlog.dev.Column(
        "event",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=colorama.Style.BRIGHT + colorama.Fore.MAGENTA,
            reset_style=colorama.Style.RESET_ALL,
            value_repr=str,
        ),
    ),
    # Default formatter for all keys not explicitly mentioned. The key is
    # cyan, the value is green.
    structlog.dev.Column(
        "",
        structlog.dev.KeyValueColumnFormatter(
            key_style=colorama.Fore.CYAN,
            value_style=colorama.Fore.GREEN,
            reset_style=colorama.Style.RESET_ALL,
            value_repr=str,
        ),
    ),
]


class LogSettings:
    """
    Configure the log levels, output type, method, filename
    """
    def __init__(self, output_type: str, method: str, file_name: Optional[str], log_level: str) -> None:
        self.output_type = output_type
        self.method = method
        self.file_name = file_name
        self.log_level = log_level

    def configure_structlog(self):
        """
        configure structlog and set format 
        """
        if self.output_type == "str":
            # renderer = structlog.dev.ConsoleRenderer()
            renderer = structlog.dev.ConsoleRenderer(columns=columns)
        else:
            renderer = structlog.processors.JSONRenderer()
            
        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.CallsiteParameterAdder(
                    {
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    },
                ),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                renderer,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger, 
        )
    
    def set_log_output(
        self,
        method: Optional[str] = None,
        file_name: Optional[str] = None,
        output_type: Optional[str] = None,
    ):
        """
            check the output settings and filter output by log level
        """
        if method is not None:
            if method not in ["console", "file"]:
                raise ValueError("method provided can only be 'console', 'file'")
            self.method = method
            if method == "file":
                if file_name is None:
                    raise ValueError("file_name must be provided when method = 'file'")
                self.file_name = file_name

        if output_type is not None:
            if output_type not in ["str", "json"]:
                raise ValueError("output_type provided can only be 'str', 'json'")
            self.output_type = output_type
            
        # Update Renderer
        self.configure_structlog()
        
        # Reset and configure the logger
        log = logging.getLogger()
        log.setLevel(self.log_level.upper())  # Set logger to the specified level
        for handler in log.handlers[:]:
            log.removeHandler(handler)
        
        # Add new Handler
        handler = logging.FileHandler(self.file_name) if self.method == "file" else logging.StreamHandler(sys.stdout)
        handler.setLevel(self.log_level.upper())  # Set handler level to the specified level
        log.addHandler(handler)



def setup_logging(output_type: str, method: str, file_name: Optional[str], log_level: str):
    """
        initialize the logging and configure
    """
    # # For http.client (used internally by several libraries)
    # logging.getLogger("http.client").setLevel(logging.WARNING)

    # # For urllib3, commonly used by the 'requests' library
    # logging.getLogger("urllib3").setLevel(logging.WARNING)

    # # For httpx
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    
    settings = LogSettings(output_type=output_type, method=method, file_name=file_name, log_level=log_level)
    settings.configure_structlog()
    settings.set_log_output()

def getLogger(name):
    """
        use structlog for better format, after setting up logging, the structlog is also configured
    """
    import structlog
    return structlog.get_logger(name)
