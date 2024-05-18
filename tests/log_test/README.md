## **Logging Configuration Guide**

The logging system leverages `structlog` for structured logging and Python's built-in `logging` module for traditional log management. Structured logging makes it easier to read, process, and analyze log files, particularly in complex systems with multiple components.

The output format of each message is structured as:

`timestamp - logger name - log level - [file path] - message`

### Setup

1. Install structlog:
    
    ```python
    pip install structlog
    ```
    
2. Import and configure the logging settings in your main application script or a dedicated configuration module:
    
    ```python
    from utils.logging_config import LogSettings
    
    # Configure the logger
    log_settings = LogSettings(
        output_type="str",  # 'str' for string output or 'json' for JSON output
        method="file",      # 'console' for console output or 'file' for file logging
        file_name="path/to/logfile.log",  # Configure the output path if method is 'file'
        log_level="INFO"    # Default logging level
    )
    logger = log_settings.get_logger()
    ```
    
    **Configurable Parameters**
    
    - output_type: `'str'` for plain text, `'json'` for JSON structured output.
    - method: `'console'` for console output, `'file'` for logging to a file.
    - file_name: Path to the log file (required if `method` is `'file'`).
    - log_level: Sets the logging level (`'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'CRITICAL'`). log level reference: https://docs.python.org/3/library/logging.html#logging-levels
3. Once configured, you can use the logger across your application to log various levels of messages:
    
    ```python
    logger.info("This is an info level log message.")
    logger.error("This is an error level log message.")
    
    response = LLM_function()
    logger.info(f'response: {response}')
    ```
    

### **Examples**

Basic Console Logging

```python

from utils.logging_config import LogSettings

log_settings = LogSettings(method="console", log_level="DEBUG")
logger = log_settings.get_logger()
logger.debug("Debug message for troubleshooting.")
```

File Logging

```python
from utils.logging_config import LogSettings

log_settings = LogSettings(method="file", file_name="./logs/app.log", log_level="INFO")
logger = log_settings.get_logger()
logger.info("Application started.")
```

Component output Logging

```python
from utils.logging_config import LogSettings

# configure the logging once and use it
log_settings = LogSettings(file_name="./tests/log_test/app.log")
logger = log_settings.get_logger()  # Retrieve the configured logger

# define your application
simple_qa = SimpleQA() # this is the application you write
response = simple_qa.call("What is the capital of France?")
logger.info(f'response: {response}')

# output in the file: 
# 2024-05-17 21:38:43 - AppLog - INFO - [.../simple_qa_logging_test.py:38] - response: The capital of France is Paris.
```