Logging System Instruction
==========================

What is a logging system?
-------------------------

A logging system is crucial for software development, providing advanced monitoring and debugging capabilities beyond simple print statements. It allows for customizable log levels, various output formats, and persistent storage, essential for managing complex text-in-text-out data flows in LLM applications. With its ability to record detailed activity logs efficiently, the logging system simplifies tracking and debugging, making it indispensable for developing and maintaining large language models.

Why use our logging system?
---------------------------

Python’s `default logging module <https://docs.python.org/3/library/logging.html>`_ is versatile but requires effort to set up across different application parts. Our standardized logging system, built on the logging module, aims to simplify logging by standardizing file paths, formats, and colors across the application. This alleviates the need for detailed configuration of handlers and formatters but also ensures consistency and simplicity, enabling developers to focus more on development than on logging infrastructure.

How does the logging system work?
---------------------------------

Our logging system, built on Python’s versatile logging module, introduces a ``BaseLogger`` that simplifies logging setup. This ``BaseLogger`` facilitates both console and file outputs, with configurable ``file path``, ``file name`` and ``log level`.  

Developers can import the ``BaseLogger`` into any script requiring debugging, configure it as needed, and start logging messages. Upon initialization, ``BaseLogger`` creates **one log file** at the specified location. All log messages are automatically appended to this file in a standardized format and are also displayed in the console.

Our logging system maintains the properties of the logging module:

1. **Root Logger vs. Named Logger**: The root logger oversees the overall logging configuration, while named loggers like our ``BaseLogger`` provide module-specific logging capabilities.
2. **Singleton Logger**: Only one instance of each logger name exists throughout the Python process, which keeps consistency and prevents redundant logger instances.
3. **Hierarchical Logging**: Our system enhances `Python's hierarchical logging <https://docs.python.org/3/library/logging.html>` by explicitly indicating the file name and line number in log entries, providing direct traces to the message origins and clear display of module names. 

**Output format:** 

``timestamp - logger name - log level - [filename: line number]- logging message``

.. note::
    While the ``BaseLogger`` acts as a named logger within the Python logging hierarchy, it respects the hierarchical nature of Python's logging. This means logs can be passed to higher-level loggers, including the root logger, unless explicitly configured not to do so. This capability ensures that ``BaseLogger`` can be used either in conjunction with a centrally configured root logger or on its own, accommodating a wide range of logging strategies to suit various development needs.


Examples
--------

Configurable arguments
^^^^^^^^^^^^^^^^^^^^^^

* ``directory (str, optional)``: Directory that stores the log file. Defaults to './logs'.
* ``filename (str, optional)``: Name of the log file. Defaults to 'app.log'.
* ``log_level (int, optional)``: Level of log to show. Defaults to `logging.INFO`. See `Python logging levels <https://docs.python.org/3/library/logging.html#logging-levels>`_.

Operations
^^^^^^^^^^

Demonstration of basic operations with the configured logger::

    from utils import BaseLogger
    import logging
    logger = BaseLogger(filename="sample.log", log_level=logging.DEBUG).logger
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

    # sample.log
    2024-05-20 21:50:42 - utils.logger - DEBUG - [test.py:6] - This is a debug message
    2024-05-20 21:50:42 - utils.logger - INFO - [test.py:7] - This is an info message
    2024-05-20 21:50:42 - utils.logger - WARNING - [test.py:8] - This is a warning message
    2024-05-20 21:50:42 - utils.logger - ERROR - [test.py:9] - This is an error message
    2024-05-20 21:50:42 - utils.logger - CRITICAL - [test.py:10] - This is a critical message

This sample illustrates how you can import the logger in separate scripts.

For Developer
^^^^^^^^^^^^^

Developers integrating new modules can utilize ``BaseLogger`` to streamline debugging. Here is a sample code:

.. code-block:: python

    # react_agent.py -- Example of integrating BaseLogger for module debugging
    from utils import BaseLogger
    import logging

    # Initialize the BaseLogger with INFO level for general application logging
    base_logger = BaseLogger(log_level=logging.INFO).logger

    class ReActAgent:
        ...
        def _run_one_step(self, input: str, step: int, ...):
            response = ...
            # Log each step's response for easy tracking and debugging
            base_logger.info(f"step {step}: {response}")


.. code-block:: python

    # hotpotqa.py -- Example of using the logging in an application
    from utils import BaseLogger

    # No need to set log level as INFO is default
    logger = BaseLogger().logger

    num_questions = 1
    for i in range(num_questions):
        question = ...
        gt_answer = ...
        # Log questions and answers for debugging
        logger.info(f'question: {question}')
        pred_answer = react_agent(question)
        logger.info(f'gt_answer: {gt_answer}; pred_answer: {pred_answer}')

    # app.log - my log
    2024-05-20 21:23:29 - utils.logger - INFO - [hotpotqa.py:81] - question: Were Scott Derrickson and Ed Wood of the same nationality?
    2024-05-20 21:23:30 - utils.logger - INFO - [react_agent.py:272] - step 1: {'thought': 'To determine the nationality of Scott Derrickson and Ed Wood, I need to search for their information on Wikipedia.', 'action': "search('Scott Derrickson')"}
    2024-05-20 21:23:31 - utils.logger - INFO - [react_agent.py:272] - step 2: {'thought': "I have the information about Scott Derrickson, now I need to search for Ed Wood's information on Wikipedia.", 'action': "search('Ed Wood')"}
    2024-05-20 21:23:57 - utils.logger - INFO - [react_agent.py:272] - step 3: {'thought': 'I have the information about Scott Derrickson and Ed Wood, now I need to determine if they were of the same nationality.', 'action': "lookup(search('Scott Derrickson'), 'American')"}
    2024-05-20 21:24:29 - utils.logger - INFO - [react_agent.py:272] - step 4: {'thought': 'I have the information about Ed Wood, now I need to determine if he was of the same nationality as Scott Derrickson.', 'action': "lookup(search('Ed Wood'), 'American')"}
    2024-05-20 21:25:05 - utils.logger - INFO - [react_agent.py:272] - step 5: {'thought': 'I have the information about Scott Derrickson and Ed Wood, now I need to determine if they were of the same nationality.', 'action': 'finish("Yes, both Scott Derrickson and Ed Wood were American.")'}
    2024-05-20 21:25:05 - utils.logger - INFO - [hotpotqa.py:90] - gt_answer: yes; pred_answer: Yes, both Scott Derrickson and Ed Wood were American.

.. note::

    As a developer, you can consider to add an optional logger argument in your component to make it convenient for the users to run your module with logged messages.

    For example, Pytorch lightning has `CSVLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html#lightning.pytorch.loggers.CSVLogger>`_, and users can configure a csvlogger and pass it to the Trainer. So that the users won’t need to spend time insert the loggers in multiple modules.

For Users
^^^^^^^^^

Users of LightRAG can also use the logging system as an assistant to debug applications efficiently.

Here is a sample code:

.. code-block:: python

    # hotpotqa.py -- User application utilizing BaseLogger for debugging
    from utils import BaseLogger

    # Initialize logger with default settings
    logger = BaseLogger().logger

    num_questions = 1
    for i in range(num_questions):
        question = dataset[i].get("question")
        gt_answer = dataset[i].get("answer")
        # Logging the question and the ground truth answer for reference
        logger.info(f'question: {question}')
        pred_answer = react_agent(question)
        res = evaluator.compute_match_acc_single_query(pred_answer=pred_answer, gt_answer=gt_answer)
        # Log both the predicted and actual answers for comparison
        logger.info(f'gt_answer: {gt_answer}; pred_answer: {pred_answer}')


.. code-block:: python

    #app.log
    2024-05-20 22:08:28 - utils.logger - INFO - [hotpotqa.py:81] - question: Were Scott Derrickson and Ed Wood of the same nationality?
    2024-05-20 22:09:20 - utils.logger - INFO - [hotpotqa.py:90] - gt_answer: yes; pred_answer: Both Scott Derrickson and Ed Wood were American.

This simple example illustrates the ``BaseLogger``'s function as a debugging assistant for LightRAG users. In your main application, you can import it and debug as you need.

Design Details & Improvement Directions
---------------------------------------

The ``BaseLogger`` serves as the core of our logging system, designed for both immediate use and future enhancements. We plan to introduce advanced loggers derived from ``BaseLogger``, enhancing functionality and facilitating integration with various applications. This approach ensures our logging framework remains adaptable and scalable.