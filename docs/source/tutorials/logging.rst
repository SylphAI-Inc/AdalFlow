Logging
====================


The Python logging module [1]_ is a powerful and flexible tool for debugging and tracing.
LightRAG uses the native logging module as the *first line of defense*.

Design
--------------------
Some libraries may use hooks [2]_ and callbacks [3]_ [4]_, or advanced web-based debugging tools [5]_ [6]_ [7]_.
Hooks and callbacks are conceptually similar in that they both allow users to execute custom code at specific points during the execution of a program.
Both provide mechanisms to inject additional behavior in response to certain events or conditions, without modifying the core logic.
PyTorch defines, registers, and executes hooks mainly in its base classes like `nn.Module` and `Tensor`, without polluting the functional and user-facing APIs.

At this point, our objectives are:

1. Maximize debugging capabilities via the simple logging module to keep the source code clean.
2. Additionally, as we can't always control the outputs of generators, we will provide customized logger and tracers(drop-in decorators) for them, for which we will explain in :doc:`logging_tracing`. This will not break the first objective.

In the future, when we have more complex requirements from users, we will consider adding hooks/callbacks but we will do it in a way to keep the functional and user-facing APIs clean.

How the library logs
~~~~~~~~~~~~~~~~~~~~~~
In each file, we simply set the logger with the following code:

.. code-block:: python

    import logging

    log = logging.getLogger(__name__)

We then use `log` and decide what level of logging we want to use in each function.
Here is how :ref:`Generator <core.generator>` logs.

How users set up the logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a simple function :func:`get_logger<utils.logger.get_logger>` to help users set up loggers with great default formatting and handlers.
The simplest way to get up a logger and see the library logs is to call this function:

.. code-block:: python

    from lightrag.utils.logger import get_logger


    root_logger = get_logger()

Here is an example of the printed log message if users run the generator after setting up the logger:

.. code-block::

    2024-07-05 18:49:39 - generator - INFO - [generator.py:249:call] - output: GeneratorOutput(data="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?", error=None, usage=None, raw_response="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?", metadata=None)


printc function
~~~~~~~~~~~~~~~~~~~~~~
A neat tiny function :func:`printc<utils.logger.printc>` is provided to print colored text to the console.
It has similar formatting to the logger.
Here is an example of how to use it:

.. code-block:: python

    from lightrag.utils.logger import printc

    printc("All logging examples are done. Feeling green!", color="green")

The output will be:

.. raw:: html

    <pre style="color: green; background-color: black;">
    2024-07-05 22:25:43 - [logging_config.py:98:&lt;module&gt;] - All logging examples are done. Feeling green!
    </pre>



.. .. list-table:: Parameters of :func:`get_logger<utils.logger.get_logger>`
..    :header-rows: 1
..    :widths: 30 70

..    * - **Argument**
..      - **Description**
..    * - ``name: Optional[str] = None``
..      - Name of the logger. If None, the root logger is used.
..    * - ``level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"``
..      - Log level for the logger. Can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
..    * - ``save_dir: Optional[str] = None``
..      - Directory where the log files will be saved. Defaults to "./logs".
..    * - ``filename: Optional[str] = None``
..      - Name of the log file. Defaults to "lib.log" for the root logger and "{name}.log" for named logger.
..    * - ``enable_console: bool = True``
..      - Whether to enable logging to console. Defaults to True.
..    * - ``enable_file: bool = False``
..      - Whether to enable logging to a file. Defaults to False.

Use Logger in Projects
-------------------------
There are two distinct ways to set up the logging in your project:

1. Have both the library loggind and your application logging in a single file. This is the simplest setup.
2. Use both root and named logger to log library and application logs separately.

Set up all logs in one file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assume your source code is at `src/task.py`. You can log simply by:

.. code-block:: python

    import logging

    log = logging.getLogger(__name__)

    class Task:
        def __init__(self):
            log.info("This is a user program child logger")

In the main file, you can config a single root logger to log both library and application logs:

.. code-block:: python

    import logging
    from lightrag.utils.logger import get_logger

    root_logger = get_logger(level="DEBUG", save_dir="./logs") # log to ./logs/lib.log

    # run code from the library components such as generator
    # ....

    root_logger.info("This is the log in the main file")

This way, all logs will be saved in `./logs/lib.log`.

Separate library and application logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, if users prefer to separate the library and application logs, they can use a named logger.

In the user program, such as at `src/task.py`, you can set up a named logger and logs to `./logs/my_app.log`:

.. code-block:: python

    from lightrag.utils.logger import get_logger

    app_logger = get_logger(name="my_app", level="DEBUG", save_dir="./logs") # log to ./logs/my_app.log

    class Task:
        def __init__(self):
            app_logger.info("This is a user program child logger")

The difference is that you have already attached handlers to the app_logger.
In the main file, you do not need to set up a root logger to enable your application logs.
However, you can still set up a root logger to log the library logs separately if needed, and create another named logger to continue logging in the main file.



.. It works similarly if it is to be logged to console.
.. Config 3 can be quite neat:

.. - You can enable different levels of logging for the library and your application.
.. - You can easily focus on debugging your own code without being distracted by the library logs and still have the option to see the library logs if needed.

.. .. Create a named logger
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. code-block:: python

..     from lightrag.utils.logger import get_logger

..     app_logger = get_logger(name="my_app", level="DEBUG", save_dir="./logs") # log to ./logs/my_app.log
..     # or
..     logger = get_logger(name=__name__, level="DEBUG", save_dir="./logs", filename="my_app.log")

..     app_logger.debug("This is a debug message")
..     app_logger.info("This is an info message")
..     app_logger.warning("This is a warning message")
..     app_logger.error("This is an error message")
..     app_logger.critical("This is a critical message")


.. admonition:: References
   :class: highlight

   .. [1] Python logging module: https://docs.python.org/3/library/logging.html
   .. [2] Hooks in PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
   .. [3] Callbacks in Transformers: https://huggingface.co/docs/transformers/main/en/main_classes/callback
   .. [4] Callbacks in PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/callbacks.html
   .. [5] Weights & Biases: https://wandb.ai/site
   .. [6] TensorBoard: https://www.tensorflow.org/tensorboard
   .. [7] Arize Phoenix: https://docs.arize.com/phoenix



.. admonition:: API References
   :class: highlight

   - :func:`utils.logger.get_logger`
   - :func:`utils.logger.printc`
   - :ref:`Generator<core.generator>`
