import os
from adalflow.utils.logger import get_logger, printc
from adalflow.utils.file_io import load_json
from adalflow.core import Generator
from adalflow.utils import setup_env


def prepare_paths():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = "default_config.json"
    config_path = os.path.join(script_dir, config_path)
    log_dir = os.path.join(script_dir, "logs")
    env_path = os.path.join(script_dir, ".env")
    return config_path, log_dir, env_path


config_path, log_dir, env_path = prepare_paths()

setup_env(dotenv_path=env_path)

config = load_json(config_path)
generator_config = config["generator"]


def check_console_logging():
    get_logger(
        level="INFO",
        enable_console=True,
        enable_file=False,
        save_dir=log_dir,
    )
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    print(f"output console: {output}")


def get_logger_and_enable_library_logging_in_same_file():
    file_name = "lib_app_mix.log"
    root_logger = get_logger(
        level="INFO",
        enable_console=False,
        enable_file=True,
        save_dir=log_dir,
        filename=file_name,
    )
    # log.info("app: test log message")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    root_logger.info(f"logging from the root logger: {output}")


def separate_library_logging_and_app_logging():
    lib_logfile = "lib_seperate.log"
    app_logfile = "app_seperate.log"
    get_logger(
        level="DEBUG",
        enable_console=False,
        enable_file=True,
        save_dir=log_dir,
        filename=lib_logfile,
    )
    log = get_logger(
        level="INFO",
        save_dir=log_dir,
        filename=app_logfile,
        enable_console=True,
        enable_file=True,
        name=__name__,
    )
    log.info("test log message")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    log.info(f"output using child logger {__name__}: {output}")


def use_only_child_logger():
    file_name = "app.log"
    child_logger = get_logger(
        level="INFO",
        save_dir=log_dir,
        filename=file_name,
        enable_console=True,
        enable_file=True,
        name=__name__,
    )
    child_logger.info("test log message")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    child_logger.info(f"output using app logger {__name__}: {output}")


def user_program_adalflow_config():
    from adalflow.utils.logger import get_logger

    log = get_logger(
        name=__name__,
        level="INFO",
        enable_console=True,
        enable_file=True,
        filename="app.log",
    )
    log.info("This is a user program child logger in app.log")


def user_program():
    import logging

    log = logging.getLogger(__name__)
    log.info("This is a user program child logger")


def use_native_root_logging():
    # usually the main program will use the root logger to gather all logs

    # use it with root logger
    root_logger = get_logger(
        level="INFO",
        enable_console=True,
        enable_file=True,
    )

    # call user program
    user_program()

    root_logger.info("test root logger")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    root_logger.info(f"output using root logger: {output}")


def use_named_logger():
    # use it with named logger
    named_logger = get_logger(
        name="app1",
        level="INFO",
        enable_console=True,
        enable_file=True,
    )

    user_program()
    # will only include logs here
    named_logger.info("test named logger")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    named_logger.info(f"output using named logger {__name__}: {output}")


def use_adalflow_logger():
    # set up root logger for the library
    # get_logger(
    #     level="INFO",
    #     enable_file=True,
    #     filename="app.log",
    # )
    # # use the library
    # generator = Generator.from_config(generator_config)
    # output = generator(prompt_kwargs={"input_str": "how are you?"})
    # use a logger for the user program
    user_program_adalflow_config()


if __name__ == "__main__":
    # check_console_logging()
    # get_logger_and_enable_library_logging_in_same_file()
    # separate_library_logging_and_app_logging()
    # use_only_child_logger()
    # use_native_root_logging()
    # use_named_logger()
    use_adalflow_logger()
    printc("All logging examples are done. Feeling green!", color="green")
