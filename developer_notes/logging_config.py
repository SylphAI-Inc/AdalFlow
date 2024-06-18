from lightrag.utils.logger import enable_library_logging, get_logger
from lightrag.utils.file_io import load_json
from lightrag.core import Generator

log_dir = "developer_notes/logs"


config = load_json("developer_notes/default_config.json")
generator_config = config["generator"]


def check_console_logging():
    enable_library_logging(
        level="INFO",
        enable_console=True,
        enable_file=False,
    )
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    print(f"output console: {output}")


def check_file_logging():
    file_name = "lib1.log"
    enable_library_logging(
        level="DEBUG",
        enable_console=False,
        enable_file=True,
        save_dir=log_dir,
        filename=file_name,
    )

    generator = Generator.from_config(generator_config)
    generator(prompt_kwargs={"input_str": "how are you?"})


def get_logger_and_enable_library_logging_in_same_file():
    file_name = "lib_app_2.log"
    log = enable_library_logging(
        level="DEBUG",
        enable_console=False,
        enable_file=True,
        save_dir=log_dir,
        filename=file_name,
        name="test",
    )
    # log.info("app: test log message")
    generator = Generator.from_config(generator_config)
    output = generator(prompt_kwargs={"input_str": "how are you?"})
    log.info(f"output app: {output}")


def separate_library_logging_and_app_logging():
    lib_logfile = "lib3.log"
    app_logfile = "app.log"
    enable_library_logging(
        level="INFO",
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
    log.info(f"output: {output}")


if __name__ == "__main__":
    check_console_logging()
    check_file_logging()
    get_logger_and_enable_library_logging_in_same_file()
    separate_library_logging_and_app_logging()
