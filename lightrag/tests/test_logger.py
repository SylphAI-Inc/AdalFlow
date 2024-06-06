import pytest

import logging
from unittest.mock import MagicMock

from lightrag.utils import get_logger, printc


class TestGetDefaultLogger:
    def test_console_logging_only(self, mocker):
        mocker.patch("os.makedirs")
        mock_handler = mocker.patch(
            "logging.StreamHandler", return_value=MagicMock(spec=logging.StreamHandler)
        )

        logger = get_logger(
            name="test_console_logging_only", enable_file=False, enable_console=True
        )

        assert logger.hasHandlers()
        assert mock_handler.called
        mock_handler.return_value.setFormatter.assert_called()

    def test_file_logging_only(self, mocker):
        mocker.patch("os.makedirs")
        mock_file_handler = mocker.patch(
            "logging.FileHandler", return_value=MagicMock(spec=logging.FileHandler)
        )

        logger = get_logger(
            name="test_file_logging_only", enable_console=False, enable_file=True
        )

        assert logger.hasHandlers()
        assert mock_file_handler.called
        mock_file_handler.return_value.setFormatter.assert_called()

    def test_both_console_and_file_logging(self, mocker):
        mocker.patch("os.makedirs")
        mocker.patch(
            "logging.StreamHandler", return_value=MagicMock(spec=logging.StreamHandler)
        )
        mocker.patch(
            "logging.FileHandler", return_value=MagicMock(spec=logging.FileHandler)
        )

        logger = get_logger(name="test_both_console_and_file_logging")

        assert logger.hasHandlers()
        assert len(logger.handlers) == 2  # Both handlers should be added

    def test_no_logging(self, mocker):
        mocker.patch("os.makedirs")
        logger = get_logger(
            name="test_no_logging", enable_console=False, enable_file=False
        )

        assert not logger.hasHandlers()


class TestPrintc:
    def test_colored_print(self, mocker):
        mocker.patch("builtins.print")

        printc("hello world", color="green")

        assert print.called
        args, _ = print.call_args
        assert "\x1b[32m" in args[0]  # Check if green color code is in the print output

    def test_default_color(self, mocker):
        mocker.patch("builtins.print")

        printc("hello world")

        assert print.called
        args, _ = print.call_args
        assert (
            "\x1b[36m" in args[0]
        )  # Check if default cyan color code is in the print output
