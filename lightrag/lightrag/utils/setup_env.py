import dotenv
import os
import logging

log = logging.getLogger(__name__)


def setup_env(dotenv_path: str = ".env"):
    """Load environment variables from .env file."""

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f"File not found: {dotenv_path}")

    try:
        dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=True, override=False)
    except Exception as e:
        log.error(f"Error loading .env file: {e}")
        raise e
