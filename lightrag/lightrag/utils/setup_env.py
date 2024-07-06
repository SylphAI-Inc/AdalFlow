import dotenv


def setup_env(dotenv_path: str = ".env"):
    """Load environment variables from .env file."""

    dotenv.load_dotenv(dotenv_path=dotenv_path)
