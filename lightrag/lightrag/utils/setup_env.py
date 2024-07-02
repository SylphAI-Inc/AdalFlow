import dotenv


def setup_env():
    dotenv.load_dotenv(dotenv_path=".env", override=True)
