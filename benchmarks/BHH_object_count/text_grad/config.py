import textgrad as tg
from adalflow.utils import setup_env

setup_env()

gpt4o = tg.get_engine(engine_name="gpt-4o")
gpt_3_5 = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
