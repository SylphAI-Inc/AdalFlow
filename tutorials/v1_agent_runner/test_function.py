from adalflow.core import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.utils import setup_env, get_logger
import asyncio
from typing import Generator


setup_env()
logger = get_logger(level="DEBUG", enable_file=False)


def how_many_jokes() -> str:
    return "silly"


async def dummy_tool() -> str:
    return "dummy"


async def dummy_generator() -> Generator[str, None, None]:
    yield "dummy generator"


agent = Agent(
    name="TestAgent",
    model_client=OpenAIClient(),
    model_kwargs={"model": "o3"},
    tools=[how_many_jokes, dummy_tool, dummy_generator],
    answer_data_type=str,
)

runner = Runner(agent=agent, max_steps=12)


async def main():
    joke_query = "call dummy generator and tell me the result of them all"
    cmd_query = "ls -la"
    result = await runner.acall(prompt_kwargs={"input_str": joke_query})
    print(result)


def main_sync():
    joke_query = "call the how_many_jokes tool and tell me the result"
    cmd_query = "ls -la"
    joke_query = "call dummy generator and tell me the result of them all"

    result = runner.call(prompt_kwargs={"input_str": joke_query})
    print(result)


if __name__ == "__main__":
    # asyncio.run(main())
    main_sync()
