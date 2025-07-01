from adalflow.core import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from tools.ask_for_user.examples.permission_tool import bash_command
from adalflow.utils import setup_env
import asyncio

setup_env()


def how_many_jokes() -> str:
    return "silly"


async def dummy_tool() -> str:
    return "dummy"


agent = Agent(
    name="TestAgent",
    model_client=OpenAIClient(),
    model_kwargs={"model": "o3"},
    tools=[bash_command, how_many_jokes, dummy_tool],
    answer_data_type=str,
)

runner = Runner(agent=agent, max_steps=12)


async def main():
    joke_query = "call the how_many_jokes and dummy tool and tell me the result"
    result = await runner.acall(prompt_kwargs={"input_str": joke_query})
    print(result)


def main_sync():
    joke_query = "call the how_many_jokes tool and tell me the result"
    result = runner.call(prompt_kwargs={"input_str": joke_query})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
    main_sync()
