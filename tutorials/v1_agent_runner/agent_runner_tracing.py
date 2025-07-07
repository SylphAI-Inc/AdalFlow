# import asyncio
# from agents import Agent, Runner, trace, add_trace_processor
# from agents.tracing.processors import ConsoleSpanExporter
# from adalflow.utils import setup_env

# setup_env()

# # Tell the tracer to also print spans to the console
# add_trace_processor(ConsoleSpanExporter())  # :contentReference[oaicite:0]{index=0}

# async def main():
#     agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

#     with trace("Joke workflow"):
#         first_result = await Runner.run(agent, "Tell me a joke")
#         second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
#         print(f"Joke: {first_result.final_output}")
#         print(f"Rating: {second_result.final_output}")

# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio

from agents import Agent, Runner, function_tool
import mlflow
from adalflow.utils import setup_env

setup_env()

# Enable automatic tracing for all OpenAI API calls
mlflow.openai.autolog()

mlflow.set_tracking_uri("http://localhost:5001")

# Set up MLflow experiment

mlflow.set_experiment("OpenAI Agent")


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


# If you are running this code in a Jupyter notebook, replace this with `await main()`.
if __name__ == "__main__":
    asyncio.run(main())
