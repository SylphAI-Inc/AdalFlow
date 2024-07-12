from lightrag.components.agent import ReActAgent
from lightrag.core import Generator, ModelClientType, ModelClient
from lightrag.utils import setup_env

setup_env()


# Define tools
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


def add(a: int, b: int) -> int:
    """
    Add two numbers.
    """
    return a + b


def divide(a: float, b: float) -> float:
    """
    Divide two numbers.
    """
    return float(a) / b


llama3_model_kwargs = {
    "model": "llama3-70b-8192",  # llama3 70b works better than 8b here.
    "temperature": 0.0,
}
gpt_model_kwargs = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
}


def test_react_agent(model_client: ModelClient, model_kwargs: dict):
    tools = [multiply, add, divide]
    queries = [
        "What is the capital of France? and what is 465 times 321 then add 95297 and then divide by 13.2?",
        "Give me 5 words rhyming with cool, and make a 4-sentence poem using them",
    ]
    # define a generator without tools for comparison

    generator = Generator(
        model_client=model_client,
        model_kwargs=model_kwargs,
    )

    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=tools,
        model_client=model_client,
        model_kwargs=model_kwargs,
    )
    # print(react)

    for query in queries:
        print(f"Query: {query}")
        agent_response = react.call(query)
        llm_response = generator.call(prompt_kwargs={"input_str": query})
        print(f"Agent response: {agent_response}")
        print(f"LLM response: {llm_response}")
        print("")


def test_react_agent_use_examples(model_client: ModelClient, model_kwargs: dict):
    tools = [multiply, add, divide]
    queries = [
        "What is the capital of France? and what is 465 times 321 then add 95297 and then divide by 13.2?",
        "Give me 5 words rhyming with cool, and make a 4-sentence poem using them",
    ]

    from lightrag.core.types import FunctionExpression

    # add examples for the output format str
    example_using_multiply = FunctionExpression.from_function(
        func=multiply,
        thought="Now, let's multiply two numbers.",
        a=3,
        b=4,
    )
    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=tools,
        model_client=model_client,
        model_kwargs=model_kwargs,
        examples=[example_using_multiply],
    )

    print(react)

    # see the output format
    react.planner.print_prompt()

    for query in queries:
        print(f"Query: {query}")
        agent_response = react.call(query)
        print(f"Agent response: {agent_response}")
        print("")


if __name__ == "__main__":
    test_react_agent(ModelClientType.GROQ(), llama3_model_kwargs)
    test_react_agent(ModelClientType.OPENAI(), gpt_model_kwargs)
    print("Done")

    test_react_agent_use_examples(ModelClientType.GROQ(), llama3_model_kwargs)
