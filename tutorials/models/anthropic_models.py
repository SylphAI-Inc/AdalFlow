# This doc shows how to use all different providers in the Generator class.

import adalflow as adal
from adalflow.utils.logger import get_logger

log = get_logger(enable_file=False, level="DEBUG")

thinking_model_kwargs = {
    "model": "claude-sonnet-4-20250514",
    "thinking": {"type": "enabled", "budget_tokens": 10000},
    # this will enable interleaved thinking
    "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
}


def test_non_reasoning_model():
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={"model": "claude-3-5-sonnet-20241022"},
    )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}

    anthropic_response = anthropic_llm(prompt_kwargs)

    print(f"Anthropic: {anthropic_response}\n")


def test_reasoning_model():
    # starts from claude-4
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={
            "model": "claude-sonnet-4-20250514",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            # this does not make a difference as there is no tool calls
            "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
        },
    )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}

    anthropic_response = anthropic_llm(prompt_kwargs)

    print(f"Anthropic: {anthropic_response}\n")


def test_reasoning_model_with_tool_calls():
    r"""
    Reference:https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#tool-use-with-interleaved-thinking
    """
    # this demonstrates the basic anthropic interleaved thinking with tool calls
    # if using adal.FunctionTool, the code would be simpler
    # starts from claude-4
    # Same tool definitions as before
    calculator_tool = {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }

    database_tool = {
        "name": "database_query",
        "description": "Query product database",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"}
            },
            "required": ["query"],
        },
    }
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={
            "model": "claude-sonnet-4-20250514",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            # this will enable interleaved thinking
            "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
            "tools": [calculator_tool, database_tool],
        },
    )

    prompt_kwargs = {
        "input_str": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }

    # 1. first response
    anthropic_response = anthropic_llm(prompt_kwargs)

    print(f"Anthropic: {anthropic_response}\n")
    calculator_response = "7500"  # the answer to the calculator tool call

    # use chat_history_str to store the assistant's response
    step_str = f"Assistant: Thinking: {anthropic_response.thinking}\n"
    step_str += f"Content: {anthropic_response.data}\n"
    step_str += f"Tool Use: {anthropic_response.tool_use}\n"
    step_str += f"Tool Result: {calculator_response}\n"

    # add the Function and the Function result to the prompt kwargs
    # put this in steps_str as this is agentic with multiple loop
    steps_str = "step: 1\n"
    steps_str += step_str

    # 2. second response
    prompt_kwargs = {
        "input_str": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?",
        "steps_str": steps_str,
    }
    anthropic_response = anthropic_llm(prompt_kwargs)
    print(f"Anthropic: {anthropic_response}\n")


def test_reasoning_model_with_tool_calls_using_react():

    from adalflow.components.agent.react import ReActAgent

    def calculator(expression: str, **kwargs) -> str:
        """Perform mathematical calculations"""
        return "7500"

    def database_query(query: str, **kwargs) -> str:
        """Query product database"""
        return "5200"

    agent = ReActAgent(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs=thinking_model_kwargs,
        tools=[calculator, database_query],
        use_cache=False,
        is_thinking_model=True,
    )

    prompt_kwargs = {
        "input_str": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }

    agent_response = agent(prompt_kwargs)
    print(f"Agent: {agent_response}\n")


if __name__ == "__main__":
    adal.setup_env()
    # test_non_reasoning_model()
    # test_reasoning_model()
    # test_reasoning_model_with_tool_calls()
    # test_reasoning_model_with_tool_calls_using_react()
