# This doc shows how to use all different providers in the Generator class.

import adalflow as adal
from adalflow.utils.logger import get_logger

log = get_logger(enable_file=False)


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


if __name__ == "__main__":
    adal.setup_env()
    # test_non_reasoning_model()
    test_reasoning_model()
