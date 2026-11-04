from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient

import asyncio

from dotenv import load_dotenv

load_dotenv()

# class MathHomeworkOutput(BaseModel):
#     is_math_homework: bool
#     reasoning: str


class MathHomeworkOutput:
    is_math_homework: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
    model_client=OpenAIClient(api_key="fake_api_key"),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
)

# Commented out until guardrail types are properly imported
# @input_guardrail
# async def math_guardrail(
#     ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
# ) -> GuardrailFunctionOutput:
#     result = await Runner.run(guardrail_agent, input, context=ctx.context)
#
#     return GuardrailFunctionOutput(
#         output_info=result.final_output,
#         tripwire_triggered=result.final_output.is_math_homework,
#     )


agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    model_client=OpenAIClient(api_key="fake_api_key"),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    # input_guardrails=[math_guardrail],  # Commented out until guardrails are implemented
)


async def main():
    # Basic test without guardrails
    response = await Runner.run(
        agent, "Hello, can you help me solve for x: 2x + 3 = 11?"
    )
    print("Agent response:")
    print(response)
    print(response.final_output)
    print(type(response.final_output))


if __name__ == "__main__":
    asyncio.run(main())
