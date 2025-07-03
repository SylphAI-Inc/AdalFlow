from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.agent import input_guardrail
from adalflow.core.types import TResponseInputItem
from adalflow.core.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
)

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
)


@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )


agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[math_guardrail],
)


async def main():
    # This should trip the guardrail
    try:
        response = await Runner.run(
            guardrail_agent, "Hello, can you help me solve for x: 2x + 3 = 11?"
        )
        print("Guardrail didn't trip - this is unexpected")
        print(response)
        print(response.final_output)
        print(type(response.final_output))

    except InputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped")


if __name__ == "__main__":
    asyncio.run(main())
