from openai import OpenAI
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
client = OpenAI()


class Step2(BaseModel):
    explanation: str
    output: str


class Step(BaseModel):
    explanation: str
    output: str
    step_number: Step2


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


# Example usage
# response = client.responses.parse(
#     model="gpt-4o-2024-08-06",
#     input=[
#         {
#             "role": "system",
#             "content": "You are an advanced math tutor. Solve the equation step by step, providing detailed operations and validations.",
#         },
#         {"role": "user", "content": "Solve the equation: 2(3x - 5) + 4 = 3x + 7"},
#     ],
#     text_format=MathReasoning,
# )


response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input=[
        {
            "role": "system",
            "content": "You are an advanced math tutor. Solve the equation step by step, providing detailed operations and validations.",
        },
        {"role": "user", "content": "Solve the equation: 2(3x - 5) + 4 = 3x + 7"},
    ],
)

print(response)
