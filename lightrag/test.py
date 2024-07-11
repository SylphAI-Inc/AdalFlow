from lightrag.core.types import StepOutput, FunctionExpression
from dataclasses import dataclass


@dataclass
class MyStepOutput(StepOutput[FunctionExpression]):
    pass


StepOutputClass = StepOutput.with_action_type(FunctionExpression)
schema = StepOutputClass.to_schema()

print(schema)

instance = StepOutput(
    step=0, action=FunctionExpression(thought="hello", action="print")
)

print(instance)
