import asyncio
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field
from openai import OpenAI
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

from openai.types.responses import ResponseTextDeltaEvent
from agents import AgentOutputSchema

import json

load_dotenv()

OpenAI.log = "debug"  # or: openai.debug = True


# 1) Your structured output model
class ActionOut(BaseModel):
    action: str = Field(..., description="The action the agent chose")
    timestamp: str = Field(..., description="UTC time when the action was taken")


# 2) A small function‐tool for getting the time
@function_tool
async def get_time() -> str:
    return datetime.utcnow().isoformat()


# 3) (Optional) your DI‐style context
@dataclass
class MyContext:
    user_id: str
    is_premium: bool


# 4) Instantiate your OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 5) Build the Agent correctly
agent = Agent[MyContext](
    name="Action Agent",
    instructions=(
        "Decide on exactly one action and return JSON matching the ActionOut schema."
    ),
    model="gpt-4o-mini",
    tools=[get_time],
    output_type=ActionOut,  # <-- not output_schema
    # text_format = ActionOut
)


async def simple_nested_structure():
    ctx = MyContext(user_id="abc123", is_premium=True)

    # —— Streaming run ——
    stream_res = Runner.run_streamed(agent, "Please choose an action.", context=ctx)
    iterations = 0
    final_response = ""
    async for responses in stream_res.stream_events():
        # you'll receive RawResponsesStreamEvent, RunItemStreamEvent, etc.
        print(responses)
        # if responses.type == "raw_response_event" and isinstance(responses.data, ResponseTextDeltaEvent):
        #     iterations += 1
        #     print(responses.data.delta)
        #     final_response += str(responses.data.delta)

    print(f"\nTotal iterations: {iterations}")
    print("=" * 80)
    print("Final response:")
    print(final_response)
    print("=" * 80)


async def complicated_structure():
    class StepDetail(BaseModel):
        step_number: int
        action_taken: str
        observation: str
        tool_call: Optional[Dict[str, Any]] = None

    class Summary(BaseModel):
        step_count: int
        final_output: Any
        steps: List[StepDetail]
        tool_calls: List[Dict[str, Any]]

    class Metrics(BaseModel):
        time_taken_seconds: float
        tokens_used: int

    class PersonSummary(BaseModel):
        summary: Summary
        metrics: Metrics

    ctx = MyContext(user_id="abc123", is_premium=True)

    # Create a more complex agent with structured output
    structured_agent = Agent[MyContext](
        name="Structured Output Agent",
        instructions=(
            "You are an advanced AI that provides detailed, structured responses. "
            "Format your response as a JSON object with 'summary' and 'metrics' fields."
        ),
        model="gpt-4o-mini",
        tools=[get_time],
        output_type=AgentOutputSchema(PersonSummary, strict_json_schema=False),
    )

    # # Run the agent with streaming
    stream_res = Runner.run_streamed(
        structured_agent,
        "Provide a detailed analysis with multiple steps and metrics.",
        context=ctx,
    )

    # # Process the stream
    iterations = 0
    full_response = ""
    async for response in stream_res.stream_events():
        if response.type == "raw_response_event" and isinstance(
            response.data, ResponseTextDeltaEvent
        ):
            print(vars(response))
            # print(response)
            full_response += str(response.data.delta)

    print("=" * 80)
    print("Total iterations:", iterations)
    print("=" * 80)
    print("Final response:")
    print(full_response)
    print("=" * 80)

    # Parse and structure the response
    try:
        # full_response = json.loads("""
        # {"summary":{"step_count":4,"steps":[{"step_number":1,"action_taken":"Identified the main objectives for the analysis.","observation":"The objectives include evaluating performance metrics and user engagement.","tool_call":null},{"step_number":2,"action_taken":"Gathered relevant data from available sources.","observation":"Data sources included internal databases and user feedback surveys.","tool_call":null},{"step_number":3,"action_taken":"Analyzed the gathered data for trends and patterns.","observation":"Found significant trends in user engagement over the past year, indicating seasonality.","tool_call":null},{"step_number":4,"action_taken":"Formulated recommendations based on analysis.","observation":"Recommendations include targeted marketing campaigns during peak engagement times.","tool_call":null}],"final_output":"The analysis identifies key trends in user engagement, suggesting strategic interventions."},"metrics":{"time_taken_seconds":150,"tokens_used":200}}
        # """)
        response_data = eval(
            full_response
        )  # In production, use json.loads() with proper error handling
        # response_data = full_response

        # Create structured output
        steps = [
            StepDetail(
                step_number=i + 1,
                action_taken=step.get("action", ""),
                observation=step.get("observation", ""),
                tool_call=step.get("tool_call"),
            )
            for i, step in enumerate(response_data.get("steps", []))
        ]

        summary = Summary(
            step_count=len(steps),
            final_output=response_data.get("final_output", ""),
            steps=steps,
            tool_calls=response_data.get("tool_calls", []),
        )

        metrics = Metrics(
            time_taken_seconds=response_data.get("metrics", {}).get(
                "time_taken_seconds", 0.0
            ),
            tokens_used=response_data.get("metrics", {}).get("tokens_used", 0),
        )

        result = PersonSummary(summary=summary, metrics=metrics)
        print("\n\nStructured output:")
        print(asdict(result))
        return result

    except Exception as e:
        print(f"\nError processing response: {e}")
        return None


async def complicated_structures():
    class StepDetail(BaseModel):
        step_number: int = 0
        action_taken: str = ""
        observation: str = ""
        tool_call: Optional[Dict[str, Any]] = None

    class Summary(BaseModel):
        step_count: int = 0
        final_output: Any = None
        steps: List[StepDetail] = Field(default_factory=list)
        tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    class Metrics(BaseModel):
        time_taken_seconds: float = 0.0
        tokens_used: int = 0

    class PersonSummary(BaseModel):
        summary: Summary = Field(default_factory=Summary)
        metrics: Metrics = Field(default_factory=Metrics)

    ctx = MyContext(user_id="abc123", is_premium=True)

    # Create a more complex agent with structured output
    structured_agent = Agent[MyContext](
        name="Structured Output Agent",
        model="gpt-4o-mini",
        tools=[get_time],
        output_type=AgentOutputSchema(PersonSummary, strict_json_schema=False),
    )

    final_response = Runner.run(
        structured_agent,
        "Provide a detailed analysis with multiple steps and metrics.",
        context=ctx,
    )
    print(final_response)

    # Run the agent with streaming
    stream_res = Runner.run_streamed(
        structured_agent,
        "Provide a detailed analysis with multiple steps and metrics.",
        context=ctx,
    )

    # Process the stream
    iterations = 0
    full_response = ""
    async for response in stream_res.stream_events():
        # if response.type == "raw_response_event" and isinstance(response.data, ResponseTextDeltaEvent):
        #     print(vars(response))
        #     print(response.data)
        #     full_response += str(response.data.delta)

        print(response)
    print("=" * 80)
    print("Total iterations:", iterations)
    print("=" * 80)
    print("Final response:")
    print(full_response)
    print("=" * 80)

    # Parse and structure the response
    try:
        print(AgentOutputSchema(PersonSummary, strict_json_schema=False)._output_schema)
        response_data = json.loads(
            full_response
        )  # In production, use json.loads() with proper error handling

        # # Create structured output
        # steps = [
        #     StepDetail(
        #         step_number=i+1,
        #         action_taken=step.get('action', ''),
        #         observation=step.get('observation', ''),
        #         tool_call=step.get('tool_call')
        #     )
        #     for i, step in enumerate(response_data.get('steps', []))
        # ]

        # summary = Summary(
        #     step_count=len(steps),
        #     final_output=response_data.get('final_output', ''),
        #     steps=steps,
        #     tool_calls=response_data.get('tool_calls', [])
        # )

        # metrics = Metrics(
        #     time_taken_seconds=response_data.get('metrics', {}).get('time_taken_seconds', 0.0),
        #     tokens_used=response_data.get('metrics', {}).get('tokens_used', 0)
        # )

        response = PersonSummary.model_validate(response_data)
        print("\n\nStructured output:")
        print(response)
        print(type(response))
        return response

    except Exception as e:
        print(f"\nError processing response: {e}")
        return None


if __name__ == "__main__":
    # run simple output structure
    asyncio.run(simple_nested_structure())
    # asyncio.run(complicated_structure())
    asyncio.run(complicated_structures())
