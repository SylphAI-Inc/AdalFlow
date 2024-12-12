"""
This script demonstrates the usage of AdalFlow's Tool Helper functionality.
It can be run independently to showcase function calling capabilities.
"""

from adalflow.components import Generator
from adalflow.components.model_client import OpenAIClient
from adalflow.utils import setup_env
from typing import List, Dict
import json


def setup_generator():
    """Initialize and configure the Generator with OpenAI client."""
    setup_env()
    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0, "max_tokens": 1000},
    )
    return generator


def define_tools() -> List[Dict]:
    """Define the available tools/functions that can be called."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ]


def get_weather(location: str, unit: str) -> str:
    """Mock function to simulate weather data retrieval."""
    # This is a mock implementation
    weather_data = {
        "San Francisco, CA": {"celsius": 20, "fahrenheit": 68},
        "New York, NY": {"celsius": 22, "fahrenheit": 72},
    }

    if location in weather_data:
        temp = weather_data[location][unit]
        return f"The temperature in {location} is {temp}°{'C' if unit == 'celsius' else 'F'}"
    return f"Weather data not available for {location}"


def process_function_calls(generator: Generator, query: str):
    """Process user query and handle any function calls."""
    # Get the response from the model
    response = generator.generate(prompt_kwargs={"query": query}, tools=define_tools())

    # Check if the response includes a function call
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.function.name == "get_weather":
                # Parse the function arguments
                args = json.loads(tool_call.function.arguments)

                # Call the function with the provided arguments
                weather_result = get_weather(args["location"], args["unit"])

                # Generate final response incorporating the function result
                final_response = generator.generate(
                    prompt_kwargs={"query": query},
                    tools=define_tools(),
                    tool_results=[
                        {"tool_call_id": tool_call.id, "output": weather_result}
                    ],
                )
                return final_response

    return response


def main():
    """Main function to demonstrate tool helper functionality."""
    # Initialize generator
    generator = setup_generator()

    # Example queries
    queries = [
        "What's the weather like in San Francisco?",
        "Tell me the temperature in New York in Celsius",
    ]

    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        response = process_function_calls(generator, query)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
