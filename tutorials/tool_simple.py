"""
Simple FunctionTool Examples

This file demonstrates all the different types of functions and use cases
supported by AdalFlow's FunctionTool class from the documentation.
"""

import asyncio
from typing import List
from dataclasses import dataclass
import numpy as np
from adalflow.core.func_tool import FunctionTool
from adalflow.core import Component
from adalflow.core.types import ToolOutput
from adalflow.utils import setup_env

setup_env()


# ---------------------------------------------------------------------------
# 1. Basic Example
# ---------------------------------------------------------------------------

def add(a: int, b: int) -> int:
    """Add two numbers."""  # This docstring becomes the tool description for LLM
    return a + b


def basic_example():
    """Basic FunctionTool example showing output fields."""
    print("\n=== Basic FunctionTool Example ===")
    
    # Wrap the function
    add_tool = FunctionTool(add)
    
    # Execute the tool
    result = add_tool.call(2, 3)
    print(f"result.output: {result.output}")  # Output: 5
    print(f"result.name: {result.name}")      # Output: "add"


# ---------------------------------------------------------------------------
# 2. Synchronous Functions
# ---------------------------------------------------------------------------

def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width


def sync_function_example():
    """Synchronous function example."""
    print("\n=== Synchronous Function Example ===")
    
    area_tool = FunctionTool(calculate_area)
    result = area_tool.call(5.0, 3.0)
    print(f"Rectangle area: {result.output}")  # Output: 15.0


# ---------------------------------------------------------------------------
# 3. Asynchronous Functions
# ---------------------------------------------------------------------------

async def fetch_data(url: str) -> dict:
    """Fetch data from a URL asynchronously."""
    # Simulate async operation
    await asyncio.sleep(1)
    return {"data": f"Content from {url}", "status": "success"}


async def async_function_example():
    """Asynchronous function example."""
    print("\n=== Asynchronous Function Example ===")
    
    fetch_tool = FunctionTool(fetch_data)
    # Use acall for async functions
    result = await fetch_tool.acall("https://api.example.com")
    print(f"Fetched data: {result.output}")
    # Output: {"data": "Content from https://api.example.com", "status": "success"}


# ---------------------------------------------------------------------------
# 4. Synchronous Generators
# ---------------------------------------------------------------------------

def count_to_n(n: int):
    """Count from 1 to n, yielding each number."""
    for i in range(1, n + 1):
        yield i


def sync_generator_example():
    """Synchronous generator example."""
    print("\n=== Synchronous Generator Example ===")
    
    counter_tool = FunctionTool(count_to_n)
    result = counter_tool.call(5)
    
    # For generators, output contains the generator object
    print("Counting:")
    for num in result.output:
        print(f"  {num}")  # Outputs: 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# 5. Asynchronous Generators
# ---------------------------------------------------------------------------

async def stream_updates(source: str):
    """Stream updates from a source."""
    for i in range(3):
        await asyncio.sleep(0.5)
        yield f"Update {i} from {source}"


async def async_generator_example():
    """Asynchronous generator example."""
    print("\n=== Asynchronous Generator Example ===")
    
    stream_tool = FunctionTool(stream_updates)
    result = await stream_tool.acall("sensor1")
    
    print("Streaming updates:")
    async for update in result.output:
        print(f"  {update}")  # Outputs updates over time


# ---------------------------------------------------------------------------
# 6. Class Methods and Component Integration
# ---------------------------------------------------------------------------

class DataProcessor(Component):
    def __init__(self):
        super().__init__()
        self.preprocessing_steps = ["normalize", "clean"]
    
    def process_text(self, text: str) -> str:
        """Process text through predefined steps."""
        # Access instance attributes
        for step in self.preprocessing_steps:
            text = f"[{step}] {text}"
        return text


def class_method_example():
    """Class method wrapping example."""
    print("\n=== Class Method Example ===")
    
    processor = DataProcessor()
    # Wrap instance method - maintains access to self
    process_tool = FunctionTool(processor.process_text)
    result = process_tool.call("Hello World")
    print(f"Processed text: {result.output}")  
    # Output: "[normalize] [clean] Hello World"


# ---------------------------------------------------------------------------
# 7. Working with Complex Types
# ---------------------------------------------------------------------------

@dataclass
class Point:
    x: float
    y: float
    
    def __str__(self):
        return f"Point(x={self.x:.3f}, y={self.y:.3f})"


def calculate_centroid(points: List[Point]) -> Point:
    """Calculate the centroid of a list of points."""
    if not points:
        return Point(0, 0)
    avg_x = sum(p.x for p in points) / len(points)
    avg_y = sum(p.y for p in points) / len(points)
    return Point(avg_x, avg_y)


def complex_types_example():
    """Complex types handling example."""
    print("\n=== Complex Types Example ===")
    
    # FunctionTool handles complex parameter types
    centroid_tool = FunctionTool(calculate_centroid)
    points = [Point(0, 0), Point(2, 0), Point(1, 2)]
    result = centroid_tool.call(points)
    print(f"Centroid: {result.output}")  # Output: Point(x=1.000, y=0.667)


# ---------------------------------------------------------------------------
# 8. Using ToolOutput for Enhanced Control
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> ToolOutput:
    """Analyze sentiment with detailed feedback."""
    # Simulate analysis
    score = 0.8 if "happy" in text.lower() else 0.2
    
    return ToolOutput(
        output={"sentiment": "positive" if score > 0.5 else "negative", "score": score},
        observation=f"Sentiment analysis complete. Score: {score}",
        display=f"😊 Positive ({score:.0%})" if score > 0.5 else f"😢 Negative ({score:.0%})",
        metadata={"model": "simple-rule-based", "confidence": "low"}
    )


def tool_output_example():
    """ToolOutput usage example."""
    print("\n=== ToolOutput Example ===")
    
    sentiment_tool = FunctionTool(analyze_sentiment)
    result = sentiment_tool.call("I am very happy today!")
    
    print(f"Output (data): {result.output}")       # The actual data
    print(f"Observation: {result.output.observation}")    # For agent reasoning
    print(f"Display: {result.output.display}")            # For user display


# ---------------------------------------------------------------------------
# 9. Error Handling
# ---------------------------------------------------------------------------

def divide(a: float, b: float) -> float:
    """Divide two numbers safely."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def error_handling_example():
    """Error handling example."""
    print("\n=== Error Handling Example ===")
    
    divide_tool = FunctionTool(divide)
    
    # Successful call
    print("Successful division:")
    result = divide_tool.call(10, 2)
    print(f"  result.output: {result.output}")  # Output: 5.0
    print(f"  result.error: {result.error}")    # Output: None
    
    # Error case
    print("\nDivision by zero:")
    result = divide_tool.call(10, 0)
    print(f"  result.output: {result.output}")  # Output: "Error: Cannot divide by zero"
    print(f"  result.error: {result.error}")    # Contains the actual exception


# ---------------------------------------------------------------------------
# 10. Integration with Trainable Components (Structure Example)
# ---------------------------------------------------------------------------

def trainable_component_example():
    """Shows the structure for integrating trainable components."""
    print("\n=== Trainable Component Integration (Structure) ===")
    
    print("""
    from adalflow.core import Component
    from adalflow.core.generator import Generator
    
    class AgenticRAG(Component):
        def __init__(self):
            super().__init__()
            self.retriever = Retriever()  # Trainable component
            self.llm = Generator()         # Trainable component
            
            # Method 1: Wrap component method
            def retrieve_with_context(query: str) -> str:
                \"\"\"Retrieve relevant documents for the query.\"\"\"
                return self.retriever(query)
            
            # Tools maintain gradient flow for training
            self.tools = [
                FunctionTool(retrieve_with_context, component=self.retriever),
                FunctionTool(self.llm.__call__, component=self.llm)
            ]
    """)
    
    print("Note: When wrapping trainable components, pass the 'component' parameter")
    print("      to maintain gradient flow during training.")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_sync_examples():
    """Run all synchronous examples."""
    basic_example()
    sync_function_example()
    sync_generator_example()
    class_method_example()
    complex_types_example()
    tool_output_example()
    error_handling_example()
    trainable_component_example()


async def run_async_examples():
    """Run all asynchronous examples."""
    await async_function_example()
    await async_generator_example()


async def main():
    """Main function to run all examples."""
    print("=" * 60)
    print("FunctionTool Examples")
    print("=" * 60)
    
    # Run synchronous examples
    run_sync_examples()
    
    # Run asynchronous examples
    print("\n" + "=" * 60)
    print("Asynchronous Examples")
    print("=" * 60)
    await run_async_examples()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())