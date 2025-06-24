#!/usr/bin/env python3
"""
Minimal MCP Server using the official Python SDK
Demonstrates resources, tools, and prompts using FastMCP

Installation:
pip install "mcp[cli]"

Usage:
# Test with MCP Inspector (recommended for development)
mcp dev server.py

# Install in Claude Desktop
mcp install server.py

# Run directly
python server.py


"""

import asyncio
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP, Context

# Create an MCP server instance
mcp = FastMCP("Demo Server")

# === RESOURCES ===
# Resources provide data to LLMs (like GET endpoints)


@mcp.resource("config://app")
def get_app_config() -> str:
    """Get application configuration information"""
    config = {
        "app_name": "MCP Demo Server",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "features": ["calculator", "weather", "greetings"],
    }
    return json.dumps(config, indent=2)


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting for someone"""
    if not name:
        return "Hello, stranger!"
    return f"Hello, {name}! Welcome to the MCP Demo Server."


@mcp.resource("time://current")
def get_current_time() -> str:
    """Get the current date and time"""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# === TOOLS ===
# Tools allow LLMs to perform actions (like POST endpoints)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers together"""
    return a / b


@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI given weight in kg and height in meters"""
    if height_m <= 0:
        raise ValueError("Height must be greater than 0")

    bmi = weight_kg / (height_m**2)

    # Determine BMI category
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "bmi": round(bmi, 2),
        "category": category,
        "weight_kg": weight_kg,
        "height_m": height_m,
    }


@mcp.tool()
def generate_password(length: int = 12, include_symbols: bool = True) -> str:
    """Generate a random password"""
    import random
    import string

    if length < 4:
        raise ValueError("Password length must be at least 4 characters")

    chars = string.ascii_letters + string.digits
    if include_symbols:
        chars += "!@#$%^&*"

    password = "".join(random.choice(chars) for _ in range(length))
    return password


# === PROMPTS ===
# Prompts are reusable templates for LLM interactions


@mcp.prompt()
def code_review(code: str) -> str:
    """Generate a code review prompt"""
    return f"""Please review the following code and provide feedback on:
1. Code quality and readability
2. Potential bugs or issues
3. Performance improvements
4. Best practices

Code to review:
```
{code}
```

Please provide constructive feedback and suggestions for improvement."""


@mcp.prompt()
def explain_concept(concept: str, audience: str = "general") -> str:
    """Generate a prompt to explain a concept to a specific audience"""
    return f"""Please explain the concept of "{concept}" in a way that's appropriate for a {audience} audience.

Make sure to:
1. Start with a simple definition
2. Provide relevant examples
3. Explain why it's important or useful
4. Use language appropriate for the {audience} level

Concept to explain: {concept}
Target audience: {audience}"""


@mcp.prompt()
def debug_error(error_message: str, code_context: str = "") -> str:
    """Generate a debugging prompt"""
    prompt = f"""I'm encountering the following error and need help debugging it:

Error: {error_message}"""

    if code_context:
        prompt += f"""

Code context:
```
{code_context}
```"""

    prompt += """

Please help me:
1. Understand what this error means
2. Identify the likely cause
3. Suggest specific steps to fix it
4. Provide best practices to prevent similar errors"""

    return prompt


# === ADVANCED TOOL WITH CONTEXT ===


@mcp.tool()
async def process_list(items: list[str], operation: str, ctx: Context) -> dict:
    """Process a list of items with progress reporting"""
    valid_operations = ["uppercase", "lowercase", "reverse", "length"]

    if operation not in valid_operations:
        raise ValueError(f"Operation must be one of: {valid_operations}")

    results = []
    total_items = len(items)

    for i, item in enumerate(items):
        # Report progress
        await ctx.report_progress(i, total_items)
        await ctx.info(f"Processing item {i+1}/{total_items}: {item}")

        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Apply operation
        if operation == "uppercase":
            result = item.upper()
        elif operation == "lowercase":
            result = item.lower()
        elif operation == "reverse":
            result = item[::-1]
        elif operation == "length":
            result = len(item)

        results.append(result)

    return {
        "operation": operation,
        "original_items": items,
        "processed_items": results,
        "total_processed": len(results),
    }


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # Run the server
    print("Starting MCP Demo Server...")
    print("Use 'mcp dev server.py' to test with MCP Inspector")
    print("Use 'mcp install server.py' to install in Claude Desktop")
    mcp.run()
