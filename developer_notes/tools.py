from dataclasses import dataclass
from typing import List
import numpy as np
import time
import asyncio


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    time.sleep(1)
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    time.sleep(1)
    return a + b


async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    await asyncio.sleep(1)
    return float(a) / b


async def search(query: str) -> List[str]:
    """Search for query and return a list of results."""
    await asyncio.sleep(1)
    return ["result1" + query, "result2" + query]


def numpy_sum(arr: np.ndarray) -> float:
    """Sum the elements of an array."""
    return np.sum(arr)


x = 2


@dataclass
class Point:
    x: int
    y: int


def add_points(p1: Point, p2: Point) -> Point:
    return Point(p1.x + p2.x, p1.y + p2.y)


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    import json

    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
