#!/usr/bin/env python3
"""
Test script to demonstrate streaming from Anthropic and OpenAI APIs
and print the event types for comparison.
"""

import os
import asyncio
import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def test_anthropic_streaming():
    """Test Anthropic API streaming and print event types"""
    print("=== ANTHROPIC API STREAMING ===")

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        async with client.messages.stream(
            max_tokens=100,
            messages=[{"role": "user", "content": "Tell me a short joke"}],
            model="claude-3-haiku-20240307",
        ) as stream:
            async for event in stream:
                print("Event:", event)
                print(f"Event type: {event.type}")
                print("-" * 50)
    except Exception as e:
        print(f"Anthropic streaming error: {e}")


async def test_openai_streaming():
    """Test OpenAI API streaming and print event types"""
    print("\n=== OPENAI API STREAMING ===")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a short joke"}],
            stream=True,
        )

        async for chunk in stream:
            print("Event:", chunk)
            print(f"Event type: {type(chunk)}")
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    print(f"Content: {delta.content}")
            print("-" * 50)
    except Exception as e:
        print(f"OpenAI streaming error: {e}")


async def test_anthropic_via_openai_sdk():
    """Test Anthropic API via OpenAI SDK streaming"""
    print("\n=== ANTHROPIC API VIA OPENAI SDK ===")

    api_key = os.getenv("ANTHROPIC_API_KEY")

    client = AsyncOpenAI(
        api_key=api_key,  # Your Anthropic API key
        base_url="https://api.anthropic.com/v1/",  # Anthropic's API endpoint
    )

    try:
        stream = await client.chat.completions.create(
            model="claude-3-haiku-20240307",  # Anthropic model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"},
            ],
            stream=True,
        )

        async for chunk in stream:
            print("Event:", chunk)
            print(f"Event type: {type(chunk)}")
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    print(f"Content: {delta.content}")
            print("-" * 50)
    except Exception as e:
        print(f"Anthropic via OpenAI SDK streaming error: {e}")


async def main():
    """Main function to run both streaming tests"""
    print("Starting streaming API tests...")
    print(
        "Make sure you have ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables set."
    )

    # Test Anthropic streaming
    await test_anthropic_streaming()

    # Test OpenAI streaming
    await test_openai_streaming()

    # Test Anthropic via OpenAI SDK
    await test_anthropic_via_openai_sdk()


if __name__ == "__main__":
    asyncio.run(main())
