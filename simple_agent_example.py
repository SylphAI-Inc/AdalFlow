#!/usr/bin/env python3
"""
Enhanced Agent Setup with GPT-OSS

This script shows how to properly set up and use the Agent class with OllamaClient for GPT-OSS models.
Features:
- Better error handling
- More flexible model selection
- Improved tool execution
- Enhanced logging

Before running:
1. Install Ollama: https://ollama.ai/
2. Pull the GPT-OSS model: `ollama pull gpt-oss:20b` (or other gpt-oss variants)
3. Start Ollama server: `ollama serve`
4. The example is configured to use 'gpt-oss:20b' model by default
"""

from typing import Optional
from adalflow.components.agent.agent import Agent
from adalflow.components.agent.runner import Runner
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.utils import printc
import json
import math
import datetime
import sys



def calculator_tool(expression: str) -> str:
    """Enhanced calculator with better error handling."""
    try:
        # Safe math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error for '{expression}': {str(e)}"


def get_datetime_info() -> str:
    """Get comprehensive date/time information."""
    now = datetime.datetime.now()
    return json.dumps({
        "current_datetime": now.strftime('%Y-%m-%d %H:%M:%S'),
        "day_of_week": now.strftime('%A'),
        "month": now.strftime('%B'),
        "year": now.year,
        "timestamp": now.timestamp()
    }, indent=2)


def text_analyzer(text: str) -> str:
    """Enhanced text analysis tool."""
    if not text or not text.strip():
        return "No valid text provided for analysis."
    
    text = text.strip()
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    analysis = {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(' ', '')),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "average_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
        "longest_word": max(words, key=len) if words else "",
        "shortest_word": min(words, key=len) if words else ""
    }
    
    return f"Text Analysis:\n{json.dumps(analysis, indent=2)}"


def create_agent_with_gpt_oss(
    model_name: str = "gpt-oss:20b",
    temperature: float = 0.7,
    max_steps: int = 5,
) -> Agent:
    """
    Create an Agent with OllamaClient for GPT-OSS models.
    
    Args:
        model_name: Name of the GPT-OSS model to use
        temperature: Model temperature (0.0-1.0)
        max_steps: Maximum planning steps
        host: Ollama server host (default: http://localhost:11434)
    
    Returns:
        Configured Agent instance
    """
    try:
        model_client = OllamaClient()

        model_kwargs = {
            "model": model_name,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
            },
            "generate": True  # Use non-chat for LLM
        }
        
        tools = [
            calculator_tool,
            get_datetime_info,
            text_analyzer
        ]
        
        # Create agent with enhanced configuration
        agent = Agent(
            name="EnhancedGPT-OSSAgent",
            model_client=model_client,
            model_kwargs=model_kwargs,
            tools=tools,
            max_steps=max_steps,
            add_llm_as_fallback=True,
            use_cache=False,
            role_desc="You are a helpful AI assistant with access to calculation, time, and text analysis tools. Always think step-by-step and use appropriate tools when needed."
        )
        
        printc(f"Agent created successfully with model: {model_name}")
        return agent
        
    except Exception as e:
        printc(f"Failed to create agent: {e}")
        raise


def main():
    """Enhanced main function using built-in Runner class."""
    print("Enhanced GPT-OSS Agent Setup")
    print("=" * 40)
    
    try:
        agent = create_agent_with_gpt_oss(
            model_name="gpt-oss:20b",  # Change to your preferred model
            temperature=0.7,
            max_steps=2
        )
        
        runner = Runner(agent=agent)
        
        test_tasks = [
            "Calculate the area of a circle with radius 5",
            "What's the current date and time?",
            "Analyze this text: 'Artificial intelligence is transforming our world.'",
            "What is 2^10 plus 50?"
        ]
        
        
        # Interactive mode
        print(f"\n{'='*60}")
        print("💬 Interactive Mode (type 'quit' to exit)")
        
        while True:
            try:
                user_task = input("\nYour task: ").strip()
                if user_task.lower() in ['quit', 'exit', 'q']:
                    break
                if user_task:
                    result = runner.call(prompt_kwargs={"input_str": user_task})
                    print(f"\n🎯 Final Result: {result.answer}")
                    print(f"Steps taken: {len(result.step_history)}")
                    if result.error:
                        print(f"Error: {result.error}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
    except Exception as e:
        printc(f"Main execution error: {e}")
        printc(f"❌ Error: {e}")
        return 1

    printc("\n✅ Session completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
