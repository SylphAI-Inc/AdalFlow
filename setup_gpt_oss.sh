#!/bin/bash

# GPT-OSS Setup Script for Ollama
# This script helps you set up and test OpenAI's GPT-OSS models with Ollama

echo "GPT-OSS Setup Script"
echo "==================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "Please install Ollama from: https://ollama.ai/"
    echo "After installation, run this script again."
    exit 1
fi

echo "✅ Ollama is installed"

# Start Ollama server if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🚀 Starting Ollama server..."
    ollama serve &
    sleep 5  # Wait for server to start
    echo "✅ Ollama server started"
else
    echo "✅ Ollama server is already running"
fi

# Show available models
echo -e "\nCurrent models:"
ollama list

# Check if GPT-OSS is already installed
if ollama list | grep -q "gpt-oss"; then
    echo "✅ GPT-OSS model is already installed"
else
    echo -e "\n📥 GPT-OSS model not found. Installing gpt-oss:20b..."
    echo "This may take several minutes..."
    
    if ollama pull gpt-oss:20b; then
        echo "✅ Successfully installed gpt-oss:20b"
    else
        echo "❌ Failed to install gpt-oss:20b"
        echo "You can try manually: ollama pull gpt-oss:20b"
        exit 1
    fi
fi

# Quick test
echo -e "\n🧪 Running a quick test..."
echo "Query: What is machine learning?"
echo "Response:"
ollama run gpt-oss:20b "Explain machine learning in one paragraph."

echo -e "\n✅ Setup complete!"
echo -e "\nUsage examples:"
echo "1. Interactive chat: ollama run gpt-oss:20b"
echo "2. Single query: ollama run gpt-oss:20b 'Your question here'"
echo "3. Run Python test script: python3 test_gpt_oss.py"
echo -e "\nFor the larger model, use: ollama run gpt-oss:120b"