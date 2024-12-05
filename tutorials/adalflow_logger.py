"""
This script demonstrates the usage of AdalFlow's Logger functionality.
It can be run independently to showcase logging capabilities.
"""

from adalflow.components import Generator
from adalflow.components.model_client import OpenAIClient
from adalflow.utils import setup_env
from adalflow.utils.logger import get_logger
import logging
from typing import Dict, Any
import json


def setup_logging(log_file: str = "adalflow.log") -> logging.Logger:
    """
    Initialize and configure the logger.

    Args:
        log_file: Name of the log file

    Returns:
        Configured logger instance
    """
    logger = get_logger(__name__)

    # Add file handler if not already present
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def setup_generator() -> Generator:
    """
    Initialize and configure the Generator with OpenAI client.

    Returns:
        Configured Generator instance
    """
    setup_env()
    return Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0, "max_tokens": 1000},
    )


def process_query(
    generator: Generator, query: str, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a query using the generator and log the interaction.

    Args:
        generator: The configured Generator instance
        query: User query to process
        logger: Logger instance for recording the interaction

    Returns:
        Dictionary containing the query and response
    """
    logger.info(f"Processing query: {query}")

    try:
        # Generate response
        response = generator.generate(prompt_kwargs={"query": query})

        # Log successful response
        logger.info(f"Generated response: {response}")

        return {"query": query, "response": str(response), "status": "success"}

    except Exception as e:
        # Log error if generation fails
        logger.error(f"Error processing query: {str(e)}")
        return {"query": query, "response": None, "status": "error", "error": str(e)}


def analyze_logs(log_file: str, logger: logging.Logger) -> Dict[str, int]:
    """
    Analyze the log file to gather statistics.

    Args:
        log_file: Path to the log file
        logger: Logger instance for recording the analysis

    Returns:
        Dictionary containing log statistics
    """
    stats = {"total_queries": 0, "successful_queries": 0, "failed_queries": 0}

    try:
        with open(log_file, "r") as f:
            for line in f:
                if "Processing query:" in line:
                    stats["total_queries"] += 1
                if "Generated response:" in line:
                    stats["successful_queries"] += 1
                if "Error processing query:" in line:
                    stats["failed_queries"] += 1

        logger.info(f"Log analysis complete: {json.dumps(stats, indent=2)}")
        return stats

    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        return stats


def main():
    """Main function to demonstrate logger functionality."""
    # Setup
    log_file = "adalflow.log"
    logger = setup_logging(log_file)
    generator = setup_generator()

    # Example queries
    queries = [
        "What is artificial intelligence?",
        "Explain the concept of machine learning.",
        "Tell me about neural networks.",
    ]

    # Process queries
    results = []
    for query in queries:
        result = process_query(generator, query, logger)
        results.append(result)
        print(f"\nQuery: {query}")
        print(f"Response: {result['response']}")

    # Analyze logs
    stats = analyze_logs(log_file, logger)
    print("\nLog Analysis:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
