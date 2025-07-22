# Start the MLflow Tracking Server (in a separate terminal):
#    mlflow server --host 0.0.0.0 --port 8080

import os
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# the environemnt variable must be set before the import
os.environ["ADALFLOW_DISABLE_TRACING"] = "False"

# AdalFlow imports
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.components.model_client import OpenAIClient
from adalflow.tracing import set_trace_processors, trace

# MLflow integration
import mlflow
from mlflow.openai._agent_tracer import MlflowOpenAgentTracingProcessor

MLFLOW_AVAILABLE = True

from adalflow.utils import setup_env

setup_env()


# ───────────────────────────────────────────────────────────
# MLflow configuration and trace processor setup
# ───────────────────────────────────────────────────────────
def setup_mlflow_tracing():
    """Set up MLflow tracing for AdalFlow workflows."""
    if not MLFLOW_AVAILABLE:
        log.warning("MLflow not available - traces will only be logged to console")
        return

    # Point MLflow to the tracking server
    mlflow.set_tracking_uri("http://localhost:8080")  # Match your server port

    # Create or set an experiment to log traces under
    mlflow.set_experiment("AdalFlow-Agent-Tracing-Experiment")

    # Instantiate MLflow's OpenAI Agents tracing processor
    mlflow_processor = MlflowOpenAgentTracingProcessor(
        project_name="AdalFlow-Agent-Project"
    )

    # Replace default trace processors with MLflow's
    set_trace_processors([mlflow_processor])
    log.info("✅ MLflow tracing configured successfully")


# ───────────────────────────────────────────────────────────
# Define AdalFlow agents and workflow
# ───────────────────────────────────────────────────────────


def create_spanish_agent() -> Agent:
    """Agent that only speaks Spanish."""
    return Agent(
        name="Spanish-Agent",
        model_client=OpenAIClient(),
        # model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=3,
        answer_data_type=str,
    )


def create_english_agent() -> Agent:
    """Agent that only speaks English."""
    return Agent(
        name="English-Agent",
        model_client=OpenAIClient(),
        # model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3},
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=3,
        answer_data_type=str,
    )


def create_triage_agent() -> Agent:
    """Triage agent: decides language and provides appropriate response."""
    return Agent(
        name="Triage-Agent",
        model_client=OpenAIClient(),
        # model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3},
        model_kwargs={"model": "gpt-4o", "temperature": 0.3},
        max_steps=5,
        answer_data_type=str,
    )


# ───────────────────────────────────────────────────────────
# Execute the AdalFlow workflow
# ───────────────────────────────────────────────────────────
async def main():
    """Main function to run the AdalFlow workflow with tracing."""

    # Setup MLflow tracing
    setup_mlflow_tracing()

    # Create the triage agent
    triage_agent = create_triage_agent()

    # Create runner
    runner = Runner(agent=triage_agent)

    # Create a trace for the entire workflow
    with trace(workflow_name="AdalFlow-Agent"):
        # Run the triage workflow with a sample input
        result = runner.call(
            prompt_kwargs={"input_str": "Hola, ¿cómo estás?"}, id="triage_run_001"
        )

        # Print the final output
        if result:
            print("Final Output:", result.answer)
            print("Steps executed:", len(result.step_history))
        else:
            print("❌ No result returned from runner")


async def demo_multiple_languages():
    """Demonstrate with multiple language inputs."""

    test_cases = [
        ("Hola, ¿cómo estás? ¿Puedes ayudarme?", "Spanish-Input"),
        ("Hello, how are you? Can you help me?", "English-Input"),
        ("Bonjour, comment ça va?", "French-Input"),
    ]

    triage_agent = create_triage_agent()
    runner = Runner(agent=triage_agent)

    for query, case_name in test_cases:
        print(f"\n--- Testing {case_name} ---")
        print(f"Input: {query}")

        with trace(workflow_name=f"AdalFlow-{case_name}"):
            result = runner.call(
                prompt_kwargs={"input_str": query}, id=f"run_{case_name.lower()}"
            )

            if result:
                print(f"Output: {result.answer}")
            else:
                print("❌ No result returned from runner")


async def demo_astream_tracing():
    """Demonstrate Runner.astream with tracing."""
    print("\n🔄 Testing AdalFlow Runner.astream with tracing...")

    # Create the triage agent
    triage_agent = create_triage_agent()
    runner = Runner(agent=triage_agent)

    # Test streaming with tracing
    with trace(workflow_name="AdalFlow-Streaming-Agent"):
        print("Starting streaming execution...")

        # Use astream for streaming execution
        stream_result = runner.astream(
            prompt_kwargs={"input_str": "Hello, can you help me with a task?"},
            id="streaming_run_001",
        )

        print("Waiting for streaming to complete...")
        await stream_result.wait_for_completion()

        # Check if there was an error
        if hasattr(stream_result, "exception") and stream_result.exception:
            print(f"❌ Streaming failed with error: {stream_result.exception}")
            return

        # Check if we have a final result
        if stream_result.answer:
            print("✅ Streaming completed successfully!")
            print(f"Final Answer: {stream_result.answer}")
            print(f"Steps executed: {len(stream_result.step_history)}")

            # Print step details
            for i, step in enumerate(stream_result.step_history):
                print(f"  Step {i}: {step.function.name} -> {step.observation}")
        else:
            print("❌ No result returned from streaming runner")


async def demo_astream_error_handling():
    """Demonstrate Runner.astream error handling with tracing."""
    print("\n⚠️  Testing AdalFlow Runner.astream error handling...")

    # Create an agent that might encounter errors
    triage_agent = create_triage_agent()
    runner = Runner(agent=triage_agent)

    # Test streaming with potential error conditions
    with trace(workflow_name="AdalFlow-Streaming-Error-Test"):
        print("Starting streaming execution with potential errors...")

        # Use astream with a complex query that might cause issues
        stream_result = runner.astream(
            prompt_kwargs={
                "input_str": "This is a very complex query that might cause the agent to exceed max steps or encounter errors in processing."
            },
            id="streaming_error_test_001",
        )

        print("Waiting for streaming to complete...")
        await stream_result.wait_for_completion()

        # Check the final result
        if hasattr(stream_result, "exception") and stream_result.exception:
            print(f"⚠️  Streaming completed with error: {stream_result.exception}")
        elif stream_result.answer:
            print("✅ Streaming completed!")
            print(f"Final Answer: {stream_result.answer}")
            print(f"Steps executed: {len(stream_result.step_history)}")
        else:
            print("⚠️  Streaming completed without final result")


async def demo_astream_multi_step():
    """Demonstrate Runner.astream with multiple steps and tracing."""
    print("\n🔢 Testing AdalFlow Runner.astream with multi-step workflow...")

    # Create an agent with more steps to demonstrate multi-step streaming
    multi_step_agent = create_triage_agent()
    multi_step_agent.max_steps = 8  # Allow more steps for complex workflow
    runner = Runner(agent=multi_step_agent)

    # Test streaming with multi-step workflow
    with trace(workflow_name="AdalFlow-Streaming-MultiStep"):
        print("Starting multi-step streaming execution...")

        stream_result = runner.astream(
            prompt_kwargs={
                "input_str": "Please help me plan a detailed strategy for learning a new programming language, including resources, timeline, and practice projects."
            },
            id="streaming_multistep_001",
        )

        print("Waiting for multi-step streaming to complete...")
        await stream_result.wait_for_completion()

        # Check the final result
        if hasattr(stream_result, "exception") and stream_result.exception:
            print(
                f"❌ Multi-step streaming failed with error: {stream_result.exception}"
            )
        elif stream_result.answer:
            print("✅ Multi-step streaming completed!")
            print(f"Final Answer: {stream_result.answer}")
            print(f"Total steps executed: {len(stream_result.step_history)}")

            # Print detailed step information
            print("\n📋 Step-by-step execution:")
            for i, step in enumerate(stream_result.step_history):
                print(f"  Step {i+1}: Function '{step.function.name}'")
                print(f"    Args: {step.function.args}")
                print(f"    Result: {str(step.observation)[:100]}...")
                print()
        else:
            print("❌ Multi-step streaming did not produce a result")


if __name__ == "__main__":
    print("🚀 Starting AdalFlow tracing demo...")

    # Run basic demo
    asyncio.run(main())

    # # Run streaming demos
    print("\n" + "=" * 60)
    print("🔄 Running streaming demos...")
    asyncio.run(demo_astream_tracing())

    # print("\n" + "=" * 60)
    # print("🔢 Running multi-step streaming demo...")
    # asyncio.run(demo_astream_multi_step())

    print("\n" + "=" * 60)
    print("⚠️  Running streaming error handling demo...")
    asyncio.run(demo_astream_error_handling())

    # # Run extended demo
    print("\n" + "=" * 60)
    print("🌍 Running multi-language demo...")
    asyncio.run(demo_multiple_languages())

    print("\n" + "=" * 60)
    print("✅ Demo complete!")

    if MLFLOW_AVAILABLE:
        print(
            "📊 View traces under experiment 'AdalFlow-Agent-Tracing-Experiment' in the MLflow UI."
        )
        print("🔗 MLflow UI: http://localhost:8080")
        print(
            "🔍 Look for streaming workflows with names starting with 'AdalFlow-Streaming-'"
        )
    else:
        print("💡 Install MLflow to view traces in UI: pip install mlflow")
        print("🔧 Traces are still being generated - check console output")
