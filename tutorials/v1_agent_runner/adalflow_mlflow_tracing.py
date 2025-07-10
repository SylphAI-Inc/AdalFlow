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

    try:
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

    except Exception as e:
        log.error(f"Failed to setup MLflow tracing: {e}")
        log.info("Continuing with default tracing...")


# ───────────────────────────────────────────────────────────
# Define AdalFlow agents and workflow
# ───────────────────────────────────────────────────────────


def create_spanish_agent() -> Agent:
    """Agent that only speaks Spanish."""
    return Agent(
        name="Spanish-Agent",
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
        max_steps=3,
        answer_data_type=str,
    )


def create_english_agent() -> Agent:
    """Agent that only speaks English."""
    return Agent(
        name="English-Agent",
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.7},
        max_steps=3,
        answer_data_type=str,
    )


def create_triage_agent() -> Agent:
    """Triage agent: decides language and provides appropriate response."""
    return Agent(
        name="Triage-Agent",
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3},
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


if __name__ == "__main__":
    print("🚀 Starting AdalFlow tracing demo...")

    # Run basic demo
    asyncio.run(main())

    # # Run extended demo
    # print("\n" + "="*60)
    # print("🌍 Running multi-language demo...")
    # asyncio.run(demo_multiple_languages())

    # print("\n" + "="*60)
    # print("✅ Demo complete!")

    # if MLFLOW_AVAILABLE:
    #     print("📊 View traces under experiment 'AdalFlow-Agent-Tracing-Experiment' in the MLflow UI.")
    #     print("🔗 MLflow UI: http://localhost:8080")
    # else:
    #     print("💡 Install MLflow to view traces in UI: pip install mlflow")
    #     print("🔧 Traces are still being generated - check console output")
