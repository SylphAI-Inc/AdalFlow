# """
# Instructions:

# 1. Install dependencies:
#    pip install mlflow opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# 2. Start the MLflow Tracking Server (in a separate terminal):
#    mlflow server --host 0.0.0.0 --port 8080

# 3. Run this script:
#    python this_script.py

# 4. Open your browser to http://localhost:8080
#    - Find your run under the default experiment.
#    - Click on the run, then the "Traces" tab to view your spans!
# """

# import os
# import time

# import mlflow
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# # Use the HTTP OTLP exporter
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# # ─── MLflow Setup ────────────────────────────────────────────────────────────────

# # Point MLflow at your tracking server
# mlflow.set_tracking_uri("http://localhost:8080")
# mlflow.set_experiment("otel-mlflow-multi-span-demo")

# # ─── OpenTelemetry Setup ─────────────────────────────────────────────────────────

# # 1) Tell MLflow server to accept OTLP traces at its /otel/v1/traces endpoint
# os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:8080/otel/v1/traces"  # MLflow’s OTLP receiver :contentReference[oaicite:0]{index=0}
# os.environ["OTEL_SERVICE_NAME"] = "otel-mlflow-demo"

# # 2) Install the tracer provider and OTLP exporter
# trace.set_tracer_provider(TracerProvider())
# provider = trace.get_tracer_provider()
# otlp_exporter = OTLPSpanExporter(
#     endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
# )
# provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# # 3) Grab a tracer
# tracer = trace.get_tracer(__name__)

# # ─── Your Workload ───────────────────────────────────────────────────────────────

# def load_data():
#     with tracer.start_as_current_span("load_data") as span:
#         span.set_attribute("operation", "load")
#         span.set_attribute("source", "csv")
#         time.sleep(0.5)  # Simulate loading
#         data = list(range(10_000))
#         span.set_attribute("num_records", len(data))
#         return data

# def process_data(data):
#     with tracer.start_as_current_span("process_data") as span:
#         span.set_attribute("operation", "normalize")
#         span.set_attribute("input_type", "list")
#         with tracer.start_as_current_span("scale_data") as subspan:
#             subspan.set_attribute("method", "min-max-scaling")
#             time.sleep(0.3)  # Simulate scaling
#         processed = [x / max(data) for x in data]
#         span.set_attribute("output_type", "list")
#         return processed

# # ─── Tie It All Together ────────────────────────────────────────────────────────

# with mlflow.start_run(run_name="otel-mlflow-multi-span-demo"):
#     mlflow.log_param("example_param", 42)

#     data = load_data()
#     processed = process_data(data)

#     mlflow.log_metric("max_processed", max(processed))
#     mlflow.log_metric("mean_processed", sum(processed) / len(processed))

# print("Run complete! View traces in the MLflow UI (http://localhost:8080).")

"""
Using set trace processor of the OpenAI Agents SDK
"""


import asyncio

import mlflow
from agents import Agent, Runner, set_trace_processors, WebSearchTool
from mlflow.openai._agent_tracer import MlflowOpenAgentTracingProcessor

from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────────────────
# MLflow configuration
# ───────────────────────────────────────────────────────────
# Point MLflow to the tracking server
mlflow.set_tracking_uri("http://localhost:8080")  # Match your server port

# Create or set an experiment to log traces under
mlflow.set_experiment("CustomAgentTraceExperiment")

# ───────────────────────────────────────────────────────────
# Register MLflow as the trace processor
# ───────────────────────────────────────────────────────────
# Instantiate MLflow's OpenAI Agents tracing processor
mlflow_processor = MlflowOpenAgentTracingProcessor(project_name="MyAgentProject")

# Replace default OpenAI trace processors with MLflow's
set_trace_processors(
    [mlflow_processor]
)  # Overrides default exporters :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

# def print_hello_in_spanish():
#     return "Hola"

# def print_hello_in_english():
#     return "Hello"

# # ───────────────────────────────────────────────────────────
# # Define agents and workflow
# # ───────────────────────────────────────────────────────────
# # Agent that only speaks Spanish
# spanish_agent = Agent(
#     name="SpanishAgent",
#     instructions="“Whenever you need to look up facts, you must call the WebSearchTool first. Do not produce any answer text until after you’ve fetched the tool’s resultl",
#     tools=[WebSearchTool()]
# )

# # Agent that only speaks English
# english_agent = Agent(
#     name="EnglishAgent",
#     instructions="“Whenever you need to look up facts, you must call the WebSearchTool first. Do not produce any answer text until after you’ve fetched the tool’s resultl",
#     tools=[WebSearchTool()]
# )

# # Triage agent: decides which sub-agent to hand off to based on language
# triage_agent = Agent(
#     name="TriageAgent",
#     instructions="Detect the language of the user message and hand off to the appropriate agent.",
#     handoffs=[spanish_agent, english_agent]
# )

# # ───────────────────────────────────────────────────────────
# # Execute the agent workflow
# # ───────────────────────────────────────────────────────────
# async def main():
#     # Run the triage workflow with a sample input
#     result = await Runner.run(
#         triage_agent,
#         input="Hola, ¿cómo estás? Give me some five recent world news evens that are from the web in spanish using the tool"
#     )
#     # Print the final output
#     print("Final Output:", result.final_output)

# if __name__ == "__main__":
#     asyncio.run(main())
#     print("Run complete! View traces under experiment 'CustomAgentTraceExperiment' in the MLflow UI.")


agent = Agent(
    name="Assistant",
    instructions=(
        "You are a research assistant.  "
        "Whenever you need to fetch up‐to‐date or location‐specific information, "
        "you **must** call the `WebSearchTool.search` function first.  "
        "Do not generate any answer text until after you’ve received the tool’s result."
    ),
    tools=[WebSearchTool()],
)


async def main():
    result = await Runner.run(
        agent, "Which coffee shop should I go to in SF I want the most popular cafe?"
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
