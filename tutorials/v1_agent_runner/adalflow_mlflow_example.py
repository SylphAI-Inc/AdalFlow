"""Example of using MLflow tracing with AdalFlow.

This example shows how to use the new simplified MLflow integration.
"""

from adalflow.tracing import enable_mlflow_local, trace
from adalflow.components.model_client import OpenAIClient
from adalflow.utils import setup_env
import os

# Setup environment (make sure you have .env file with API keys)
# setup_env()  # Uncomment if you have .env file

# Method 1: Simple usage - auto-starts MLflow server and uses ~/.adalflow/mlruns
print("Method 1: Enabling MLflow with auto-start...")
success = enable_mlflow_local()
print(f"MLflow enabled: {success}")
print("Visit http://localhost:8080 to see traces in the 'Traces' tab\n")

# Method 2: Custom configuration
# enable_mlflow_local(
#     port=5000,                    # Use different port
#     experiment_name="MyProject",  # Custom experiment name
#     kill_existing_server=True     # Kill any existing server on the port
# )

# Method 3: Use with existing MLflow server (no auto-start)
# enable_mlflow_local(
#     tracking_uri="http://localhost:8080",  # Use existing server
#     auto_start_server=False
# )

# Method 4: Don't use adalflow directory
# enable_mlflow_local(
#     use_adalflow_dir=False,  # Will use ./mlruns instead
#     auto_start_server=False
# )

# Now use tracing as normal
if success:
    print("Creating a test trace...")
    with trace("example_trace") as t:
        print(f"Trace created with ID: {t.trace_id}")
        
        # Example: trace an OpenAI call (requires API key)
        if os.getenv("OPENAI_API_KEY"):
            try:
                client = OpenAIClient()
                response = client.call(
                    messages=[{"role": "user", "content": "Say hello in 3 words"}],
                    model="gpt-3.5-turbo"
                )
                print(f"Response: {response.response}")
            except Exception as e:
                print(f"OpenAI call failed: {e}")
        else:
            print("No OPENAI_API_KEY found, skipping OpenAI call")
    
    print("\n✅ Done! Check http://localhost:8080 for traces.")
else:
    print("\n❌ MLflow setup failed. Make sure MLflow is installed: pip install mlflow")