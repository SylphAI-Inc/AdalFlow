"""MLflow integration helpers for AdalFlow tracing."""

import logging
import os
from pathlib import Path

from adalflow.utils import get_adalflow_default_root_path

log = logging.getLogger(__name__)

# Check if MLflow is available
try:
    # Do NOT set ADALFLOW_DISABLE_TRACING here - it should be set before imports
    import mlflow
    from mlflow.openai._agent_tracer import MlflowOpenAgentTracingProcessor

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("MLflow not available. Install with: pip install mlflow")


from .setup import GLOBAL_TRACE_PROVIDER

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions for managing trace processors
# ═══════════════════════════════════════════════════════════════════════════════


def set_trace_processors(processors):
    """Set the global trace processors."""
    GLOBAL_TRACE_PROVIDER.set_processors(processors)


def enable_mlflow_local(
    tracking_uri: str = None,
    experiment_name: str = "AdalFlow-Agent-Experiment",
    project_name: str = "AdalFlow-Agent-Project",
    port: int = 8080,
) -> bool:
    """Enable MLflow local tracing without auto-starting server.

    This function sets up MLflow tracing but does NOT automatically start an MLflow server.
    You need to have an MLflow server already running.

    To start an MLflow server:
    ```bash
    mlflow server --host 127.0.0.1 --port 8080
    ```

    Args:
        tracking_uri: MLflow tracking server URI. If None, defaults to http://localhost:{port}
        experiment_name: Name of the MLflow experiment to create/use
        project_name: Project name for the MLflow tracing processor
        port: Port for the default tracking URI if tracking_uri is None

    Returns:
        bool: True if MLflow was successfully enabled, False otherwise

    Example:
        >>> from adalflow.tracing import enable_mlflow_local
        >>> # Use with existing MLflow server on default port
        >>> enable_mlflow_local()
        >>> # Or with custom tracking URI:
        >>> enable_mlflow_local(tracking_uri="http://localhost:5000")
    """

    if not MLFLOW_AVAILABLE:
        log.error("MLflow is not installed. Cannot enable MLflow tracing.")
        return False

    try:
        # Set the environment variable BEFORE accessing the provider
        # This ensures the TraceProvider is created with tracing enabled
        os.environ["ADALFLOW_DISABLE_TRACING"] = "False"

        # Now access the provider - it will be created with tracing enabled
        GLOBAL_TRACE_PROVIDER.set_disabled(False)

        # Determine tracking URI
        if tracking_uri is None:
            # Default to localhost with specified port
            tracking_uri = f"http://localhost:{port}"

        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Create or set the experiment
        experiment = mlflow.set_experiment(experiment_name)
        log.info(
            f"MLflow experiment set: {experiment_name} (ID: {experiment.experiment_id})"
        )

        # Create MLflow processor
        mlflow_processor = MlflowOpenAgentTracingProcessor(project_name=project_name)
        log.info(f"MLflow processor created for project: {project_name}")

        # Replace all existing processors with MLflow
        set_trace_processors([mlflow_processor])

        log.info(
            f"✅ MLflow tracing enabled: tracking_uri={tracking_uri}, "
            f"experiment={experiment_name}, project={project_name}"
        )
        return True

    except Exception as e:
        log.error(f"Failed to enable MLflow tracing: {e}")
        return False


def enable_mlflow_local_with_server(
    tracking_uri: str = None,
    experiment_name: str = "AdalFlow-Agent-Experiment",
    project_name: str = "AdalFlow-Agent-Project",
    use_adalflow_dir: bool = True,
    port: int = 8080,
    kill_existing_server: bool = False,
) -> bool:
    """Enable MLflow local tracing with auto-starting server.

    This function sets up MLflow tracing and automatically starts an MLflow server if needed.

    Args:
        tracking_uri: MLflow tracking server URI. If None, will auto-start server
        experiment_name: Name of the MLflow experiment to create/use
        project_name: Project name for the MLflow tracing processor
        use_adalflow_dir: If True and server fails to start, use ~/.adalflow/mlruns as fallback
        port: Port to run MLflow server on
        kill_existing_server: If True, kill any existing MLflow server on the port before starting

    Returns:
        bool: True if MLflow was successfully enabled, False otherwise

    Example:
        >>> from adalflow.tracing import enable_mlflow_local_with_server
        >>> # Auto-start server on port 8080
        >>> enable_mlflow_local_with_server()
        >>> # Or kill existing and start fresh:
        >>> enable_mlflow_local_with_server(kill_existing_server=True)
        >>> # Or use custom port:
        >>> enable_mlflow_local_with_server(port=5000)
    """

    if not MLFLOW_AVAILABLE:
        log.error("MLflow is not installed. Cannot enable MLflow tracing.")
        return False

    try:
        # Set the environment variable BEFORE accessing the provider
        # This ensures the TraceProvider is created with tracing enabled
        os.environ["ADALFLOW_DISABLE_TRACING"] = "False"

        # Now access the provider - it will be created with tracing enabled
        GLOBAL_TRACE_PROVIDER.set_disabled(False)

        # Determine if we need to start a server
        if tracking_uri is None:
            # Try to start MLflow server with AdalFlow backend
            if start_mlflow_server(port=port, kill_existing=kill_existing_server):
                tracking_uri = f"http://localhost:{port}"
                log.info(f"Successfully started MLflow server on port {port}")
            else:
                log.warning(
                    "Failed to auto-start MLflow server, falling back to file backend"
                )
                if use_adalflow_dir:
                    # Use adalflow root dir/mlruns for local file storage
                    adalflow_dir = Path(get_adalflow_default_root_path()) / "mlruns"
                    adalflow_dir.mkdir(parents=True, exist_ok=True)
                    tracking_uri = str(adalflow_dir.absolute())
                    log.info(
                        f"Using AdalFlow directory for MLflow artifacts: {adalflow_dir}"
                    )
                else:
                    # Still try to use the localhost URL in case server is running externally
                    tracking_uri = f"http://localhost:{port}"

        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Create or set the experiment
        experiment = mlflow.set_experiment(experiment_name)
        log.info(
            f"MLflow experiment set: {experiment_name} (ID: {experiment.experiment_id})"
        )

        # Create MLflow processor
        mlflow_processor = MlflowOpenAgentTracingProcessor(project_name=project_name)
        log.info(f"MLflow processor created for project: {project_name}")

        # Replace all existing processors with MLflow
        set_trace_processors([mlflow_processor])

        log.info(
            f"✅ MLflow tracing enabled with server: tracking_uri={tracking_uri}, "
            f"experiment={experiment_name}, project={project_name}"
        )
        return True

    except Exception as e:
        log.error(f"Failed to enable MLflow tracing with server: {e}")
        return False


def get_mlflow_server_command(host: str = "0.0.0.0", port: int = 8080) -> str:
    """Get the MLflow server command with AdalFlow backend store.

    Args:
        host: Host to bind the server to (default: "0.0.0.0")
        port: Port to run the server on (default: 8080)

    Returns:
        str: The MLflow server command to run

    Example:
        >>> from adalflow.tracing import get_mlflow_server_command
        >>> print(get_mlflow_server_command())
        mlflow server --backend-store-uri file:///Users/yourname/.adalflow/mlruns --host 0.0.0.0 --port 8080
    """
    adalflow_dir = Path(get_adalflow_default_root_path()) / "mlruns"
    adalflow_dir.mkdir(parents=True, exist_ok=True)
    # Use absolute path without file:// prefix for the command
    backend_path = str(adalflow_dir.absolute())
    return (
        f"mlflow server --backend-store-uri {backend_path} --host {host} --port {port}"
    )


def start_mlflow_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    wait: bool = True,
    kill_existing: bool = False,
) -> bool:
    """Start MLflow server with AdalFlow backend store.

    Args:
        host: Host to bind the server to (default: "0.0.0.0")
        port: Port to run the server on (default: 8080)
        wait: If True, wait for server to be ready before returning
        kill_existing: If True, kill any existing process on the port before starting

    Returns:
        bool: True if server started successfully, False otherwise

    Example:
        >>> from adalflow.tracing import start_mlflow_server
        >>> start_mlflow_server()  # Starts server on http://0.0.0.0:8080
        >>> # Or with custom settings:
        >>> start_mlflow_server(port=5000)
    """
    if not MLFLOW_AVAILABLE:
        log.error("MLflow is not installed. Cannot start MLflow server.")
        return False

    import subprocess
    import time
    import urllib.request
    import signal

    try:
        import psutil

        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False

    # Check if server is already running
    try:
        with urllib.request.urlopen(f"http://localhost:{port}", timeout=1) as response:
            if response.status == 200:
                if kill_existing:
                    log.info(f"Killing existing MLflow server on port {port}")
                    # Find and kill processes using the port
                    if PSUTIL_AVAILABLE:
                        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                            try:
                                cmdline = proc.info.get("cmdline") or []
                                if "mlflow" in " ".join(cmdline) and str(
                                    port
                                ) in " ".join(cmdline):
                                    log.info(f"Killing MLflow process {proc.pid}")
                                    proc.kill()
                                    time.sleep(1)  # Give it time to shut down
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                    else:
                        # Alternative: use lsof on Unix-like systems
                        try:
                            result = subprocess.run(
                                ["lsof", "-ti", f":{port}"],
                                capture_output=True,
                                text=True,
                            )
                            if result.stdout:
                                pids = result.stdout.strip().split("\n")
                                for pid in pids:
                                    try:
                                        os.kill(int(pid), signal.SIGTERM)
                                    except (ProcessLookupError, ValueError, OSError):
                                        pass
                                time.sleep(1)
                        except Exception:
                            log.warning("Could not kill existing server")
                else:
                    log.info(f"MLflow server already running on port {port}")
                    return True
    except Exception:
        pass

    # Get server command
    cmd = get_mlflow_server_command(host, port)

    try:
        # Start server in background
        log.info(f"Starting MLflow server: {cmd}")
        subprocess.Popen(
            cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        if wait:
            # Wait for server to be ready (max 10 seconds)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen(
                        f"http://localhost:{port}", timeout=1
                    ) as response:
                        if response.status == 200:
                            log.info(
                                f"✅ MLflow server started successfully at http://{host}:{port}"
                            )
                            return True
                except Exception:
                    continue

            log.warning("MLflow server started but may not be ready yet")
        else:
            log.info(f"MLflow server starting in background at http://{host}:{port}")

        return True

    except Exception as e:
        log.error(f"Failed to start MLflow server: {e}")
        return False
