"""
CLI Permission Handler for testing tool approval flow.
"""

import asyncio
from typing import Optional
from adalflow.apps.permission_manager import PermissionManager, ApprovalOutcome
from adalflow.core.types import Function, FunctionRequest
import threading
import queue


class CLIPermissionHandler(PermissionManager):
    """
    A permission manager that prompts for approval via command line interface.
    Useful for testing and development.
    """

    def __init__(self, approval_mode: str = "default", timeout: float = 30.0):
        """
        Initialize CLI permission handler.

        Args:
            approval_mode: Mode for handling approvals ("default", "auto_approve", "yolo")
            timeout: Timeout for user input in seconds
        """
        super().__init__(
            approval_callback=self._cli_approval_callback, approval_mode=approval_mode
        )
        self.timeout = timeout
        self._output_queue = queue.Queue()
        self._output_thread = None
        self._start_output_thread()

    def _start_output_thread(self):
        """Start a separate thread for handling output."""
        if self._output_thread is None or not self._output_thread.is_alive():
            self._output_thread = threading.Thread(target=self._output_worker, daemon=True)
            self._output_thread.start()

    def _output_worker(self):
        """Worker thread that handles all output operations."""
        while True:
            try:
                msg = self._output_queue.get(timeout=1)
                if msg is None:  # Shutdown signal
                    break
                print(msg, flush=True)
            except queue.Empty:
                continue
            except Exception:
                # Silently ignore output errors
                pass

    def _safe_print(self, text: str):
        """Queue text for output without blocking."""
        try:
            self._output_queue.put_nowait(text)
        except queue.Full:
            # If queue is full, skip this output
            pass

    def _truncate_value(self, value, max_length: int = 200) -> str:
        """Truncate large values to prevent I/O blocking."""
        if value is None:
            return "None"
        
        str_value = str(value)
        if len(str_value) > max_length:
            return f"{str_value[:max_length]}... (truncated)"
        return str_value

    async def _cli_approval_callback(self, request: FunctionRequest) -> ApprovalOutcome:
        """
        Prompt user for approval via CLI.

        Args:
            request: The function request

        Returns:
            The approval outcome
        """
        try:
            # Format the prompt using safe print
            self._safe_print("\n" + "=" * 60)
            self._safe_print("🔧 TOOL PERMISSION REQUEST")
            self._safe_print("=" * 60)
            self._safe_print(f"Tool: {request.tool_name}")

            # Truncate args and kwargs to prevent blocking on large output
            if request.tool and request.tool.args:
                args_str = self._truncate_value(request.tool.args)
                self._safe_print(f"Arguments: {args_str}")
            if request.tool and request.tool.kwargs:
                # For kwargs, show keys with truncated values
                self._safe_print("Keyword Arguments:")
                for key, value in request.tool.kwargs.items():
                    value_str = self._truncate_value(value, max_length=100)
                    self._safe_print(f"  {key}: {value_str}")

            self._safe_print(f"\nAllow execution of '{request.tool_name}'?")
            self._safe_print("\nOptions:")
            self._safe_print("1. Allow once")
            self._safe_print("2. Allow always for this tool")
            self._safe_print("3. Cancel")
            self._safe_print("-" * 60)
            self._safe_print("Your choice (1-3): ")

            # Get user input using executor to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                # Run input in executor with timeout
                future = loop.run_in_executor(None, input, "")
                choice = await asyncio.wait_for(future, timeout=self.timeout)
            except asyncio.TimeoutError:
                self._safe_print(f"\n⏱️ Input timeout after {self.timeout} seconds")
                return ApprovalOutcome.CANCEL
            except Exception as e:
                self._safe_print(f"\n❌ Input error: {e}")
                return ApprovalOutcome.CANCEL

            # Process choice
            choice = choice.strip()

            if choice == "1":
                self._safe_print("✅ Approved for this execution only\n")
                return ApprovalOutcome.PROCEED_ONCE
            elif choice == "2":
                self._safe_print(f"✅ Tool '{request.tool_name}' will always be allowed\n")
                return ApprovalOutcome.PROCEED_ALWAYS
            elif choice == "3":
                self._safe_print("❌ Execution cancelled\n")
                return ApprovalOutcome.CANCEL
            else:
                self._safe_print("❌ Invalid choice, cancelling execution\n")
                return ApprovalOutcome.CANCEL

        except Exception as e:
            self._safe_print(f"❌ Permission handler error: {e}")
            return ApprovalOutcome.CANCEL

    def __del__(self):
        """Cleanup the output thread on deletion."""
        if hasattr(self, '_output_queue'):
            self._output_queue.put(None)  # Signal thread to stop


class AutoApprovalHandler(PermissionManager):
    """
    A permission manager that automatically approves all requests.
    Useful for testing without interruption.
    """

    def __init__(self):
        """Initialize auto approval handler."""
        super().__init__(
            approval_callback=self._auto_approval_callback, approval_mode="auto_approve"
        )

    async def _auto_approval_callback(
        self, request: FunctionRequest
    ) -> ApprovalOutcome:
        """Automatically approve all requests."""
        print(f"🤖 Auto-approving tool: {request.tool_name}")
        return ApprovalOutcome.PROCEED_ONCE


def create_test_permission_manager(mode: str = "cli") -> Optional[PermissionManager]:
    """
    Create a permission manager for testing.

    Args:
        mode: The mode to use ("cli", "auto", "none")

    Returns:
        A permission manager instance or None
    """
    if mode == "cli":
        return CLIPermissionHandler()
    elif mode == "auto":
        return AutoApprovalHandler()
    elif mode == "none":
        return None
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Example usage for testing
if __name__ == "__main__":

    async def test_cli_handler():
        # Create handler
        handler = CLIPermissionHandler()

        # Create a test function
        test_func = Function(
            name="search_web", args=["AI safety"], kwargs={"max_results": 10}
        )

        # Test permission check
        allowed, modified_func = await handler.check_permission(test_func)

        print(f"\nResult: allowed={allowed}")
        if modified_func:
            print(f"Modified function: {modified_func}")

    # Run test
    asyncio.run(test_cli_handler())
