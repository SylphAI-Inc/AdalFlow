"""
CLI Permission Handler for testing tool approval flow.
"""

import asyncio
from typing import Optional
from adalflow.core.permission_manager import PermissionManager, ApprovalOutcome
from adalflow.core.types import Function, FunctionRequest

try:
    import aioconsole
    HAS_AIOCONSOLE = True
except ImportError:
    HAS_AIOCONSOLE = False
    print("Warning: aioconsole not installed. Install it with: pip install aioconsole")


class CLIPermissionHandler(PermissionManager):
    """
    A permission manager that prompts for approval via command line interface.
    Useful for testing and development.
    """
    
    def __init__(self, approval_mode: str = "default"):
        """
        Initialize CLI permission handler.
        
        Args:
            approval_mode: Mode for handling approvals ("default", "auto_approve", "yolo")
        """
        super().__init__(
            approval_callback=self._cli_approval_callback,
            approval_mode=approval_mode
        )
        
    async def _cli_approval_callback(self, request: FunctionRequest) -> ApprovalOutcome:
        """
        Prompt user for approval via CLI.
        
        Args:
            request: The function request
            
        Returns:
            The approval outcome
        """
        # Format the prompt
        print("\n" + "="*60)
        print("🔧 TOOL PERMISSION REQUEST")
        print("="*60)
        print(f"Tool: {request.tool_name}")
        
        if request.tool and request.tool.args:
            print(f"Arguments: {request.tool.args}")
        if request.tool and request.tool.kwargs:
            print(f"Keyword Arguments: {request.tool.kwargs}")
        
        print(f"\nAllow execution of '{request.tool_name}'?")
        print("\nOptions:")
        print("1. Allow once")
        print("2. Allow always for this tool")
        print("3. Cancel")
        print("4. Modify arguments (not implemented)")
        print("-"*60)
        
        # Get user input
        if HAS_AIOCONSOLE:
            choice = await aioconsole.ainput("Your choice (1-4): ")
        else:
            # Fallback to sync input with asyncio
            loop = asyncio.get_event_loop()
            choice = await loop.run_in_executor(None, input, "Your choice (1-4): ")
        
        # Process choice
        choice = choice.strip()
        
        if choice == "1":
            print("✅ Approved for this execution only\n")
            return ApprovalOutcome.PROCEED_ONCE
        elif choice == "2":
            print(f"✅ Tool '{request.tool_name}' will always be allowed\n")
            return ApprovalOutcome.PROCEED_ALWAYS
        elif choice == "3":
            print("❌ Execution cancelled\n")
            return ApprovalOutcome.CANCEL
        elif choice == "4":
            print("🔧 Modification not implemented yet, proceeding with original\n")
            return ApprovalOutcome.PROCEED_ONCE
        else:
            print("❌ Invalid choice, cancelling execution\n")
            return ApprovalOutcome.CANCEL


class AutoApprovalHandler(PermissionManager):
    """
    A permission manager that automatically approves all requests.
    Useful for testing without interruption.
    """
    
    def __init__(self):
        """Initialize auto approval handler."""
        super().__init__(
            approval_callback=self._auto_approval_callback,
            approval_mode="auto_approve"
        )
        
    async def _auto_approval_callback(self, request: FunctionRequest) -> ApprovalOutcome:
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
            name="search_web",
            args=["AI safety"],
            kwargs={"max_results": 10}
        )
        
        # Test permission check
        allowed, modified_func = await handler.check_permission(test_func)
        
        print(f"\nResult: allowed={allowed}")
        if modified_func:
            print(f"Modified function: {modified_func}")
            
    # Run test
    asyncio.run(test_cli_handler())