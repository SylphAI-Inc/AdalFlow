"""
Permission Manager for tool execution approval in AdalFlow.
"""

from typing import Dict, Set, Optional, Callable, Any, Awaitable, Union
from enum import Enum
import asyncio
import logging

from adalflow.core.types import Function, FunctionRequest, ToolCallPermissionRequest

log = logging.getLogger(__name__)


class ApprovalOutcome(Enum):
    """Possible outcomes for tool approval requests."""
    PROCEED_ONCE = "proceed_once"
    PROCEED_ALWAYS = "proceed_always"
    CANCEL = "cancel"
    MODIFY = "modify"


class PermissionManager:
    """
    Manages tool execution permissions and approval flow.
    
    This manager can be configured with different approval callbacks
    to support various UI patterns (CLI, WebSocket, REST, etc).
    """
    
    def __init__(
        self,
        approval_callback: Optional[Callable[[FunctionRequest], Awaitable[ApprovalOutcome]]] = None,
        approval_mode: str = "default",  # "default", "auto_approve", "yolo"
    ):
        """
        Initialize the permission manager.
        
        Args:
            approval_callback: Async function to call for approval requests
            approval_mode: Mode for handling approvals
        """
        self.approval_callback = approval_callback
        self.approval_mode = approval_mode
        self.always_allowed_tools: Set[str] = set()
        self.blocked_tools: Set[str] = set()
        self.pending_approvals: Dict[str, FunctionRequest] = {}
        self.tool_require_approval: Dict[str, bool] = {}  # Track which tools require approval
        
    def register_tool(self, tool_name: str, require_approval: bool = True):
        """Register a tool and whether it requires approval."""
        self.tool_require_approval[tool_name] = require_approval
        log.debug(f"Registered tool '{tool_name}' with require_approval={require_approval}")
        
    def is_approval_required(self, tool_name: str) -> bool:
        """Check if a tool requires approval."""
        # Check approval mode first
        if self.approval_mode == "yolo" or self.approval_mode == "auto_approve":
            return False
            
        # Check if tool is in always allowed list
        if tool_name in self.always_allowed_tools:
            return False
            
        # Check if tool is blocked
        if tool_name in self.blocked_tools:
            return True
            
        # Check tool-specific setting (default to True if not registered)
        return self.tool_require_approval.get(tool_name, True)
        
    async def check_permission(self, func: Function) -> tuple[bool, Optional[Function]]:
        """
        Check if the function/tool execution is permitted.
        
        Args:
            func: The function to check permission for
            
        Returns:
            Tuple of (is_allowed, modified_function)
        """
        tool_name = func.name
        
        # Quick checks that don't require approval
        if not self.is_approval_required(tool_name):
            return True, func
            
        if tool_name in self.blocked_tools:
            log.info(f"Tool '{tool_name}' is blocked")
            return False, None
            
        # No callback means default allow
        if not self.approval_callback:
            log.warning(f"No approval callback set, allowing tool '{tool_name}' by default")
            return True, func
            
        # Create function request using the types from types.py
        request = FunctionRequest(
            tool_name=tool_name,
            tool=func
        )
        
        # Store pending request
        self.pending_approvals[request.id] = request
        
        try:
            # Call approval callback
            outcome = await self.approval_callback(request)
            
            # Handle outcome
            if outcome == ApprovalOutcome.PROCEED_ALWAYS:
                self.always_allowed_tools.add(tool_name)
                log.info(f"Tool '{tool_name}' added to always allowed list")
                return True, func
            elif outcome == ApprovalOutcome.PROCEED_ONCE:
                log.info(f"Tool '{tool_name}' approved for single execution")
                return True, func
            elif outcome == ApprovalOutcome.MODIFY:
                # In a real implementation, this would involve editing the function
                # For now, just proceed with original
                log.info(f"Tool '{tool_name}' modification requested (not implemented)")
                return True, func
            else:  # CANCEL
                log.info(f"Tool '{tool_name}' execution cancelled by user")
                return False, None
                
        finally:
            # Clean up pending request
            self.pending_approvals.pop(request.id, None)
            
    def add_to_always_allowed(self, tool_name: str):
        """Add a tool to the always allowed list."""
        self.always_allowed_tools.add(tool_name)
        
    def add_to_blocked(self, tool_name: str):
        """Add a tool to the blocked list."""
        self.blocked_tools.add(tool_name)
        
    def remove_from_always_allowed(self, tool_name: str):
        """Remove a tool from the always allowed list."""
        self.always_allowed_tools.discard(tool_name)
        
    def remove_from_blocked(self, tool_name: str):
        """Remove a tool from the blocked list."""
        self.blocked_tools.discard(tool_name)
        
    def get_pending_approvals(self) -> Dict[str, FunctionRequest]:
        """Get all pending approval requests."""
        return self.pending_approvals.copy()
        
    def create_permission_event(self, func: Function) -> ToolCallPermissionRequest:
        """Create a permission request event for streaming."""
        request = FunctionRequest(
            tool_name=func.name,
            tool=func
        )
        return ToolCallPermissionRequest(data=request)