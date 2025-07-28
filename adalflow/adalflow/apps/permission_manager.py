"""
Permission Manager for tool execution approval in AdalFlow.
"""

from typing import Dict, Set, Optional, Callable, Awaitable, Union
from enum import Enum
from dataclasses import dataclass
import logging

from adalflow.core.types import Function, FunctionRequest, ToolCallPermissionRequest, ToolOutput
from adalflow.core.tool_manager import ToolManager, FunctionTool


log = logging.getLogger(__name__)


class ApprovalOutcome(Enum):
    """Possible outcomes for tool approval requests."""

    PROCEED_ONCE = "proceed_once"
    PROCEED_ALWAYS = "proceed_always"
    CANCEL = "cancel"


@dataclass
class ApprovalResponse:
    """Response from approval callback including outcome and optional data."""

    outcome: ApprovalOutcome
    response_data: Optional[str] = None


class PermissionManager:
    __doc__ = """Manages tool execution permissions and approval flow.

    This manager can be configured with different approval callbacks
    to support various UI patterns (CLI, WebSocket, REST, etc).

    Users will register tools into three categories:
    - always_allowed_tools: Tools that are always allowed to execute.
    - blocked_tools: Tools that are blocked and cannot execute.
    - tool_require_approval: Tools that require explicit approval before execution.
     (e.g. delete_file, send_email)

    If tool is not registered, it will default to False (no approval required).

    The approval_mode parameter controls how tool permissions are handled:

    - "default": respects all three categories.

    - "auto_approve": approves tool_require_approval tools automatically.
    Still respects the blocked_tools list. Useful for trusted environments.

    - "yolo" (Yes, Only Live Once): Bypasses all categories entirely.
    Maximum performance but no safety checks. Use only in development or
    fully trusted environments.

    For the PermissionManager to be function, see exampels in /apps
    * cli_permission_handler.py
    * fastapi_permission_handler.py
    Where the approval_callback is implemented.
    """

    def __init__(
        self,
        approval_callback: Optional[
            Callable[
                [FunctionRequest], Awaitable[Union[ApprovalOutcome, ApprovalResponse]]
            ]
        ] = None,
        approval_mode: str = "default",  # "default", "auto_approve", "yolo"
        tool_manager: Optional[ToolManager] = None,
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
        self.tool_require_approval: Dict[str, bool] = (
            {}
        )  # Track which tools require approval
        self.tool_manager = tool_manager

    def set_tool_manager(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager

    def register_tool(self, tool_name: str, require_approval: bool = True):
        """Register a tool and whether it requires approval."""
        self.tool_require_approval[tool_name] = require_approval
        log.debug(
            f"Registered tool '{tool_name}' with require_approval={require_approval}"
        )

    def is_approval_required(self, tool_name: str) -> bool:
        """Check if a tool requires approval."""
        # Yolo mode bypasses all checks
        if self.approval_mode == "yolo":
            return False

        # Always check blocked tools first (in all modes)
        if tool_name in self.blocked_tools:
            return True

        # Always check always_allowed tools (in all modes)
        if tool_name in self.always_allowed_tools:
            return False

        # Mode-specific logic for remaining tools
        if self.approval_mode == "auto_approve":
            return False  # Auto-approve everything not blocked/always-allowed

        # Default mode: check tool-specific setting
        # Default to False (no approval required) for unregistered tools
        return self.tool_require_approval.get(tool_name, False)

    async def check_permission(
        self, func: Function
    ) -> tuple[bool, Optional[Function], Optional[str]]:
        """
        Check if the function/tool execution is permitted.

        Args:
            func: The function to check permission for

        Returns:
            Tuple of (is_allowed, modified_function, response_data)
        """
        tool_name = func.name

        # Check if approval is required using our centralized logic
        if not self.is_approval_required(tool_name):
            return True, func, None

        # If approval is required but no callback is set, default to allow with warning
        if not self.approval_callback:
            log.warning(
                f"No approval callback set, allowing tool '{tool_name}' by default"
            )
            return True, func, None

        # Ensure to use the same id as the func
        request = FunctionRequest(id=func.id, tool_name=tool_name, tool=func)

        self.pending_approvals[request.id] = request

        try:
            # Interacts with user to get approval
            # the frontend need to display the request and wait for the user to respond
            result = await self.approval_callback(request)

            # Handle both old style (just outcome) and new style (with response data)
            if isinstance(result, ApprovalOutcome):
                outcome = result
                response_data = None
            else:  # ApprovalResponse
                outcome = result.outcome
                response_data = result.response_data

            # Handle outcome
            if outcome == ApprovalOutcome.PROCEED_ALWAYS:
                self.always_allowed_tools.add(tool_name)
                log.info(f"Tool '{tool_name}' added to always allowed list")
                return True, func, response_data
            elif outcome == ApprovalOutcome.PROCEED_ONCE:
                log.info(f"Tool '{tool_name}' approved for single execution")

                # TODO: avoid using function name, but more of function type,
                # or add "clarify" function as a default function, and ensure
                # this check can be disabled by configuration

                # For clarify tool, inject response data into function kwargs
                if response_data and tool_name == "clarify":
                    modified_kwargs = {**func.kwargs, "user_response": response_data}
                    modified_func = Function(
                        id=func.id,
                        name=func.name,
                        args=func.args,
                        kwargs=modified_kwargs,
                        thought=func.thought,
                    )
                    return True, modified_func, response_data

                return True, func, response_data
            else:  # CANCEL
                log.info(f"Tool '{tool_name}' execution cancelled by user")
                return False, None, response_data

        except Exception as e:
            log.error(f"Error during approval callback for tool '{tool_name}': {e}")
            # Default to deny on error for safety
            return False, None, None
        finally:
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

    def create_permission_event(self, func: Function) -> Union[ToolCallPermissionRequest, ToolOutput]:
        """Create a permission request event for streaming. Now used only for streaming. Only streaming supports human in the loop"""
        request = FunctionRequest(id=func.id, tool_name=func.name, tool=func)
        function_tool: FunctionTool = self.tool_manager.get_function_tool_by_name(func.name)
        # TODO: validate the tool, and if tool does not exits, how should we process it here?
        if function_tool is None:
            log.warning(f"Tool '{func.name}' not found in tool manager")
            # defer the error to the later step
            return ToolCallPermissionRequest(data=request) 

        if function_tool.pre_execute_callback is not None:
            # pass the func.parameters to execute the callback (ensure only sync function for now)
            # TODO: wrap the pre_execute_call
            result = function_tool.pre_execute_callback(**func.kwargs)

            # check if there is an error field 
            error = None
            if hasattr(result, "error") and result.error is not None:
                error = result.error
                # we need to output a ToolOutput (pre_execute_error)
                log.error(f"Pre execute callback result: {result}")
                return ToolOutput(output=error, observation = error, status="error")
            # we need to present user the error 
            if result is not None:
                request.confirmation_details = result            
        return ToolCallPermissionRequest(data=request)
