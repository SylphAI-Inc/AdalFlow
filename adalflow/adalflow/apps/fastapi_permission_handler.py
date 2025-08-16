"""
FastAPI Permission Handler for REST API-based tool approval flow.
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from adalflow.apps.permission_manager import (
    PermissionManager,
    ApprovalOutcome,
    ApprovalResponse as PermissionApprovalResponse,
)
from adalflow.core.types import FunctionRequest
import logging

log = logging.getLogger(__name__)


class ApprovalRequest(BaseModel):
    """API model for approval requests."""

    request_id: str
    tool_name: str
    tool_args: Optional[List] = []
    tool_kwargs: Optional[Dict] = {}
    timestamp: datetime
    status: str = "pending"  # pending, approved, rejected, expired


class ApprovalResponse(BaseModel):
    """API model for approval responses."""

    request_id: str
    outcome: str  # proceed_once, proceed_always, cancel
    response_data: Optional[str] = None  # Optional response data for tools like clarify


class ApprovalQueue:
    """Manages pending approval requests with optional timeout."""

    def __init__(self, timeout_seconds: Optional[int] = None):
        self.pending_requests: Dict[str, FunctionRequest] = {}
        self.responses: Dict[str, asyncio.Future] = {}
        self.request_metadata: Dict[str, ApprovalRequest] = {}
        self.timeout_seconds = timeout_seconds  # None means no timeout

    async def create_request(self, request: FunctionRequest) -> str:
        """Create a new approval request and return its ID."""
        request_id = request.id
        self.pending_requests[request_id] = request
        # Get the current event loop to ensure Future is created in the right context
        loop = asyncio.get_event_loop()
        self.responses[request_id] = loop.create_future()
        log.info(f"Created Future for request {request_id} in loop {loop}")

        # Store metadata for API responses
        self.request_metadata[request_id] = ApprovalRequest(
            request_id=request_id,
            tool_name=request.tool_name,
            tool_args=request.tool.args if request.tool else [],
            tool_kwargs=request.tool.kwargs if request.tool else {},
            timestamp=datetime.now(),
            status="pending",
        )

        return request_id

    async def wait_for_response(self, request_id: str) -> ApprovalOutcome:
        """Wait for approval response with optional timeout."""
        if request_id not in self.responses:
            raise ValueError("Invalid request ID")

        try:
            log.info(f"Starting to wait for Future for request {request_id} (timeout: {self.timeout_seconds or 'indefinite'})")
            
            if self.timeout_seconds is None:
                # Wait indefinitely for user approval
                result = await self.responses[request_id]
            else:
                # Wait with timeout
                result = await asyncio.wait_for(
                    self.responses[request_id], timeout=self.timeout_seconds
                )
            
            log.info(f"Future resolved for request {request_id} with result {result}")
            # Status already set in approve endpoint
            return result
        except asyncio.TimeoutError:
            log.warning(f"Request {request_id} timed out after {self.timeout_seconds} seconds")
            self.request_metadata[request_id].status = "expired"
            return ApprovalOutcome.CANCEL
        finally:
            # Clean up
            log.info(f"Cleaning up request {request_id}")
            self.pending_requests.pop(request_id, None)
            self.responses.pop(request_id, None)

    def set_response(
        self,
        request_id: str,
        outcome: ApprovalOutcome,
        response_data: Optional[str] = None,
    ):
        """Set the response for a pending request."""
        log.info(
            f"set_response called for {request_id} with outcome {outcome} and response_data: {response_data}"
        )
        if request_id not in self.responses:
            log.error(f"Request {request_id} not found in responses!")
        elif self.responses[request_id].done():
            log.error(f"Request {request_id} future is already done!")
        else:
            future = self.responses[request_id]
            loop = future.get_loop()
            log.info(f"Setting result for {request_id} on loop {loop}")
            # Create response with data if provided
            if response_data is not None:
                result = PermissionApprovalResponse(
                    outcome=outcome, response_data=response_data
                )
            else:
                result = outcome
            # Use call_soon_threadsafe to set the result in the correct event loop
            loop.call_soon_threadsafe(future.set_result, result)
            log.info(f"Result scheduled for {request_id}")

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        # Return only pending requests - they wait indefinitely for user approval
        return [
            metadata
            for metadata in self.request_metadata.values()
            if metadata.status == "pending"
        ]


class FastAPIPermissionHandler(PermissionManager):
    """
    A permission manager that provides REST API endpoints for approval.
    """

    def __init__(
        self,
        app: Optional[FastAPI] = None,
        approval_mode: str = "default",
        timeout_seconds: Optional[int] = None,
        api_prefix: str = "/api/v1/approvals",
    ):
        """
        Initialize FastAPI permission handler.

        Args:
            app: FastAPI application instance. If None, creates a new one.
            approval_mode: Mode for handling approvals
            timeout_seconds: Timeout for approval requests in seconds. None means no timeout.
            api_prefix: URL prefix for API endpoints
        """
        self.approval_queue = ApprovalQueue(timeout_seconds)
        self.api_prefix = api_prefix

        # Create or use existing FastAPI app
        self.app = app or FastAPI(
            title="Tool Approval API",
            description="REST API for approving tool executions",
            version="1.0.0",
        )

        # Register API endpoints
        self._register_endpoints()

        super().__init__(
            approval_callback=self._api_approval_callback, approval_mode=approval_mode
        )

    def _register_endpoints(self):
        """Register FastAPI endpoints for approval management."""

        @self.app.get(
            f"{self.api_prefix}/pending", response_model=List[ApprovalRequest]
        )
        async def get_pending_approvals():
            """Get all pending approval requests."""
            return self.approval_queue.get_pending_requests()

        @self.app.get(
            f"{self.api_prefix}/{{request_id}}", response_model=ApprovalRequest
        )
        async def get_approval_request(request_id: str):
            """Get a specific approval request."""
            if request_id not in self.approval_queue.request_metadata:
                raise HTTPException(status_code=404, detail="Request not found")
            return self.approval_queue.request_metadata[request_id]

        @self.app.post(f"{self.api_prefix}/{{request_id}}/approve")
        async def approve_request(request_id: str, response: ApprovalResponse):
            """Approve or reject a tool execution request."""
            log.info(
                f"Approval endpoint called with URL request_id={request_id}, body={response}"
            )
            log.info(f"Current queue state - Pending requests: {list(self.approval_queue.pending_requests.keys())}")
            log.info(f"Current queue state - Response futures: {list(self.approval_queue.responses.keys())}")
            log.info(f"Current queue state - Metadata: {list(self.approval_queue.request_metadata.keys())}")
            
            if request_id not in self.approval_queue.request_metadata:
                log.error(
                    f"Request {request_id} not found in metadata. Available: {list(self.approval_queue.request_metadata.keys())}"
                )
                raise HTTPException(status_code=404, detail="Request not found")

            if self.approval_queue.request_metadata[request_id].status != "pending":
                raise HTTPException(
                    status_code=400,
                    detail=f"Request is no longer pending (status: {self.approval_queue.request_metadata[request_id].status})",
                )

            # Map string outcome to enum
            outcome_map = {
                "proceed_once": ApprovalOutcome.PROCEED_ONCE,
                "proceed_always": ApprovalOutcome.PROCEED_ALWAYS,
                "cancel": ApprovalOutcome.CANCEL,
                # "modify": ApprovalOutcome.MODIFY  # TODO: Implement modify functionality in the future
            }

            if response.outcome not in outcome_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid outcome. Must be one of: {list(outcome_map.keys())}",
                )

            outcome = outcome_map[response.outcome]

            # Set the response immediately (not in background)
            log.info(
                f"About to set response for {request_id} with outcome {outcome} and response_data: {response.response_data}"
            )
            # Update status immediately
            self.approval_queue.request_metadata[request_id].status = "approved"
            self.approval_queue.set_response(
                request_id, outcome, response.response_data
            )

            return {
                "status": "accepted",
                "request_id": request_id,
                "outcome": response.outcome,
                "response_data": response.response_data,
            }

        @self.app.delete(f"{self.api_prefix}/{{request_id}}")
        async def cancel_request(request_id: str):
            """Cancel a pending approval request."""
            if request_id not in self.approval_queue.request_metadata:
                raise HTTPException(status_code=404, detail="Request not found")

            self.approval_queue.set_response(request_id, ApprovalOutcome.CANCEL)
            self.approval_queue.request_metadata[request_id].status = "cancelled"

            return {"status": "cancelled", "request_id": request_id}

        @self.app.get(f"{self.api_prefix}/stats")
        async def get_approval_stats():
            """Get statistics about approval requests."""
            metadata = self.approval_queue.request_metadata

            stats = {
                "total_requests": len(metadata),
                "pending": len([m for m in metadata.values() if m.status == "pending"]),
                "approved": len(
                    [m for m in metadata.values() if m.status == "approved"]
                ),
                "rejected": len(
                    [m for m in metadata.values() if m.status == "rejected"]
                ),
                "expired": len([m for m in metadata.values() if m.status == "expired"]),
                "always_allowed_tools": list(self.always_allowed_tools),
                "blocked_tools": list(self.blocked_tools),
            }

            return stats

    async def _api_approval_callback(self, request: FunctionRequest) -> ApprovalOutcome:
        """
        Handle approval via REST API.

        Args:
            request: The function request

        Returns:
            The approval outcome
        """
        # Create request in queue
        request_id = await self.approval_queue.create_request(request)

        log.info(
            f"Created approval request {request_id} for tool '{request.tool_name}'"
        )
        log.info(f"Waiting for approval at: GET {self.api_prefix}/pending")

        # Wait for response
        log.info(f"Waiting for response for request {request_id}...")
        outcome = await self.approval_queue.wait_for_response(request_id)

        log.info(f"Received approval outcome for {request_id}: {outcome}")
        log.info(f"About to return outcome {outcome} from _api_approval_callback")

        return outcome


def create_standalone_app() -> FastAPI:
    """
    Create a standalone FastAPI app with approval endpoints.

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="AdalFlow Tool Approval API",
        description="Standalone REST API for approving AdalFlow tool executions",
        version="1.0.0",
    )

    # Add health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "tool-approval-api"}

    return app
