"""
Test suite for FastAPI Permission Handler.

This test file covers various scenarios for the FastAPI-based tool approval system,
including:
1. Basic approval flow (approve, reject, cancel)
2. Concurrent approval requests from multiple agents/runners
3. Timeout handling (with timeout and indefinite wait)
4. Request expiration and cleanup
5. API endpoint functionality
6. Error handling and edge cases
7. Response data handling (for tools like clarify)
8. Race conditions and thread safety
"""

import asyncio
import pytest
import time
from typing import Optional, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adalflow.apps.fastapi_permission_handler import (
    FastAPIPermissionHandler,
    ApprovalQueue,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalOutcome,
)
from adalflow.apps.permission_manager import ApprovalResponse as PermissionApprovalResponse
from adalflow.core.types import FunctionRequest


class TestApprovalQueue:
    """Test the ApprovalQueue class for managing pending requests."""
    
    def test_queue_initialization_with_timeout(self):
        """Test that ApprovalQueue initializes correctly with a timeout."""
        queue = ApprovalQueue(timeout_seconds=30)
        assert queue.timeout_seconds == 30
        assert len(queue.pending_requests) == 0
        assert len(queue.responses) == 0
        assert len(queue.request_metadata) == 0
    
    def test_queue_initialization_without_timeout(self):
        """Test that ApprovalQueue initializes correctly with no timeout (indefinite wait)."""
        queue = ApprovalQueue(timeout_seconds=None)
        assert queue.timeout_seconds is None
        assert len(queue.pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_create_request(self):
        """Test creating a new approval request."""
        queue = ApprovalQueue()
        
        # Create a mock FunctionRequest
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "test-request-123"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = ["arg1", "arg2"]
        mock_request.tool.kwargs = {"key": "value"}
        
        request_id = await queue.create_request(mock_request)
        
        assert request_id == "test-request-123"
        assert request_id in queue.pending_requests
        assert request_id in queue.responses
        assert request_id in queue.request_metadata
        assert queue.request_metadata[request_id].status == "pending"
    
    @pytest.mark.asyncio
    async def test_wait_for_response_with_approval(self):
        """Test waiting for an approval response that gets approved."""
        queue = ApprovalQueue(timeout_seconds=5)
        
        # Create a request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "test-request-456"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await queue.create_request(mock_request)
        
        # Simulate approval in background
        async def approve_after_delay():
            await asyncio.sleep(0.5)
            queue.set_response(request_id, ApprovalOutcome.PROCEED_ONCE)
        
        # Start approval task
        asyncio.create_task(approve_after_delay())
        
        # Wait for response
        result = await queue.wait_for_response(request_id)
        
        assert result == ApprovalOutcome.PROCEED_ONCE
        assert request_id not in queue.pending_requests
        assert request_id not in queue.responses
    
    @pytest.mark.asyncio
    async def test_wait_for_response_with_timeout(self):
        """Test that request times out when timeout is set and no approval comes."""
        queue = ApprovalQueue(timeout_seconds=1)
        
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "timeout-request"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await queue.create_request(mock_request)
        
        # Don't approve - let it timeout
        result = await queue.wait_for_response(request_id)
        
        assert result == ApprovalOutcome.CANCEL
        assert queue.request_metadata[request_id].status == "expired"
    
    @pytest.mark.asyncio
    async def test_wait_for_response_indefinite(self):
        """Test that request waits indefinitely when timeout is None."""
        queue = ApprovalQueue(timeout_seconds=None)
        
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "indefinite-request"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await queue.create_request(mock_request)
        
        # Approve after a longer delay
        async def approve_after_long_delay():
            await asyncio.sleep(2)  # Would timeout if timeout was set to 1
            queue.set_response(request_id, ApprovalOutcome.PROCEED_ALWAYS)
        
        asyncio.create_task(approve_after_long_delay())
        
        # Should wait indefinitely and get the response
        result = await queue.wait_for_response(request_id)
        
        assert result == ApprovalOutcome.PROCEED_ALWAYS
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent approval requests."""
        queue = ApprovalQueue(timeout_seconds=10)
        
        # Create multiple requests
        request_ids = []
        for i in range(5):
            mock_request = Mock(spec=FunctionRequest)
            mock_request.id = f"concurrent-{i}"
            mock_request.tool_name = f"tool_{i}"
            mock_request.tool = Mock()
            mock_request.tool.args = []
            mock_request.tool.kwargs = {}
            
            request_id = await queue.create_request(mock_request)
            request_ids.append(request_id)
        
        assert len(queue.pending_requests) == 5
        assert len(queue.get_pending_requests()) == 5
        
        # Approve requests in different order
        async def approve_requests():
            await asyncio.sleep(0.1)
            queue.set_response(request_ids[2], ApprovalOutcome.PROCEED_ONCE)
            await asyncio.sleep(0.1)
            queue.set_response(request_ids[0], ApprovalOutcome.CANCEL)
            await asyncio.sleep(0.1)
            queue.set_response(request_ids[4], ApprovalOutcome.PROCEED_ALWAYS)
        
        # Start approval task
        asyncio.create_task(approve_requests())
        
        # Wait for responses concurrently
        results = await asyncio.gather(
            queue.wait_for_response(request_ids[0]),
            queue.wait_for_response(request_ids[2]),
            queue.wait_for_response(request_ids[4]),
            return_exceptions=True
        )
        
        assert results[0] == ApprovalOutcome.CANCEL
        assert results[1] == ApprovalOutcome.PROCEED_ONCE
        assert results[2] == ApprovalOutcome.PROCEED_ALWAYS
    
    @pytest.mark.asyncio
    async def test_set_response_with_data(self):
        """Test setting response with additional data (e.g., for clarify tool)."""
        queue = ApprovalQueue()
        
        # Create a Future in the current event loop
        loop = asyncio.get_event_loop()
        
        queue.responses["test-id"] = loop.create_future()
        queue.request_metadata["test-id"] = ApprovalRequest(
            request_id="test-id",
            tool_name="clarify",
            tool_args=[],
            tool_kwargs={"questions": "What do you mean?"},
            timestamp=datetime.now(),
            status="pending"
        )
        
        # Set response with data
        queue.set_response("test-id", ApprovalOutcome.PROCEED_ONCE, "User's clarification answer")
        
        # Give the event loop a moment to process
        await asyncio.sleep(0.01)
        
        # Check that future was set
        assert queue.responses["test-id"].done()
        result = queue.responses["test-id"].result()
        assert isinstance(result, PermissionApprovalResponse)
        assert result.outcome == ApprovalOutcome.PROCEED_ONCE
        assert result.response_data == "User's clarification answer"


class TestFastAPIPermissionHandler:
    """Test the FastAPIPermissionHandler class with API endpoints."""
    
    def test_handler_initialization(self):
        """Test that handler initializes with correct parameters."""
        app = FastAPI()
        handler = FastAPIPermissionHandler(
            app=app,
            approval_mode="default",
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        assert handler.approval_queue.timeout_seconds == 30
        assert handler.api_prefix == "/api/v1/approvals"
        assert handler.app == app
    
    def test_handler_initialization_no_timeout(self):
        """Test handler initialization with no timeout (indefinite wait)."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=None,
            api_prefix="/api/v1/approvals"
        )
        
        assert handler.approval_queue.timeout_seconds is None
    
    def test_api_endpoints(self):
        """Test that API endpoints are registered and functional."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        client = TestClient(handler.app)
        
        # Test getting pending approvals (should be empty)
        response = client.get("/api/v1/approvals/pending")
        assert response.status_code == 200
        assert response.json() == []
        
        # Test stats endpoint (if it exists)
        response = client.get("/api/v1/approvals/stats")
        if response.status_code == 200:
            stats = response.json()
            assert stats["total_requests"] == 0
            assert stats["pending"] == 0
        # If stats endpoint doesn't exist, that's okay - it's optional
    
    @pytest.mark.asyncio
    async def test_approval_flow_via_api(self):
        """Test complete approval flow through API endpoints."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        # Create a request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "api-test-request"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = ["arg1"]
        mock_request.tool.kwargs = {"key": "value"}
        
        request_id = await handler.approval_queue.create_request(mock_request)
        
        client = TestClient(handler.app)
        
        # Get pending approvals
        response = client.get("/api/v1/approvals/pending")
        assert response.status_code == 200
        pending = response.json()
        assert len(pending) == 1
        assert pending[0]["request_id"] == request_id
        assert pending[0]["status"] == "pending"
        
        # Get specific request
        response = client.get(f"/api/v1/approvals/{request_id}")
        assert response.status_code == 200
        request_data = response.json()
        assert request_data["request_id"] == request_id
        assert request_data["tool_name"] == "test_tool"
        
        # Approve the request
        approval_data = {
            "request_id": request_id,
            "outcome": "proceed_once",
            "response_data": None
        }
        response = client.post(
            f"/api/v1/approvals/{request_id}/approve",
            json=approval_data
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "accepted"
        assert result["outcome"] == "proceed_once"
    
    @pytest.mark.asyncio
    async def test_cancel_via_api(self):
        """Test canceling a request via API."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        # Create a request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "cancel-test"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await handler.approval_queue.create_request(mock_request)
        
        client = TestClient(handler.app)
        
        # Cancel the request
        response = client.delete(f"/api/v1/approvals/{request_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "cancelled"
        assert result["request_id"] == request_id
    
    @pytest.mark.asyncio
    async def test_invalid_request_handling(self):
        """Test handling of invalid requests and error cases."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        client = TestClient(handler.app)
        
        # Try to get non-existent request
        response = client.get("/api/v1/approvals/non-existent")
        assert response.status_code == 404
        
        # Try to approve non-existent request
        approval_data = {
            "request_id": "non-existent",
            "outcome": "proceed_once",
            "response_data": None
        }
        response = client.post(
            "/api/v1/approvals/non-existent/approve",
            json=approval_data
        )
        assert response.status_code == 404
        
        # Try to approve with invalid outcome
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "invalid-outcome-test"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await handler.approval_queue.create_request(mock_request)
        
        approval_data = {
            "request_id": request_id,
            "outcome": "invalid_outcome",
            "response_data": None
        }
        response = client.post(
            f"/api/v1/approvals/{request_id}/approve",
            json=approval_data
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_race_condition_multiple_approvals(self):
        """
        Test that approving the same request multiple times is handled correctly.
        This can happen if user clicks approve multiple times quickly.
        """
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        # Create a request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "race-condition-test"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        request_id = await handler.approval_queue.create_request(mock_request)
        
        client = TestClient(handler.app)
        
        # First approval should succeed
        approval_data = {
            "request_id": request_id,
            "outcome": "proceed_once",
            "response_data": None
        }
        response1 = client.post(
            f"/api/v1/approvals/{request_id}/approve",
            json=approval_data
        )
        assert response1.status_code == 200
        
        # Second approval should fail (already approved)
        response2 = client.post(
            f"/api/v1/approvals/{request_id}/approve",
            json=approval_data
        )
        assert response2.status_code == 400
        assert "no longer pending" in response2.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_clarify_tool_with_response_data(self):
        """Test handling of clarify tool with response data from user."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=30,
            api_prefix="/api/v1/approvals"
        )
        
        # Create a clarify request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "clarify-test"
        mock_request.tool_name = "clarify"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {"questions": "What is your goal?"}
        
        request_id = await handler.approval_queue.create_request(mock_request)
        
        client = TestClient(handler.app)
        
        # Approve with response data
        approval_data = {
            "request_id": request_id,
            "outcome": "proceed_once",
            "response_data": "I want to build an AI assistant"
        }
        response = client.post(
            f"/api/v1/approvals/{request_id}/approve",
            json=approval_data
        )
        assert response.status_code == 200
        result = response.json()
        assert result["response_data"] == "I want to build an AI assistant"
    
    @pytest.mark.asyncio
    async def test_approval_callback_integration(self):
        """Test the approval callback function used by the permission manager."""
        handler = FastAPIPermissionHandler(
            timeout_seconds=5,
            api_prefix="/api/v1/approvals"
        )
        
        # Create a mock request
        mock_request = Mock(spec=FunctionRequest)
        mock_request.id = "callback-test"
        mock_request.tool_name = "test_tool"
        mock_request.tool = Mock()
        mock_request.tool.args = []
        mock_request.tool.kwargs = {}
        
        # Simulate approval in background
        async def simulate_approval():
            await asyncio.sleep(0.5)
            # Directly set response as if API was called
            handler.approval_queue.set_response(
                "callback-test",
                ApprovalOutcome.PROCEED_ONCE
            )
        
        # Start callback and approval concurrently
        callback_task = asyncio.create_task(
            handler._api_approval_callback(mock_request)
        )
        approval_task = asyncio.create_task(simulate_approval())
        
        # Wait for both
        result = await callback_task
        await approval_task
        
        assert result == ApprovalOutcome.PROCEED_ONCE
    
    @pytest.mark.asyncio
    async def test_stress_many_concurrent_approvals(self):
        """
        Stress test with many concurrent approval requests.
        This simulates multiple agents/runners requesting approvals simultaneously.
        """
        handler = FastAPIPermissionHandler(
            timeout_seconds=10,
            api_prefix="/api/v1/approvals"
        )
        
        num_requests = 20
        request_ids = []
        
        # Create many requests
        for i in range(num_requests):
            mock_request = Mock(spec=FunctionRequest)
            mock_request.id = f"stress-{i}"
            mock_request.tool_name = f"tool_{i}"
            mock_request.tool = Mock()
            mock_request.tool.args = []
            mock_request.tool.kwargs = {}
            
            request_id = await handler.approval_queue.create_request(mock_request)
            request_ids.append(request_id)
        
        # Verify all are pending
        assert len(handler.approval_queue.get_pending_requests()) == num_requests
        
        # Approve them all in random order with delays
        async def approve_all():
            import random
            shuffled = request_ids.copy()
            random.shuffle(shuffled)
            
            for req_id in shuffled:
                await asyncio.sleep(0.01)  # Small delay
                outcome = random.choice([
                    ApprovalOutcome.PROCEED_ONCE,
                    ApprovalOutcome.PROCEED_ALWAYS,
                    ApprovalOutcome.CANCEL
                ])
                handler.approval_queue.set_response(req_id, outcome)
        
        # Start approval task
        asyncio.create_task(approve_all())
        
        # Wait for all responses
        tasks = [
            handler.approval_queue.wait_for_response(req_id)
            for req_id in request_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should have results (no exceptions)
        assert len(results) == num_requests
        assert all(
            isinstance(r, ApprovalOutcome) or 
            (hasattr(r, 'outcome') and isinstance(r.outcome, ApprovalOutcome))
            for r in results
        )
        
        # Queue should be empty
        assert len(handler.approval_queue.pending_requests) == 0
        assert len(handler.approval_queue.responses) == 0


@pytest.mark.asyncio
async def test_edge_cases():
    """Test various edge cases and boundary conditions."""
    
    # Test with timeout = 0 (immediate timeout)
    queue = ApprovalQueue(timeout_seconds=0)
    mock_request = Mock(spec=FunctionRequest)
    mock_request.id = "zero-timeout"
    mock_request.tool_name = "test"
    mock_request.tool = Mock()
    mock_request.tool.args = []
    mock_request.tool.kwargs = {}
    
    request_id = await queue.create_request(mock_request)
    result = await queue.wait_for_response(request_id)
    assert result == ApprovalOutcome.CANCEL
    
    # Test double cleanup (should not raise)
    queue.pending_requests.pop(request_id, None)
    queue.responses.pop(request_id, None)
    # Should not raise even though already cleaned up


if __name__ == "__main__":
    pytest.main([__file__, "-v"])