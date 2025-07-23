# FastAPI Permission System - API Examples

## Overview

The FastAPI permission handler provides REST API endpoints for approving tool executions. This allows you to build web interfaces, mobile apps, or integrate with other systems for tool approval.

## Starting the Server

Run the test script and select option 1:
```bash
python test_fastapi_permission.py
# Select option 1 for interactive API approval
```

This starts a FastAPI server on `http://localhost:8000` with the following endpoints:

## API Endpoints

### 1. View API Documentation
```
http://localhost:8000/docs
```
Interactive Swagger UI documentation

### 2. Get Pending Approvals
```bash
curl http://localhost:8000/api/v1/approvals/pending
```

Response:
```json
[
  {
    "request_id": "123e4567-e89b-12d3-a456-426614174000",
    "tool_name": "search_web",
    "tool_args": ["Python tutorials"],
    "tool_kwargs": {"max_results": 5},
    "timestamp": "2024-01-17T23:30:00",
    "status": "pending"
  }
]
```

### 3. Get Specific Approval Request
```bash
curl http://localhost:8000/api/v1/approvals/{request_id}
```

### 4. Approve a Request
```bash
# Approve once
curl -X POST http://localhost:8000/api/v1/approvals/{request_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"request_id": "123e4567-e89b-12d3-a456-426614174000", "outcome": "proceed_once"}'

# Always allow this tool
curl -X POST http://localhost:8000/api/v1/approvals/{request_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"request_id": "123e4567-e89b-12d3-a456-426614174000", "outcome": "proceed_always"}'

# Cancel/Reject
curl -X POST http://localhost:8000/api/v1/approvals/{request_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"request_id": "123e4567-e89b-12d3-a456-426614174000", "outcome": "cancel"}'
```

### 5. Cancel a Request
```bash
curl -X DELETE http://localhost:8000/api/v1/approvals/{request_id}
```

### 6. Get Approval Statistics
```bash
curl http://localhost:8000/api/v1/approvals/stats
```

Response:
```json
{
  "total_requests": 10,
  "pending": 2,
  "approved": 6,
  "rejected": 1,
  "expired": 1,
  "always_allowed_tools": ["search_web"],
  "blocked_tools": []
}
```

## Python Client Example

```python
import httpx
import asyncio

async def approve_pending_requests():
    base_url = "http://localhost:8000/api/v1/approvals"

    async with httpx.AsyncClient() as client:
        # Get pending approvals
        response = await client.get(f"{base_url}/pending")
        pending = response.json()

        for request in pending:
            print(f"Tool: {request['tool_name']}")
            print(f"Args: {request['tool_args']}")

            # Approve the request
            approval = {
                "request_id": request['request_id'],
                "outcome": "proceed_once"
            }

            response = await client.post(
                f"{base_url}/{request['request_id']}/approve",
                json=approval
            )
            print(f"Approved: {response.json()}")

# Run the client
asyncio.run(approve_pending_requests())
```

## JavaScript/Frontend Example

```javascript
// Get pending approvals
fetch('http://localhost:8000/api/v1/approvals/pending')
  .then(response => response.json())
  .then(pending => {
    pending.forEach(request => {
      console.log(`Tool: ${request.tool_name}`);

      // Show approval UI
      if (confirm(`Approve ${request.tool_name}?`)) {
        // Approve the request
        fetch(`http://localhost:8000/api/v1/approvals/${request.request_id}/approve`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            request_id: request.request_id,
            outcome: 'proceed_once'
          })
        });
      }
    });
  });
```

## Integration with Web UI

You can build a web dashboard that:

1. Polls `/api/v1/approvals/pending` for new requests
2. Displays request details in a user-friendly interface
3. Provides buttons for approve/reject/always-allow actions
4. Shows real-time statistics from `/api/v1/approvals/stats`

## WebSocket Alternative

For real-time updates without polling, you could extend the FastAPI handler to include WebSocket support:

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Send pending approvals to connected clients
    # Notify when new approvals are needed
```

## Security Considerations

In production, you should:

1. Add authentication to the API endpoints
2. Use HTTPS for secure communication
3. Implement rate limiting
4. Add request validation and sanitization
5. Log all approval decisions for audit trails

## Deployment

For production deployment:

```python
# standalone_server.py
import uvicorn
from adalflow.core.fastapi_permission_handler import FastAPIPermissionHandler, create_standalone_app

app = create_standalone_app()
handler = FastAPIPermissionHandler(app=app, timeout_seconds=300)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
```
