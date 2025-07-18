# Tool Approval Design for Python Agent Systems

## Overview

This document outlines the design for implementing a tool approval system in Python-based agent frameworks, specifically for AdalFlow and similar systems. The design is inspired by the TypeScript implementation in gemini-cli but adapted for Python's async patterns.

## Python Implementation Design

### 1. **Core Components**

#### Permission Manager
```python
from enum import Enum
from typing import Optional, Callable, Dict, Any
import asyncio
from dataclasses import dataclass

class ApprovalOutcome(Enum):
    PROCEED_ONCE = "proceed_once"
    PROCEED_ALWAYS = "proceed_always" 
    CANCEL = "cancel"
    MODIFY = "modify"

@dataclass
class PermissionRequest:
    tool_name: str
    tool_args: list
    tool_kwargs: dict
    confirmation_message: str
    
class PermissionManager:
    def __init__(self, approval_callback: Optional[Callable] = None):
        self.approval_callback = approval_callback
        self.always_allowed_tools = set()
        self.pending_approvals = {}
        
    async def check_permission(self, func: Function) -> tuple[bool, Optional[Function]]:
        if func.name in self.always_allowed_tools:
            return True, func
            
        if self.approval_callback:
            request = PermissionRequest(
                tool_name=func.name,
                tool_args=func.args,
                tool_kwargs=func.kwargs,
                confirmation_message=f"Allow execution of {func.name}?"
            )
            
            outcome = await self.approval_callback(request)
            
            if outcome == ApprovalOutcome.PROCEED_ALWAYS:
                self.always_allowed_tools.add(func.name)
                return True, func
            elif outcome == ApprovalOutcome.PROCEED_ONCE:
                return True, func
            elif outcome == ApprovalOutcome.CANCEL:
                return False, None
                
        return True, func  # Default allow if no callback
```

### 2. **Integration with AdalFlow Runner**

#### Modify Runner Initialization
```python
def __init__(
    self,
    agent: Agent,
    ctx: Optional[Dict] = None,
    max_steps: Optional[int] = None,
    permission_manager: Optional[PermissionManager] = None,
    **kwargs,
) -> None:
    super().__init__(**kwargs)
    self.agent = agent
    self.permission_manager = permission_manager or PermissionManager()
    # ... rest of init
```

#### Update Tool Execution Methods

For synchronous execution (`_tool_execute_sync`):
```python
def _tool_execute_sync(self, func: Function) -> Union[FunctionOutput, Parameter]:
    """Execute tool with permission check"""
    
    # Check permission before execution
    if self.permission_manager:
        allowed, modified_func = asyncio.run(
            self.permission_manager.check_permission(func)
        )
        
        if not allowed:
            return FunctionOutput(
                name=func.name,
                input=func,
                output=ToolOutput(
                    observation="Tool execution cancelled by user",
                    error="Permission denied"
                )
            )
        
        # Use modified function if user edited it
        func = modified_func or func
    
    # Original execution code
    result = self.agent.tool_manager.execute_func(func=func)
    
    if not isinstance(result, FunctionOutput):
        raise ValueError("Result is not a FunctionOutput")
        
    return result
```

For asynchronous execution (`_tool_execute_async`):
```python
async def _tool_execute_async(self, func: Function) -> Union[FunctionOutput, Parameter]:
    """Execute tool asynchronously with permission check"""
    
    # Check permission before execution
    if self.permission_manager:
        allowed, modified_func = await self.permission_manager.check_permission(func)
        
        if not allowed:
            return FunctionOutput(
                name=func.name,
                input=func,
                output=ToolOutput(
                    observation="Tool execution cancelled by user",
                    error="Permission denied"
                )
            )
        
        func = modified_func or func
    
    # Original execution code
    result = await self.agent.tool_manager.execute_func_async(func=func)
    
    if not isinstance(result, FunctionOutput):
        raise ValueError("Result is not a FunctionOutput")
        
    return result
```

### 3. **User Input Methods**

#### WebSocket-Based Real-Time Approval

```python
# backend/approval_handler.py
import asyncio
from fastapi import FastAPI, WebSocket
from typing import Dict, Optional
import json

app = FastAPI()

class ApprovalWebSocketHandler:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.pending_approvals: Dict[str, asyncio.Future] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def request_approval(self, request: PermissionRequest) -> ApprovalOutcome:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_approvals[request_id] = future
        
        # Send to all connected clients
        message = {
            "type": "approval_request",
            "id": request_id,
            "tool": request.tool_name,
            "args": request.tool_args,
            "message": f"Allow {request.tool_name} to execute?"
        }
        
        for ws in self.active_connections.values():
            await ws.send_json(message)
        
        # Wait for user response
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            return ApprovalOutcome.CANCEL
    
    async def handle_approval_response(self, data: dict):
        request_id = data.get("id")
        if request_id in self.pending_approvals:
            outcome = ApprovalOutcome(data.get("outcome", "cancel"))
            self.pending_approvals[request_id].set_result(outcome)

# WebSocket endpoint
approval_handler = ApprovalWebSocketHandler()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await approval_handler.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "approval_response":
                await approval_handler.handle_approval_response(data)
    except:
        await approval_handler.disconnect(client_id)
```

Frontend JavaScript example:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'approval_request') {
    // Show UI prompt
    const userChoice = prompt(`${data.message}\n1. Allow once\n2. Allow always\n3. Cancel`);
    
    let outcome = 'cancel';
    if (userChoice === '1') outcome = 'proceed_once';
    else if (userChoice === '2') outcome = 'proceed_always';
    
    ws.send(JSON.stringify({
      type: 'approval_response',
      id: data.id,
      outcome: outcome
    }));
  }
};
```

#### REST API with Long Polling

```python
from fastapi import FastAPI, HTTPException
from typing import Dict, Optional
import asyncio
import uuid

app = FastAPI()

class ApprovalQueue:
    def __init__(self):
        self.pending_requests: Dict[str, PermissionRequest] = {}
        self.responses: Dict[str, asyncio.Future] = {}
    
    async def create_request(self, request: PermissionRequest) -> str:
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = request
        self.responses[request_id] = asyncio.Future()
        return request_id
    
    async def wait_for_response(self, request_id: str, timeout: float = 30.0):
        if request_id not in self.responses:
            raise ValueError("Invalid request ID")
        
        try:
            result = await asyncio.wait_for(
                self.responses[request_id], 
                timeout=timeout
            )
            return result
        finally:
            # Clean up
            self.pending_requests.pop(request_id, None)
            self.responses.pop(request_id, None)

approval_queue = ApprovalQueue()

# Endpoint to get pending approvals
@app.get("/approvals/pending")
async def get_pending_approvals():
    return {
        id: {
            "tool": req.tool_name,
            "args": req.tool_args,
            "message": req.confirmation_message
        }
        for id, req in approval_queue.pending_requests.items()
    }

# Endpoint to submit approval
@app.post("/approvals/{request_id}")
async def submit_approval(request_id: str, outcome: str):
    if request_id not in approval_queue.responses:
        raise HTTPException(404, "Request not found")
    
    approval_outcome = ApprovalOutcome(outcome)
    approval_queue.responses[request_id].set_result(approval_outcome)
    return {"status": "approved", "outcome": outcome}

# Modified permission manager
class RESTPermissionManager(PermissionManager):
    async def check_permission(self, func: Function):
        request = PermissionRequest(
            tool_name=func.name,
            tool_args=func.args,
            tool_kwargs=func.kwargs,
            confirmation_message=f"Allow {func.name}?"
        )
        
        request_id = await approval_queue.create_request(request)
        
        # Wait for user response
        outcome = await approval_queue.wait_for_response(request_id)
        
        if outcome == ApprovalOutcome.PROCEED_ALWAYS:
            self.always_allowed_tools.add(func.name)
            return True, func
        elif outcome == ApprovalOutcome.PROCEED_ONCE:
            return True, func
        else:
            return False, None
```

#### CLI-Based Approval (for Development/Testing)

```python
import aioconsole

class CLIPermissionManager(PermissionManager):
    async def check_permission(self, func: Function):
        # Format the prompt
        print("\n" + "="*50)
        print(f"🔧 Tool Request: {func.name}")
        print(f"📋 Arguments: {func.args}")
        print(f"🔑 Kwargs: {func.kwargs}")
        print("="*50)
        print("Options:")
        print("1. Allow once")
        print("2. Allow always")
        print("3. Cancel")
        print("4. Modify arguments")
        
        # Get user input asynchronously
        choice = await aioconsole.ainput("Your choice (1-4): ")
        
        if choice == "1":
            return True, func
        elif choice == "2":
            self.always_allowed_tools.add(func.name)
            return True, func
        elif choice == "3":
            return False, None
        elif choice == "4":
            # Allow user to modify arguments
            new_args = await aioconsole.ainput(f"New args (current: {func.args}): ")
            modified_func = Function(
                name=func.name,
                args=eval(new_args) if new_args else func.args,
                kwargs=func.kwargs,
                thought=func.thought
            )
            return True, modified_func
        else:
            return False, None
```

#### Streamlit Integration

```python
import streamlit as st
import asyncio
from threading import Thread

class StreamlitPermissionManager(PermissionManager):
    def __init__(self):
        super().__init__()
        self.pending_approvals = {}
        
    async def check_permission(self, func: Function):
        request_id = str(uuid.uuid4())
        
        # Store request in session state
        if 'pending_approvals' not in st.session_state:
            st.session_state.pending_approvals = {}
        
        st.session_state.pending_approvals[request_id] = {
            'tool': func.name,
            'args': func.args,
            'kwargs': func.kwargs,
            'status': 'pending'
        }
        
        # Wait for user response
        while st.session_state.pending_approvals[request_id]['status'] == 'pending':
            await asyncio.sleep(0.1)
        
        status = st.session_state.pending_approvals[request_id]['status']
        del st.session_state.pending_approvals[request_id]
        
        if status == 'approved_once':
            return True, func
        elif status == 'approved_always':
            self.always_allowed_tools.add(func.name)
            return True, func
        else:
            return False, None

# In your Streamlit app
def show_approval_requests():
    if 'pending_approvals' in st.session_state:
        for req_id, req in st.session_state.pending_approvals.items():
            if req['status'] == 'pending':
                with st.expander(f"Tool Request: {req['tool']}"):
                    st.write(f"Arguments: {req['args']}")
                    st.write(f"Kwargs: {req['kwargs']}")
                    
                    col1, col2, col3 = st.columns(3)
                    if col1.button("Allow Once", key=f"once_{req_id}"):
                        st.session_state.pending_approvals[req_id]['status'] = 'approved_once'
                    if col2.button("Allow Always", key=f"always_{req_id}"):
                        st.session_state.pending_approvals[req_id]['status'] = 'approved_always'
                    if col3.button("Cancel", key=f"cancel_{req_id}"):
                        st.session_state.pending_approvals[req_id]['status'] = 'cancelled'
```

### 4. **Streaming Support**

Add permission events for streaming in `impl_astream`:

```python
# Before function execution in streaming
if self.permission_manager and hasattr(self.permission_manager, 'needs_approval'):
    if await self.permission_manager.needs_approval(function):
        # Emit permission request event
        permission_event = RunItemStreamEvent(
            name="agent.tool_permission_request",
            item=ToolPermissionItem(
                id=tool_call_id,
                tool_name=function.name,
                args=function.args,
                kwargs=function.kwargs
            )
        )
        streaming_result.put_nowait(permission_event)

# Continue with existing tool execution
function_result = await self._tool_execute_async(function)
```

## Usage Example

```python
# main.py
async def run_agent_with_approval():
    # Choose your approval method
    if USE_WEBSOCKET:
        ws_handler = ApprovalWebSocketHandler()
        permission_manager = PermissionManager(
            approval_callback=ws_handler.request_approval
        )
    elif USE_CLI:
        permission_manager = CLIPermissionManager()
    elif USE_REST:
        permission_manager = RESTPermissionManager()
    else:
        permission_manager = None  # No approval needed
    
    # Create runner with permission support
    runner = Runner(
        agent=my_agent,
        permission_manager=permission_manager
    )
    
    # Run with approval flow
    result = await runner.acall(
        prompt_kwargs={"query": "Process this data"},
        model_kwargs={}
    )
    
    return result
```

## Key Benefits

1. **Non-intrusive**: Minimal changes to existing Runner code
2. **Flexible**: Works with both sync and async execution
3. **Streaming Support**: Integrates with the streaming API
4. **Configurable**: Can be disabled by not providing permission_manager
5. **Extensible**: Easy to add more approval modes or modify behavior

## Recommendation

For production use, the **WebSocket approach** is recommended because it:
- Provides real-time bidirectional communication
- Maintains connection state
- Supports multiple concurrent approval requests
- Works well with modern web frontends
- Can be easily integrated with various UI frameworks

The REST API approach is good for simpler setups or when WebSockets aren't available, while the CLI approach is excellent for development and testing.

## Key Design Principles

1. **Async/Await**: Use Python's asyncio to handle non-blocking waits
2. **State Management**: Track tool states to know when approval is pending
3. **Queue/Future Pattern**: Use asyncio.Queue or Future objects to pause execution
4. **WebSocket/SSE**: For real-time communication with frontend
5. **Timeout Handling**: Add timeouts to prevent indefinite waits

This approach ensures the Python agent can pause gracefully when user permission is needed and resume execution once a decision is made, similar to the TypeScript implementation but adapted for Python's async patterns.