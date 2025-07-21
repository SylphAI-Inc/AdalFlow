#!/usr/bin/env python3
"""
Test script to demonstrate the FastAPI permission system for tool execution.
Run this script and then use the API endpoints to approve/reject tool executions.
"""

import asyncio
import uvicorn
from threading import Thread
from adalflow.components.agent.agent import Agent
from adalflow.components.runner.runner import Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.apps.fastapi_permission_handler import FastAPIPermissionHandler, create_standalone_app
from adalflow.core.types import RunItemStreamEvent, ToolOutput
from adalflow.core.prompt_builder import Prompt
from adalflow.utils import setup_env, printc, get_logger
from adalflow.components.model_client.openai_client import OpenAIClient
import httpx
import time

# Setup environment
setup_env()
get_logger(level="DEBUG", enable_file=False)

# Model configuration
openai_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {"model": "gpt-4o-mini", "max_tokens": 4096}
}


# Example tools
def search_web(query: str, max_results: int = 5) -> ToolOutput:
    """Search the web for information."""
    print(f"[Tool Execution] Searching web for: {query} (max {max_results} results)")
    return ToolOutput(
        output=f"Found {max_results} results for '{query}':\n1. Python Tutorial - Learn Python Programming\n2. Advanced Python Tips and Tricks\n3. Python Best Practices Guide",
        observation=f"Search completed for '{query}'",
        display=f"🔍 Searched: {query}"
    )


def read_file(filename: str) -> ToolOutput:
    """Read contents of a file."""
    print(f"[Tool Execution] Reading file: {filename}")
    return ToolOutput(
        output=f"Contents of {filename}:\n# Example File\nThis is the content of {filename}.\nIt contains important information.",
        observation=f"Successfully read file {filename}",
        display=f"📄 Read: {filename}"
    )


def write_file(filename: str, content: str) -> ToolOutput:
    """Write content to a file."""
    print(f"[Tool Execution] Writing to file: {filename}")
    print(f"[Tool Execution] Content preview: {content[:100]}...")
    return ToolOutput(
        output=f"Successfully wrote {len(content)} characters to {filename}",
        observation=f"File {filename} written successfully",
        display=f"✍️ Wrote: {filename}"
    )


def run_fastapi_server(app, port=8000):
    """Run FastAPI server in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


async def test_api_approval_flow():
    """Test the API-based approval flow."""
    print("\n" + "="*80)
    print("🚀 FASTAPI PERMISSION SYSTEM TEST")
    print("="*80)
    
    # Create FastAPI app with permission handler
    app = create_standalone_app()
    permission_handler = FastAPIPermissionHandler(
        app=app,
        timeout_seconds=160,  # 60 second timeout for approvals
        api_prefix="/api/v1/approvals"
    )
    
    # Start FastAPI server in background thread
    server_thread = Thread(target=run_fastapi_server, args=(app, 8000), daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("\n⏳ Starting FastAPI server...")
    await asyncio.sleep(2)
    
    print("\n✅ FastAPI server is running!")
    print("\n📍 API Endpoints:")
    print("   - GET  http://localhost:8000/docs           (Swagger UI)")
    print("   - GET  http://localhost:8000/api/v1/approvals/pending")
    print("   - POST http://localhost:8000/api/v1/approvals/{request_id}/approve")
    print("   - GET  http://localhost:8000/api/v1/approvals/stats")
    print("\n" + "="*80)
    
    # Create agent
    role_desc = Prompt(template="""You are a helpful assistant with access to file operations and web search.
    Use your tools to complete the user's request. Tool usage requires approval via API.""")
    
    agent = Agent(
        name="APIApprovalAgent",
        role_desc=role_desc,
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(read_file, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model
    )
    
    # Create runner with API permission manager
    runner = Runner(
        agent=agent,
        permission_manager=permission_handler,
        max_steps=5
    )
    
    # Test prompt
    prompt_kwargs = {
        "input_str": "Search for Python tutorials, read README.md, and write a summary to output.txt"
    }
    
    print("\n🤖 Starting agent execution...")
    print(f"📝 Task: {prompt_kwargs['input_str']}")
    print("\n⚠️  Tools will require approval via API!")
    print("="*80 + "\n")
    
    # Execute with streaming
    streaming_result = runner.astream(prompt_kwargs)
    
    # Process events
    event_count = 0
    async for event in streaming_result.stream_events():
        event_count += 1
        if isinstance(event, RunItemStreamEvent):
            if event.name == "agent.tool_permission_request":
                printc(f"\n⏸️  APPROVAL REQUIRED: {event.item.data.tool_name}", color="yellow")
                print("\n   Use the API to approve/reject this tool execution:")
                print("   1. Check pending: GET http://localhost:8000/api/v1/approvals/pending")
                print("      # curl http://localhost:8000/api/v1/approvals/pending")
                print("   2. Approve: POST http://localhost:8000/api/v1/approvals/{request_id}/approve")
                print("      Body: {\"request_id\": \"...\", \"outcome\": \"proceed_once\"}")
                print("      # curl -X POST http://localhost:8000/api/v1/approvals/REQUEST_ID/approve \\")
                print("      #   -H \"Content-Type: application/json\" \\")
                print("      #   -d '{\"request_id\": \"REQUEST_ID\", \"outcome\": \"proceed_once\"}'")
                print("\n   Waiting for approval (60s timeout)...\n")
            elif event.name == "agent.tool_call_start":
                printc(f"🔧 Tool Call Start: {event.item.data.name}", color="cyan")
            elif event.name == "agent.tool_call_complete":
                printc("✅ Tool Call Complete", color="green")
            elif event.name == "agent.step_complete":
                printc(f"📍 Step {event.item.data.step} Complete", color="blue")
            elif event.name == "agent.execution_complete":
                printc("🎯 Execution Complete", color="green")
    
    print(f"\n📊 Total events: {event_count}")
    print(f"📝 Final answer: {streaming_result.answer}")
    
    # Show final stats
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/approvals/stats")
        stats = response.json()
        print(f"\n📈 Approval Statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")


async def demo_api_client():
    """Demonstrate how to use the API to approve requests."""
    print("\n" + "="*80)
    print("🔧 API CLIENT DEMO")
    print("="*80)
    
    base_url = "http://localhost:8000/api/v1/approvals"
    
    async with httpx.AsyncClient() as client:
        print("\n1. Checking for pending approvals...")
        response = await client.get(f"{base_url}/pending")
        pending = response.json()
        
        if not pending:
            print("   No pending approvals found.")
            return
            
        print(f"   Found {len(pending)} pending approval(s):")
        for req in pending:
            print(f"\n   Request ID: {req['request_id']}")
            print(f"   Tool: {req['tool_name']}")
            print(f"   Args: {req['tool_args']}")
            print(f"   Status: {req['status']}")
            
        # Approve the first request
        if pending:
            request_id = pending[0]['request_id']
            print(f"\n2. Approving request {request_id}...")
            
            approval_data = {
                "request_id": request_id,
                "outcome": "proceed_once"  # or "proceed_always", "cancel"
            }
            
            response = await client.post(
                f"{base_url}/{request_id}/approve",
                json=approval_data
            )
            
            result = response.json()
            print(f"   Result: {result}")


async def automated_test_with_client():
    """Run an automated test that approves requests programmatically."""
    print("\n" + "="*80)
    print("🤖 AUTOMATED TEST WITH API CLIENT")
    print("="*80)
    
    # Create FastAPI app with permission handler
    app = create_standalone_app()
    permission_handler = FastAPIPermissionHandler(
        app=app,
        timeout_seconds=30,
        api_prefix="/api/v1/approvals"
    )
    
    # Start server
    server_thread = Thread(target=run_fastapi_server, args=(app, 8001), daemon=True)
    server_thread.start()
    await asyncio.sleep(2)
    
    # Create agent and runner
    agent = Agent(
        name="AutomatedTestAgent",
        role_desc=Prompt(template="You are a helpful assistant with tool access."),
        tools=[
            FunctionTool(search_web, require_approval=True),
            FunctionTool(write_file, require_approval=True),
        ],
        answer_data_type=str,
        **openai_model
    )
    
    runner = Runner(
        agent=agent,
        permission_manager=permission_handler,
        max_steps=3
    )
    
    # Create a task to auto-approve requests
    async def auto_approver():
        """Automatically approve all requests after a short delay."""
        base_url = "http://localhost:8001/api/v1/approvals"
        
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(1)
                
                # Check for pending approvals
                try:
                    response = await client.get(f"{base_url}/pending")
                    pending = response.json()
                    
                    for req in pending:
                        print(f"\n🤖 Auto-approving: {req['tool_name']}")
                        
                        approval_data = {
                            "request_id": req['request_id'],
                            "outcome": "proceed_once"
                        }
                        
                        await client.post(
                            f"{base_url}/{req['request_id']}/approve",
                            json=approval_data
                        )
                except:
                    pass
    
    # Start auto-approver
    approver_task = asyncio.create_task(auto_approver())
    
    # Run agent
    prompt_kwargs = {
        "input_str": "Search for Python best practices and write a summary to best_practices.txt"
    }
    
    print(f"\n📝 Task: {prompt_kwargs['input_str']}")
    print("🤖 Auto-approver is running...\n")
    
    result = await runner.acall(prompt_kwargs)
    print(f"\n✅ Final result: {result}")
    
    # Cancel approver
    approver_task.cancel()


async def main():
    """Main test menu."""
    print("\n" + "="*60)
    print("FASTAPI PERMISSION SYSTEM TEST SUITE")
    print("="*60)
    print("\nSelect a test:")
    print("1. Interactive API Approval Flow (manual approval via API)")
    print("2. API Client Demo (shows how to use the API)")
    print("3. Automated Test (auto-approves requests)")
    print("="*60)
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        await test_api_approval_flow()
    elif choice == "2":
        # Start a simple server first
        app = create_standalone_app()
        FastAPIPermissionHandler(app=app)
        server_thread = Thread(target=run_fastapi_server, args=(app, 8000), daemon=True)
        server_thread.start()
        await asyncio.sleep(2)
        await demo_api_client()
    elif choice == "3":
        await automated_test_with_client()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    # For production use, you might want to run just the FastAPI server:
    # app = create_standalone_app()
    # handler = FastAPIPermissionHandler(app=app)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    asyncio.run(main())