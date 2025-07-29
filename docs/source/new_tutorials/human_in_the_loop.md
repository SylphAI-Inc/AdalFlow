<div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
    <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/agent/tutorial_agent_advanced_features.py" target="_blank" style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
        <span style="vertical-align: middle;"> Open Source Code (Agent Advanced)</span>
    </a>
</div>

# Human in the Loop

AdalFlow provides a permission management system that allows you to control and approve tool executions before they run. This is particularly useful for tools that perform sensitive operations like file system access, API calls, or external communications.

## Core Components

The permission system consists of:

- **PermissionManager**: Components that manage approval workflows (e.g., **CLIPermissionHandler**, **AutoApprovalHandler**)
- **Approval Modes**: Different strategies for handling permission requests:
  - `"default"`: Respects all tool categories (always_allowed, blocked, require_approval)
  - `"auto_approve"`: Automatically approves tools requiring approval (still respects blocked_tools)
  - `"yolo"`: Bypasses all permission checks entirely (use only in development/trusted environments)
- **Tool Categories**:
  - `always_allowed_tools`: Tools that never require approval
  - `blocked_tools`: Tools that are completely blocked from execution
  - `tool_require_approval`: Tools that need explicit approval before execution
- **ApprovalOutcome**:
  - `PROCEED_ONCE`: Allow this single execution
  - `PROCEED_ALWAYS`: Allow this tool always (adds to always_allowed_tools)
  - `CANCEL`: Deny execution

Note that tools are expected to have a `require_approval` attribute to be properly registered in the permission manager. It is most convenient to create tools that needs approval via the FunctionTool class.

## Basic Usage Example

Add approval workflows for sensitive operations such as creating a file:

```python
from adalflow.apps.cli_permission_handler import CLIPermissionHandler
from adalflow.utils import setup_env
from adalflow.core.types import ToolOutput
from adalflow.core.types import FunctionRequest
from adalflow.components.agent import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.func_tool import FunctionTool

setup_env()

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def file_writer(filename: str, content: str) -> ToolOutput:
    """Write content to a file - requires permission."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return ToolOutput(
            output=f"Successfully wrote {len(content)} characters to {filename}",
            observation=f"File {filename} written successfully",
            display=f"✍️ Wrote: {filename}",
        )
    except Exception as e:
        return ToolOutput(
            output=f"Error writing to file: {e}",
            observation=f"Failed to write to {filename}",
            display=f"❌ Failed: {filename}",
        )

# Create agent with tools that require permission
agent = Agent(
    name="PermissionAgent",
    tools=[
        FunctionTool(calculator),  # Safe tool - no permission needed
        FunctionTool(file_writer, require_approval=True),  # Requires permission
    ],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=6
)

permission_handler = CLIPermissionHandler(approval_mode="default")
runner = Runner(agent=agent, permission_manager=permission_handler)

# Tools will now require approval before execution
result = runner.call(prompt_kwargs={"input_str": "Create a file called 'test.txt' with some interesting content"})
```

You should see a temporary file that was created by the agent from the tutorial that contains interesting content. 

```
Title: The Whispering Woods

Introduction:
Nestled between the towering peaks of the Silver Mountains lies the Whispering Woods, a place shrouded in mystery and enchantment. Legends speak of ancient spirits that dwell within, their voices carried by the wind to those who dare to listen.

Chapter 1: The Call of the Forest
In the village of Elderglen, tales of the Whispering Woods were as common as the morning sun. Elders spoke of a time when the forest was alive with magic, a sanctuary for creatures of myth and wonder. Young Elara, a curious soul with a heart full of adventure, felt an irresistible pull towards the woods.

Chapter 2: The Enchanted Path
One crisp autumn morning, Elara set out on a journey to uncover the secrets of the forest. As she stepped onto the leaf-strewn path, the trees seemed to lean in, their branches forming a protective archway. The air was filled with a symphony of rustling leaves and distant whispers.

Chapter 3: Guardians of the Grove
Deep within the heart of the woods, Elara encountered the Guardians—ethereal beings who watched over the forest. They spoke of a hidden realm where time stood still, a place where dreams and reality intertwined. To enter, one must prove their heart's true intent.

Chapter 4: The Trials of Truth
Guided by the Guardians, Elara faced a series of trials, each designed to test her courage, wisdom, and compassion. She navigated through illusions, solved ancient riddles, and embraced the unknown, her spirit unwavering.

Chapter 5: The Heart of the Forest
Having passed the trials, Elara reached the heart of the forest—a serene glade bathed in golden light. Here, the whispers grew louder, revealing the forest's ancient secrets and the interconnectedness of all life. Elara realized her purpose: to be a bridge between the world of magic and the realm of man.

Conclusion: A New Beginning
Returning to Elderglen, Elara shared her journey with the villagers, inspiring a newfound respect for the Whispering Woods. The forest, once a place of mystery, became a symbol of harmony and hope, its whispers a reminder of the magic that lies within us all.
```

## Permission Handler Types

### CLI Permission Handler

The `CLIPermissionHandler` provides interactive command-line approval for tool execution:

```python
from adalflow.apps.cli_permission_handler import CLIPermissionHandler

# Default mode - requires approval for tools marked with require_approval=True
cli_handler = CLIPermissionHandler(approval_mode="default")

# Auto-approve mode - automatically approves requiring approval tools
auto_cli_handler = CLIPermissionHandler(approval_mode="auto_approve")

# YOLO mode - bypasses all permission checks
yolo_handler = CLIPermissionHandler(approval_mode="yolo")
```

When using CLI permission handler in default mode, you'll see prompts like:
```
============================================================
🔧 TOOL PERMISSION REQUEST
============================================================
Tool: file_writer
Arguments: ['test.txt']
Keyword Arguments: {'content': 'Hello World'}

Allow execution of 'file_writer'?

Options:
1. Allow once
2. Allow always for this tool
3. Cancel
4. Modify arguments (not implemented)
------------------------------------------------------------
Your choice (1-4):
```

### Auto Approval Handler

For development environments or trusted scenarios:

```python
# Automatically approves all tool requests
auto_handler = AutoApprovalHandler()
runner = Runner(agent=agent, permission_manager=auto_handler)
```

## API Reference

:::{admonition} API reference
:class: highlight

- {doc}`adalflow.components.agent.agent.Agent <../apis/components/components.agent.agent>`
- {doc}`adalflow.components.agent.runner.Runner <../apis/components/components.agent.runner>`
- {doc}`adalflow.core.func_tool.FunctionTool <../apis/core/core.func_tool>`
- {doc}`adalflow.core.types.ToolOutput <../apis/core/core.types>`
- {doc}`adalflow.core.types.FunctionRequest <../apis/core/core.types>`
:::
