from adalflow.apps.cli_permission_handler import AutoApprovalHandler
from adalflow.components.agent import Runner


# Automatically approves all tool requests
auto_handler = AutoApprovalHandler()
runner = Runner(agent=agent, permission_manager=auto_handler)