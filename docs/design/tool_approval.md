if user rejects a tool, we should proceed to the next tool in default,
we can additionally make it controllable in the runner config, or stop when tool is rejected.


# flow

The agent outputs a function, attach an id to it, and then the functin execution will use check_permission and emite an event : agent.tool_permission_request 

If uers has no response. It will stuck there forever. no time out.