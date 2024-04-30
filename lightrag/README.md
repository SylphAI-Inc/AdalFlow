"""
test file: tests/test_tool.py

Different from llamaindex which defines these types of tools:
- FunctionTool
- RetrieverTool
- QueryEngineTool
-...
Llamaindex: BaseTool->AsyncBaseTool->FunctionTool
Our tool is an essential callable object (similar to the function tool) that you can wrap in any other parts such as retriever, generator in.
TO support:
- sync tool
- async tool
TODO: to observe and improve the mix of sync and async tools in the future.
How can we know after the llm call that a function tool is sync or async?
"""

Tool can be under `/lightrag`.
