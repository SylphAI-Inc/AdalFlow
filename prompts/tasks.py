COT_TASK_DESC_STR_BASIC = "You are a helpful assistant. Let's think step-by-step (be concise too) to answer user's query."
# Using triple quotes to include JSON-like structure more cleanly
COT_TASK_DESC_STR_WITH_JSON_OUTPUT = f"""
{COT_TASK_DESC_STR_BASIC} Output JSON format: {{"thought": "<The thought process to answer the query>", "answer": "<The answer to the query>"}}
"""
