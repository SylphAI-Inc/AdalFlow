from adalflow.components.model_client.groq_client import GroqAPIClient
import adalflow as adal

# from adalflow.utils import get_logger

# get_logger(level="DEBUG")


rag_template = r"""<START_OF_SYSTEM_MESSAGE>
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
    <START_OF_QUERY>
    {{input_str}}
    <END_OF_QUERY>
    {% if context_str %}
    <START_OF_CONTEXT>
    {{context_str}}
    <END_OF_CONTEXT>
    {% endif %}
<END_OF_USER_MESSAGE>
"""
adal.setup_env()
llama3_model = {
    "model_client": GroqAPIClient(),
    "model_kwargs": {
        "model": "llama-3.1-8b-instant",
    },
}


generator = adal.Generator(
    **llama3_model,
    template=rag_template,
)

prompt_kwargs = {
    "input_str": "where is brian?",
    "context_str": "Brian is in the kitchen.",
}

output = generator(prompt_kwargs=prompt_kwargs)
print(output)
