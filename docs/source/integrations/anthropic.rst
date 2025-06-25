.. _integrations-anthropic:

Anthropic Integration
====================

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/models/anthropic_models.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

Introduction
------------
AdalFlow provides seamless integration with Anthropic's Claude models through the :class:`AnthropicAPIClient<components.model_client.anthropic_client.AnthropicAPIClient>`. This integration supports all Claude models including Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, and the latest Claude Sonnet 4 with advanced reasoning capabilities.

Key Features
-----------
- **Model Agnostic**: Easy switching between different Claude models via configuration
- **Reasoning Support**: Built-in support for Claude's reasoning capabilities with interleaved thinking
- **Tool Usage**: Support Anthropic's native tool via the `tools` parameter in `model_kwargs` but more powerfully, you can use the :class:`ReActAgent<components.agent.react.ReActAgent>` to use tools even easier.

Basic Usage
-----------

Simple Text Generation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import adalflow as adal

    # Basic Claude 3.5 Sonnet usage
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={"model": "claude-3-5-sonnet-20241022"},
    )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}
    response = anthropic_llm(prompt_kwargs)
    print(f"Anthropic: {response}")

Here is the output:

.. code-block:: text

    Anthropic: GeneratorOutput(id=None, data='The meaning of life is to find your own purpose while making a positive impact on others and growing through experiences, relationships, and personal discovery.', thinking=None, tool_use=None, error=None, usage=CompletionUsage(completion_tokens=31, prompt_tokens=70, total_tokens=101), raw_response='The meaning of life is to find your own purpose while making a positive impact on others and growing through experiences, relationships, and personal discovery.', api_response=Message(id='msg_01LxWtz8bjf4a9Q1GoXSMFDw', content=[TextBlock(citations=None, text='The meaning of life is to find your own purpose while making a positive impact on others and growing through experiences, relationships, and personal discovery.', type='text')], model='claude-3-5-sonnet-20241022', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=70, output_tokens=31, server_tool_use=None, service_tier='standard')), metadata=None)

The text output is in the `data` field, and the `api_response` field is the raw response from the Anthropic API.
The `raw_response` field is the text content of the api message.

Advanced Reasoning
~~~~~~~~~~~~~~~~~

Claude 4+ models support advanced reasoning with interleaved thinking:

.. code-block:: python

    # Enable reasoning with Claude Sonnet 4
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={
            "model": "claude-sonnet-4-20250514",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
        },
    )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}
    response = anthropic_llm(prompt_kwargs)
    print(f"Anthropic: {response}")


Here is the output:

.. code-block:: text

    Anthropic: GeneratorOutput(id=None, data='The meaning of life is to find purpose through love, growth, and connection—creating significance in our brief existence by caring for others and contributing something meaningful to the world.', thinking='This is a classic philosophical question that has been pondered by humans for millennia. There are many different perspectives on this - religious, philosophical, existential, scientific, etc. Since the user is asking for just one sentence, I should try to give a thoughtful but concise answer that captures something meaningful without being overly specific to any one worldview.\n\nSome possible approaches:\n- Focus on creating meaning through relationships, love, and connection\n- Emphasize personal growth and self-actualization  \n- Highlight contributing to something larger than oneself\n- Suggest that meaning is something we create rather than discover\n- Acknowledge the subjectivity of meaning\n\nI think a good answer would acknowledge that meaning is personal while suggesting some universal human elements that tend to contribute to a sense of meaning.', tool_use=None, error=None, usage=CompletionUsage(completion_tokens=212, prompt_tokens=98, total_tokens=310), raw_response='The meaning of life is to find purpose through love, growth, and connection—creating significance in our brief existence by caring for others and contributing something meaningful to the world.', api_response=Message(id='msg_01AR4WhBmWkrqYuSgy1ynR1k', content=[ThinkingBlock(signature='EoQICkYIBBgCKkB+8g6NuzKqJ/7OfwGHx5wggceB4m3tycTBSQj1NgE612HJPTTdwp2ZdOGXh8ZbRLq9GOVSi8irlbYln5jVkTFoEgwKIMoutjG61ZgGUUgaDEGYCumWe5A192OgAiIw+F7VcmCuFwoXkyDIhXXzPWVvNy43SoBJSNGgTxrPRh1oy+qS9AltBnPZB0WODj+IKusGvmgF4jkBxqigysaazpKJwyp3NMCfXhvScshflMUxt+3lIm0C2h0kCs6Rzzp+HSXXiTBUN6RKtGMYtl0iw4a9HlJzU4IwjdN/eAWlavaWo6fqR6ebmoiPAIHYxwH1HM5iP5j8SFToY8qeGTJnIRlX81PWDydR+M8RRCENBT/byaaK2vzpsmnSZR3ZDQ9ug+n+j9ROEWjfcbPuD/wzV43m1NugykP58Y1VG2+VbydB3MTwo3dJQ3+SdlkWmqO09xf+zzBPvV5JBJ9redlF9Sj84Z/jr4iDGxYESoj3LpPJvbHuCklpVvJweNuh5psw6owmZsVluxg5UFBaGLygxAeZeERMfwzsa0mumue4YFHlmP04qUO0X7BLBu0Uu9gFK4BqhdAQDCUgW2zuwU/Te3ZYQJMEBoq92zI8A6ngKBM7sR2k45aKwmJJBWQMNGnUvyF2IpKsF6WKDefVGtLbNyCuBTxUEf3mDyi6aoTGAd74lqv0TIciGn+/DlVIrP2/u9/pJghR2mZ4wh3jFnPwbQ6Nkbe2VWeJ88pNkGgfWLu0UlQB4NEuHyUWdKj58RYaSKRqhM0S1giP+oZs0b8hGZtTpMKZKviCU9G/AHqsKWt8A9HtalbFVbKIIYWUkb+OKBzH5TufhSYU6/4TT/VSee6Vwg877sneeXpRzrzwwZlKRpjtAmTHXO8w15p0t8ICvvzWNXZ1ryS5i5Na+cUM3ruufw6907mkWcuMjo2N/afEDIJVDKEr0s1FM34pu20veOKFZxxfw1iplYaV1c5XdWooWfPAFH2U4J0MrNC2hfSfPhUCEDjZG3vmn1tnCH4DiDFD2zZU8+OUbmx50fNps29hV6NmxkivLyl1oxSga+K7eJrPP1B6cTqBfNegXz8JN24jIkaUTQ4qojduaq8zRY3hedYz8a4lOKjdAcJ6uMlbvr65s0x8PGm6PKKXgZxxvgXVxHk5s4NaYGNohE5cWHORpBT5kwEJWTPobrkZwSiJ1gdSwQU9HNbA06MxSXqaoYchm0F6hrG9evV9ATu2s7EkVvgyzKlFxK5rggdkNjb08dF+D8nabfvoSns+P2LvyhA8xgMYNC6gL8tJnDJW0adKHVBUB6Rkq8li8WWRIyz1tKcIsUedcDQIrSw6pWy+b6VJWxsYBxlyRxURwG8YAQ==', thinking='This is a classic philosophical question that has been pondered by humans for millennia. There are many different perspectives on this - religious, philosophical, existential, scientific, etc. Since the user is asking for just one sentence, I should try to give a thoughtful but concise answer that captures something meaningful without being overly specific to any one worldview.\n\nSome possible approaches:\n- Focus on creating meaning through relationships, love, and connection\n- Emphasize personal growth and self-actualization  \n- Highlight contributing to something larger than oneself\n- Suggest that meaning is something we create rather than discover\n- Acknowledge the subjectivity of meaning\n\nI think a good answer would acknowledge that meaning is personal while suggesting some universal human elements that tend to contribute to a sense of meaning.', type='thinking'), TextBlock(citations=None, text='The meaning of life is to find purpose through love, growth, and connection—creating significance in our brief existence by caring for others and contributing something meaningful to the world.', type='text')], model='claude-sonnet-4-20250514', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=98, output_tokens=212, server_tool_use=None, service_tier='standard')), metadata=None)

In this case, you will see the `thinking` field is not None, and the `data` field is the final answer.



Tool Usage with Reasoning
~~~~~~~~~~~~~~~~~~~~~~~~

Claude models can use tools while leveraging reasoning capabilities.
In this case, we showcase the native tool usage via the `tools` parameter in `model_kwargs`.
And the `tool_use` field of the `GeneratorOutput` will show the final tool call.

.. code-block:: python

    # Define tools for function calling
    calculator_tool = {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }

    database_tool = {
        "name": "database_query",
        "description": "Query product database",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"}
            },
            "required": ["query"],
        },
    }

    # Configure Claude with tools and reasoning
    anthropic_llm = adal.Generator(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs={
            "model": "claude-sonnet-4-20250514",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
            "tools": [calculator_tool, database_tool],
        },
    )

    prompt_kwargs = {
        "input_str": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }

    response = anthropic_llm(prompt_kwargs)
    print(f"Anthropic: {response}")

Here is the output:

.. code-block:: text

    GeneratorOutput(id=None, data="I'll help you calculate the total revenue and compare it to your average monthly revenue. Let me start by calculating the revenue from the product A sales, then query the database for comparison.", thinking="The user is asking for two things:\n1. Calculate the total revenue from selling 150 units of product A at $50 each\n2. Compare this to the average monthly revenue from the database\n\nFor the first part, I can use the calculator tool to compute 150 × $50.\n\nFor the second part, I need to query the database to get information about monthly revenue. I'll need to construct a SQL query to get the average monthly revenue.\n\nLet me start with the calculation first, then query the database.", tool_use=Function(thought=None, name='calculator', args=[], kwargs={'expression': '150 * 50'}), error=None, usage=CompletionUsage(completion_tokens=213, prompt_tokens=704, total_tokens=917), raw_response="I'll help you calculate the total revenue and compare it to your average monthly revenue. Let me start by calculating the revenue from the product A sales, then query the database for comparison.", api_response=Message(id='msg_01QBwCzqGPLAwaxTDGChZsRN', content=[ThinkingBlock(signature='EowFCkYIBBgCKkAOLlygAYV+ek/V9AE3v7CpoG1GDak/cOxM4OrAKAwYP17ALABqh1v8TrHB4ztnCyEmTUGiS8Cxd6kYYdv+ZxrbEgzMQGltMsi2KNxD24kaDJ5D/0KuwSNkjxklUiIwNI1dR7riBX0gx8DNajHM2uW0uWy3zESb/Ow6JRpOFJyOToeWPeSEmp+MoZW7RFUIKvMDsXwGXb3yd0jjbvEEHlRhROLwPXYnMhSYtCnwu3MjshE8JnbbZibcRPz/XLCfNUv4pnorOqHumrJ7LVYcTUtzkUaVWQO4aNgbnUPbUDXTs6yFtms8DmOFHZUGGvOGm3VtfAtOW90D4l5jJ0d2i+x2tds/JeELCfEnHBdle6ftQv/7fHAafNzlbJbzpwYskqfw/IHEvA6pc17GlZRqZnFePLjD+EnrWPoUZcnfbdekQLBQCFhWOjpOqygGHl3qSe9v+Nww7wna+ZLW8PAne+wD0vnzT04K+lTVXCnucRpct4aB6cZF2h6wJTxN/lVavASAbqqXnlSLYGVQNQLuvKooHPO1Oh1a4e1y5CFQdIrBaQjipOhiKGzo7KwquFa90xD3XOkejfWrKf7yuHmqK0dZ/L24CSf9isNtI4BFEgDoPHUhl83RrXJ4xhFdoNRfTtCsWqAglQvwoCdFYvCoOoWoyc3QjIyPXEdfMcDJMrM4vVvVhNLZFhrsW+pjEqO1HCvI0plSVSqtuuT2E4xBPwjBYg3U1a6y7y4dSiDNC4CznZdpTv3JunCx4+sYGbAo8QfSU8IcWTjWRER7BYvNsKq6yGeKhwfKLDgclj7pQv4fUFutuzGhNrYxI4NFQDrYn6UP03WkYBLJPD8+aNwYYM6+W/BCiBgB', thinking="The user is asking for two things:\n1. Calculate the total revenue from selling 150 units of product A at $50 each\n2. Compare this to the average monthly revenue from the database\n\nFor the first part, I can use the calculator tool to compute 150 × $50.\n\nFor the second part, I need to query the database to get information about monthly revenue. I'll need to construct a SQL query to get the average monthly revenue.\n\nLet me start with the calculation first, then query the database.", type='thinking'), TextBlock(citations=None, text="I'll help you calculate the total revenue and compare it to your average monthly revenue. Let me start by calculating the revenue from the product A sales, then query the database for comparison.", type='text'), ToolUseBlock(id='toolu_01SwoVz9x4M24gceoybvUrqG', input={'expression': '150 * 50'}, name='calculator', type='tool_use')], model='claude-sonnet-4-20250514', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=704, output_tokens=213, server_tool_use=None, service_tier='standard')), metadata=None)

In particular, here is the `tool_use` field:

.. code-block:: text

    tool_use=Function(thought=None, name='calculator', args=[], kwargs={'expression': '150 * 50'})

As you can see, we have parsed the tool use response into a `Function` object.


ReAct Agent with Thinking Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of using the native tools, you can use the :class:`ReActAgent<components.agent.react.ReActAgent>` to use tools even easier.
The output is directly parsed into a `ReActAgentOutput` object where you can find both the `answer` and the `steps` of the agent.

In this case, we are using a thinking model, so inside of `ReActAgent`, the manual CoT is disabled on the reasoning.

.. code-block:: python

    from adalflow.components.agent.react import ReActAgent

    # Define thinking model configuration
    thinking_model_kwargs = {
        "model": "claude-sonnet-4-20250514",
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
    }

    # Define tools as Python functions
    def calculator(expression: str, **kwargs) -> str:
        """Perform mathematical calculations"""
        return "7500"

    def database_query(query: str, **kwargs) -> str:
        """Query product database"""
        return "5200"

    # Create ReAct Agent with Anthropic thinking model
    agent = ReActAgent(
        model_client=adal.AnthropicAPIClient(),
        model_kwargs=thinking_model_kwargs,
        tools=[calculator, database_query],
        use_cache=False,
        is_thinking_model=True,  # Enable thinking model support
    )

    prompt_kwargs = {
        "input_str": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }

    agent_response = agent(prompt_kwargs)
    print(f"Agent: {agent_response}")

The ReAct Agent with thinking models provides several advantages:

- **Enhanced Reasoning**: The agent can use Claude's interleaved thinking to break down complex problems
- **Tool Integration**: Seamless integration with Python functions as tools
- **Multi-step Planning**: The agent can plan and execute multi-step reasoning processes
- **Automatic Tool Calling**: Tools are automatically called based on the agent's reasoning

Available Models
---------------

AdalFlow supports all current Anthropic Claude models.
You can find the list of available models in the `Anthropic API documentation <https://docs.anthropic.com/en/docs/about-claude/models/overview>`_.

Configuration Options
--------------------

The :class:`AnthropicAPIClient<components.model_client.anthropic_client.AnthropicAPIClient>` supports various configuration options:

.. code-block:: python

    anthropic_client = adal.AnthropicAPIClient(
        api_key="your-api-key",  # Optional: uses ANTHROPIC_API_KEY env var
        base_url="https://api.anthropic.com",  # Optional: custom base URL
        timeout=60,  # Optional: request timeout in seconds
        max_retries=3,  # Optional: maximum retry attempts
    )

Model-specific parameters can be passed via `model_kwargs`:

.. code-block:: python

    model_kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "tools": [...],  # Function definitions
        "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
    }



.. admonition:: API Reference
   :class: highlight

   - :class:`components.model_client.anthropic_client.AnthropicAPIClient`
   - :class:`core.generator.Generator`
   - :class:`core.types.GeneratorOutput`
   - :class:`components.agent.react.ReActAgent`

.. admonition:: Related Tutorials
   :class: highlight

   - :ref:`Generator <generator>`
   - :ref:`ModelClient <tutorials-model_client>`
   - :ref:`RAG <tutorials-rag>`
   - :ref:`Agent <tutorials-agent>`
