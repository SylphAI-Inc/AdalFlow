Tools and Function calls
===========================

Function calls

Users might be aware of OpenAI's function call feature via its API (https://platform.openai.com/docs/guides/function-calling).
As a library, we prioritize the built-in function call capabilities via the normal prompt-response.
Function calls are often just a prerequisite for more complext agent behaviors.
This means we need to know how to form a ``prompt``, how to define ``functions`` or ``tools``, how to parse them out from the response, and how to execute them securely in your LLM applications.
We encourage our users to handle function calls on their own and we make the effort to make it easy to do so.

1. Get **maximum control and transparency** over your prompt and for researchers to help improve these capabilities.
2. Model-agnositc: Can switch to any model, either local or API based, without changing the code.
3. More powerful.




workflow
---------
It is basically show LLM a list of choices and prompt it to choose one or few of them.

1. Get a string format of available functions.
2. Add the tools to the prompt, and instruct LLM to call the function with **desired output format**. The output format will need: function_name, parameters to call the function.
3. Execute the LLM with the prompt and get the response.
4. Parse the response to the designed format. Execute the function and get the output.
5. Continue to the next step.

Additionally, we will need to manage a map of the function calls.

The basic function call is not complicated, but function calling can get more complicated:

1. Support more complicated data types in the arguments, such as an object.
2. There are different ways to call a function, the previous flow is more standard, but quite inflexible to extend to more complicated calls.
3.


1.Function formatting
--------------------------

Here is the formatting in OpenAI for the function ``get_current_weather``:

.. code-block:: python

    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        elif "san francisco" in location.lower():
            return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    formatting =
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }

Level 1: python default data types as positional and keyword arguments.
Level 2: json, list, dictionary, sequence as positional and keyword arguments.
level 3: data class as positional and keyword arguments.
level 4: a variable as positional and keyword arguments.

This applys to both the arguments and the output of the functions.



**What if im using a component?**

2.Prompt with tools/functions
-----------------------------
We will use a template to take into our tool options and we will give it instruction on how to call the function and if the parallel function calls are allowed.



3.Parse the response
---------------------

4.Execute the function
-----------------------

5.Prompt with last function call and response
----------------------------------------------




Parallel Function Calls
-------------------------
