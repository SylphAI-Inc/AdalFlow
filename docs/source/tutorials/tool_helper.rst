.. raw:: html

    <div style="display: flex; justify-content: flex-start; align-items: center; gap: 15px; margin-bottom: 20px;">
    <a target="_blank" href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_function_calls.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
    <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/adalflow_function_calls.py" target="_blank" style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
        <span style="vertical-align: middle;"> Open Source Code</span>
    </a>
    </div>

.. _tool_helper:

Function calls
===========================
.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_

Tools are means LLM can use to interact with the world beyond of its internal knowledge [1]_. Technically speaking, retrievers are tools to help LLM to get more relevant context, and memory is a tool for LLM to carry out a conversation.
Deciding when, which, and how to use a tool, and even to creating a tool is an agentic behavior:
Function calls is a process of showing LLM a list of funciton definitions and prompt it to choose one or few of them.
Many places use tools and function calls interchangably.

In this note we will covert function calls, including

1. Function call walkthrough
2. Overall design
3. Function call in action
4. MCP (Model Context Protocol) tools [2]_ as function calls.


Quick Walkthrough
--------------------
Users might already know of OpenAI's function call feature via its API (https://platform.openai.com/docs/guides/function-calling).

.. code-block:: python

   def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    import json

    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

For the above function, it is shown to LLM in the following format:

.. code-block:: python

    function_definition = {
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


Then the API will respond with a list of function names and parameters:

.. code-block:: python

    Function(arguments='{"location": "San Francisco, CA"}', name='get_current_weather')

Then the output will need to be parsed into arguments which are then passed to the function:

.. code-block:: python

    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

Scope and Design
----------------------------
Even with API, users have to (1) create the function definition, (2) Parse the response, (3) Execute the function.
What is missing in using API is: (1) How the function definitions are shown to LLM, (2) How the output format is instructured.

AdalFlow will provide built-in capabilities to do function calls simplily via prompt without relying on the tools API.

**Design Goals**

Asking LLM to call a function with keyword arguments is the simplest way of achieving the function call.
But it is limiting:

1. What if the argument value is a more complicated data structure?
2. What if you want to use a variable as an argument?

AdalFlow will also provide ``FunctionExpression`` where calling a function is asking LLM to write the code snippet of the function call directly:

.. code-block:: python

    'get_current_weather("San Francisco, CA", unit="celsius")'

This is not only more flexible, but also it is also a more efficient/compact way to call a function.

.. As a library, we prioritize the built-in function call capabilities via the normal prompt-response.
.. Function calls are often just a prerequisite for more complext agent behaviors.
.. This means we need to know how to form a ``prompt``, how to define ``functions`` or ``tools``, how to parse them out from the response, and how to execute them securely in your LLM applications.
.. We encourage our users to handle function calls on their own and we make the effort to make it easy to do so.

.. 1. Get **maximum control and transparency** over your prompt and for researchers to help improve these capabilities.
.. 2. Model-agnositc: Can switch to any model, either local or API based, without changing the code.
.. 3. More powerful.



**Data Models**

We have four ``DataClass`` models: :class:`FunctionDefinition<core.types.FunctionDefinition>`, :class:`Function<core.types.Function>`, :class:`FunctionExpression<core.types.FunctionExpression>`, and :class:`FunctionOutput<core.types.FunctionOutput>` to handle function calls.

These classes not only help with data structuring but also by being a subclass of ``DataClass``, it can be easily used in the prompt.
``Function`` has three important attributes: ``name``, ``args``, and ``kwargs`` for the function name, positional arguments and keyword arguments.
``FunctionExpression`` only has one action for the function call expression.
Both can be used to format the output in the prompt. We will demonstrate how to use it later.

**Components**

We have two components: :class:`FunctionTool<core.func_tool.FunctionTool>` and :class:`ToolManager<core.tool_manager.ToolManager>` to streamline the lifecyle of (1)
creating the function definition (2) formatting the prompt with the definitions and output format (3) parsing the response (4) executing the function.

``FunctionTool`` is a container of a single function. It handles the function definition and executing of the function. It supports both sync and async functions.
``ToolManager`` manages all tools. And it handles the execution and context_map that is used to parse the functions sercurely.

``ToolManager`` is simplified way to do function calls.

.. list-table::
    :header-rows: 1

    * -
      - Attribute/Method
      - Description
    * - Attributes
      - ``tools``
      - A list of tools managed by ToolManager. Each tool is an instance or a derivative of ``FunctionTool``.
    * -
      - ``context``
      - A dictionary combining tool definitions and additional context, used for executing function expressions.
    * - Methods
      - ``__init__``
      - Initializes a new ToolManager instance with tools and additional context. Tool can be ``FunctionTool`` or any function.
    * -
      - ``yaml_definitions``
      - Returns the YAML definitions of all tools managed by ToolManager.
    * -
      - ``json_definitions``
      - Returns the JSON definitions of all tools managed by ToolManager.
    * -
      - ``function_definitions``
      - Returns a list of function definitions for all tools.
    * -
      - ``parse_func_expr``
      - Parses a ``FunctionExpression`` and returns a ``Function`` object ready for execution.
    * -
      - ``execute_func``
      - Executes a given ``Function`` object and returns its output wrapped in ``FunctionOutput``. Support both sync and async functions.
    * -
      - ``execute_func_expr``
      - Parses and executes a ``FunctionExpression`` directly, returning the execution result as ``FunctionOutput``. Support both sync and async functions.
    * -
      - ``execute_func_expr_via_sandbox``
      - Execute the function expression via sandbox. Only support sync functions.
    * -
      - ``execute_func_expr_via_eval``
      - Execute the function expression via eval. Only support sync functions.

Function Call in Action
--------------------------

We will use the following functions as examples across this note:

.. code-block:: python

    from dataclasses import dataclass
    import numpy as np
    import time
    import asyncio


    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        time.sleep(1)
        return a * b


    def add(a: int, b: int) -> int:
        """Add two numbers."""
        time.sleep(1)
        return a + b


    async def divide(a: float, b: float) -> float:
        """Divide two numbers."""
        await asyncio.sleep(1)
        return float(a) / b


    async def search(query: str) -> List[str]:
        """Search for query and return a list of results."""
        await asyncio.sleep(1)
        return ["result1" + query, "result2" + query]


    def numpy_sum(arr: np.ndarray) -> float:
        """Sum the elements of an array."""
        return np.sum(arr)


    x = 2

    @dataclass
    class Point:
        x: int
        y: int


    def add_points(p1: Point, p2: Point) -> Point:
        return Point(p1.x + p2.x, p1.y + p2.y)

We delibrately cover both async and sync, examples of using variables and more complicated data structures as arguments.
We will demonstrate the structure and how to use each data model and component to call the above functions in different ways.

1. FunctionTool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, let's see how we help describe the function to LLM.

Use the above functions as examples, ``FunctionTool`` will generate the ``FunctionDefinition`` for each function automatically if the user did not pass it in.

.. code-block:: python

    from adalflow.core.func_tool import FunctionTool

    functions =[multiply, add, divide, search, numpy_sum, add_points]
    tools = [
        FunctionTool(fn=fn) for fn in functions
    ]
    for tool in tools:
        print(tool)

The printout shows three attributes for each function: ``fn``, ``_is_async``, and ``definition``.

.. code-block::

    FunctionTool(fn: <function multiply at 0x14d9d3f60>, async: False, definition: FunctionDefinition(func_name='multiply', func_desc='multiply(a: int, b: int) -> int\nMultiply two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'int'}, 'b': {'type': 'int'}}, 'required': ['a', 'b']}))
    FunctionTool(fn: <function add at 0x14d9e4040>, async: False, definition: FunctionDefinition(func_name='add', func_desc='add(a: int, b: int) -> int\nAdd two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'int'}, 'b': {'type': 'int'}}, 'required': ['a', 'b']}))
    FunctionTool(fn: <function divide at 0x14d9e40e0>, async: True, definition: FunctionDefinition(func_name='divide', func_desc='divide(a: float, b: float) -> float\nDivide two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'float'}, 'b': {'type': 'float'}}, 'required': ['a', 'b']}))
    FunctionTool(fn: <function search at 0x14d9e4180>, async: True, definition: FunctionDefinition(func_name='search', func_desc='search(query: str) -> List[str]\nSearch for query and return a list of results.', func_parameters={'type': 'object', 'properties': {'query': {'type': 'str'}}, 'required': ['query']}))
    FunctionTool(fn: <function numpy_sum at 0x14d9e4220>, async: False, definition: FunctionDefinition(func_name='numpy_sum', func_desc='numpy_sum(arr: numpy.ndarray) -> float\nSum the elements of an array.', func_parameters={'type': 'object', 'properties': {'arr': {'type': 'ndarray'}}, 'required': ['arr']}))
    FunctionTool(fn: <function add_points at 0x14d9e4360>, async: False, definition: FunctionDefinition(func_name='add_points', func_desc='add_points(p1: __main__.Point, p2: __main__.Point) -> __main__.Point\nNone', func_parameters={'type': 'object', 'properties': {'p1': {'type': 'Point', 'properties': {'x': {'type': 'int'}, 'y': {'type': 'int'}}, 'required': ['x', 'y']}, 'p2': {'type': 'Point', 'properties': {'x': {'type': 'int'}, 'y': {'type': 'int'}}, 'required': ['x', 'y']}}, 'required': ['p1', 'p2']}))

View the definition for ``add_point`` and also the ``get_current_weather`` function in dict format:

.. code-block:: python

    print(tools[-2].definition.to_dict())

The output will be:

.. code-block::

    {
        "func_name": "numpy_sum",
        "func_desc": "numpy_sum(arr: numpy.ndarray) -> float\nSum the elements of an array.",
        "func_parameters": {
            "type": "object",
            "properties": {"arr": {"type": "ndarray"}},
            "required": ["arr"],
            },
    }

Using ``to_json`` and ``to_yaml`` will directly get us the string that can be fed into the prompt.
And we prefer to use ``yaml`` format here as it is more token efficient:


We choose to describe the function not only with the docstring which is `Sum the elements of an array.` but also with the function signature which is `numpy_sum(arr: numpy.ndarray) -> float`.
This will give the LLM a view of the function at the code level and it helps with the function call.

.. note::
    Users should better use type hints and a good docstring to help LLM understand the function better.

In comparison, here is our definition for ``get_current_weather``:

.. code-block::

    {
        "func_name": "get_current_weather",
        "func_desc": "get_current_weather(location, unit='fahrenheit')\nGet the current weather in a given location",
        "func_parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "Any"},
                "unit": {"type": "Any", "default": "fahrenheit"},
            },
            "required": ["location"],
        },
    }

To execute function using function names requres us to manage a function map. Instead of using the raw function, we use ``FunctionTool`` instead for this context map.

.. code-block:: python

    context_map = {tool.definition.func_name: tool for tool in tools}

To execute a function, we can do:

.. code-block:: python

    function_name = "add"
    function_to_call = context_map[function_name]
    function_args = {"a": 1, "b": 2}
    function_response = function_to_call.call(**function_args)

If we use async function, we can use ``acall``.
``execute`` is a wrapper that you can call a function in both sync and async way regardless of the function type.
Check out the API documentation for more details.

2. ToolManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using ``ToolManager`` on all the above function:

.. code-block:: python

    from adalflow.core.tool_manager import ToolManager

    tool_manager = ToolManager(tools=functions)
    print(tool_manager)

The tool manager can take both ``FunctionTool``, function and async function.
The printout:

.. code-block::

    ToolManager(Tools: [FunctionTool(fn: <function multiply at 0x105e3b920>, async: False, definition: FunctionDefinition(func_name='multiply', func_desc='multiply(a: int, b: int) -> int\nMultiply two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'int'}, 'b': {'type': 'int'}}, 'required': ['a', 'b']})), FunctionTool(fn: <function add at 0x105e3bc40>, async: False, definition: FunctionDefinition(func_name='add', func_desc='add(a: int, b: int) -> int\nAdd two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'int'}, 'b': {'type': 'int'}}, 'required': ['a', 'b']})), FunctionTool(fn: <function divide at 0x104970220>, async: True, definition: FunctionDefinition(func_name='divide', func_desc='divide(a: float, b: float) -> float\nDivide two numbers.', func_parameters={'type': 'object', 'properties': {'a': {'type': 'float'}, 'b': {'type': 'float'}}, 'required': ['a', 'b']})), FunctionTool(fn: <function search at 0x104970400>, async: True, definition: FunctionDefinition(func_name='search', func_desc='search(query: str) -> List[str]\nSearch for query and return a list of results.', func_parameters={'type': 'object', 'properties': {'query': {'type': 'str'}}, 'required': ['query']})), FunctionTool(fn: <function numpy_sum at 0x1062a2840>, async: False, definition: FunctionDefinition(func_name='numpy_sum', func_desc='numpy_sum(arr: numpy.ndarray) -> float\nSum the elements of an array.', func_parameters={'type': 'object', 'properties': {'arr': {'type': 'ndarray'}}, 'required': ['arr']})), FunctionTool(fn: <function add_points at 0x106d691c0>, async: False, definition: FunctionDefinition(func_name='add_points', func_desc='add_points(p1: __main__.Point, p2: __main__.Point) -> __main__.Point\nNone', func_parameters={'type': 'object', 'properties': {'p1': {'type': 'Point', 'properties': {'x': {'type': 'int'}, 'y': {'type': 'int'}}, 'required': ['x', 'y']}, 'p2': {'type': 'Point', 'properties': {'x': {'type': 'int'}, 'y': {'type': 'int'}}, 'required': ['x', 'y']}}, 'required': ['p1', 'p2']}))], Additional Context: {})



We will show more how it can be used in the next section.

3. Function Call end-to-end
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, let us add prompt and start to do function calls via LLMs.
We use the following prompt to do a single function call.

.. code-block:: python

    template = r"""<START_OF_SYS_PROMPT>You have these tools available:
    {% if tools %}
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}.
    {{tool}}
    ------------------------
    {% endfor %}
    </TOOLS>
    {% endif %}
    <OUTPUT_FORMAT>
    {{output_format_str}}
    </OUTPUT_FORMAT>
    <END_OF_SYS_PROMPT>
    <START_OF_USER>: {{input_str}}<END_OF_USER>
    """

**Pass tools in the prompt**

We use `yaml` format here and show an example with less tools.

.. code-block:: python

    from adalflow.core.prompt_builder import Prompt

    prompt = Prompt(template=template)
    small_tool_manager = ToolManager(tools=tools[:2])

    renered_prompt = prompt(tools=small_tool_manager.yaml_definitions)
    print(renered_prompt)

The output is:

.. code-block::

    <START_OF_SYS_PROMPT>You have these tools available:
    <TOOLS>
    1.
    func_name: multiply
    func_desc: 'multiply(a: int, b: int) -> int

    Belongs to class: multiply

    Docstring: Multiply two numbers.

    '
    func_parameters:
    type: object
    properties:
        a:
        type: int
        b:
        type: int
    required:
    - a
    - b
    ------------------------
    2.
    func_name: add
    func_desc: 'add(a: int, b: int) -> int

    Belongs to class: add

    Docstring: Add two numbers.

    '
    func_parameters:
    type: object
    properties:
        a:
        type: int
        b:
        type: int
    required:
    - a
    - b
    ------------------------
    </TOOLS>
    <OUTPUT_FORMAT>
    None
    </OUTPUT_FORMAT>
    <END_OF_SYS_PROMPT>
    <START_OF_USER>: None<END_OF_USER>

**Pass the output format**

We have two ways to instruct LLM to call the function:

1. Using the function name and arguments, we will leverage ``Function`` as LLM's output data type.

.. code-block:: python

    from adalflow.core.types import Function

    output_data_class = Function
    output_format_str = output_data_class.to_json_signature(exclude=["thought", "args"])

    renered_prompt= prompt(output_format_str=output_format_str)
    print(renered_prompt)

We execluded both the ``thought`` and ``args`` as it is easier to use ``kwargs`` to represent the arguments.
The output is:

.. code-block::

    <START_OF_SYS_PROMPT>You have these tools available:
    <OUTPUT_FORMAT>
    {
        "name": "The name of the function (str) (optional)",
        "kwargs": "The keyword arguments of the function (Optional[Dict[str, object]]) (optional)"
    }
    </OUTPUT_FORMAT>
    <END_OF_SYS_PROMPT>
    <START_OF_USER>: None<END_OF_USER>



2. Using the function call expression for which we will use ``FunctionExpression``.

.. code-block:: python

    from adalflow.core.types import FunctionExpression

    output_data_class = FunctionExpression
    output_format_str = output_data_class.to_json_signature(exclude=["thought"])
    print(prompt(output_format_str=output_format_str))

The output is:

.. code-block::

    <START_OF_SYS_PROMPT>You have these tools available:
    <OUTPUT_FORMAT>
    {
        "action": "FuncName(<kwargs>) Valid function call expression. Example: \"FuncName(a=1, b=2)\" Follow the data type specified in the function parameters.e.g. for Type object with x,y properties, use \"ObjectType(x=1, y=2) (str) (required)"
    }
    </OUTPUT_FORMAT>
    <END_OF_SYS_PROMPT>
    <START_OF_USER>: None<END_OF_USER>

We will use :class:`components.output_parsers.outputs.JsonOutputParser` to streamline the formatting of our output data type.

.. code-block:: python

    from adalflow.components.output_parsers import JsonOutputParser

    func_parser = JsonOutputParser(data_class=Function, exclude_fields=["thought", "args"])
    instructions = func_parser.format_instructions()
    print(instructions)

The output is:

.. code-block::

    Your output should be formatted as a standard JSON instance with the following schema:
    ```
    {
        "name": "The name of the function (str) (optional)",
        "kwargs": "The keyword arguments of the function (Optional) (optional)"
    }
    ```
    -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
    -Use double quotes for the keys and string values.
    -Follow the JSON formatting conventions.


Function Output Format
**************************************************
Now, let's prepare our generator with the above prompt, ``Function`` data class, and ``JsonOutputParser``.

.. code-block:: python

    from adalflow.core.generator import Generator
    from adalflow.core.types import ModelClientType

    model_kwargs = {"model": "gpt-3.5-turbo"}
    prompt_kwargs = {
        "tools": tool_manager.yaml_definitions,
        "output_format_str": func_parser.format_instructions(),
    }
    generator = Generator(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=model_kwargs,
        template=template,
        prompt_kwargs=prompt_kwargs,
        output_processors=func_parser,
    )

**Run Queries**

We will use ``Function.from_dict`` to get the final output type from the json object. Here we use :meth:`core.tool_manager.ToolManager.execute_func` to execute it directly.

.. code-block:: python

    queries = [
        "add 2 and 3",
        "search for something",
        "add points (1, 2) and (3, 4)",
        "sum numpy array with arr = np.array([[1, 2], [3, 4]])",
        "multiply 2 with local variable x",
        "divide 2 by 3",
        "Add 5 to variable y",
    ]

    for idx, query in enumerate(queries):
        prompt_kwargs = {"input_str": query}
        print(f"\n{idx} Query: {query}")
        print(f"{'-'*50}")
        try:
            result = generator(prompt_kwargs=prompt_kwargs)
            # print(f"LLM raw output: {result.raw_response}")
            func = Function.from_dict(result.data)
            print(f"Function: {func}")
            func_output = tool_manager.execute_func(func)
            print(f"Function output: {func_output}")
        except Exception as e:
            print(
                f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
            )

From the output shown below, we get valide ``Function`` parsed as output for all queries.
However, we see it failed three function execution:
(1)function `add_points` due to its argument type is a data class, and `multiply` and the last `add` due to it is difficult to represent the local variable `x` and `y` in the function call.

.. code-block::

    0 Query: add 2 and 3
    --------------------------------------------------
    Function: Function(thought=None, name='add', args=[], kwargs={'a': 2, 'b': 3})
    Function output: FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 2, 'b': 3}), parsed_input=None, output=5, error=None)

    1 Query: search for something
    --------------------------------------------------
    Function: Function(thought=None, name='search', args=[], kwargs={'query': 'something'})
    Function output: FunctionOutput(name='search', input=Function(thought=None, name='search', args=(), kwargs={'query': 'something'}), parsed_input=None, output=['result1something', 'result2something'], error=None)

    2 Query: add points (1, 2) and (3, 4)
    --------------------------------------------------
    Function: Function(thought=None, name='add_points', args=[], kwargs={'p1': {'x': 1, 'y': 2}, 'p2': {'x': 3, 'y': 4}})
    Error at calling <function add_points at 0x117b98360>: 'dict' object has no attribute 'x'
    Function output: FunctionOutput(name='add_points', input=Function(thought=None, name='add_points', args=(), kwargs={'p1': {'x': 1, 'y': 2}, 'p2': {'x': 3, 'y': 4}}), parsed_input=None, output=None, error="'dict' object has no attribute 'x'")

    3 Query: sum numpy array with arr = np.array([[1, 2], [3, 4]])
    --------------------------------------------------
    Function: Function(thought=None, name='numpy_sum', args=[], kwargs={'arr': [[1, 2], [3, 4]]})
    Function output: FunctionOutput(name='numpy_sum', input=Function(thought=None, name='numpy_sum', args=(), kwargs={'arr': [[1, 2], [3, 4]]}), parsed_input=None, output=10, error=None)

    4 Query: multiply 2 with local variable x
    --------------------------------------------------
    Function: Function(thought=None, name='multiply', args=[], kwargs={'a': 2, 'b': 'x'})
    Function output: FunctionOutput(name='multiply', input=Function(thought=None, name='multiply', args=(), kwargs={'a': 2, 'b': 'x'}), parsed_input=None, output='xx', error=None)

    5 Query: divide 2 by 3
    --------------------------------------------------
    Function: Function(thought=None, name='divide', args=[], kwargs={'a': 2.0, 'b': 3.0})
    Function output: FunctionOutput(name='divide', input=Function(thought=None, name='divide', args=(), kwargs={'a': 2.0, 'b': 3.0}), parsed_input=None, output=0.6666666666666666, error=None)

    6 Query: Add 5 to variable y
    --------------------------------------------------
    Function: Function(thought=None, name='add', args=[], kwargs={'a': 5, 'b': 'y'})
    Error at calling <function add at 0x11742eca0>: unsupported operand type(s) for +: 'int' and 'str'
    Function output: FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 5, 'b': 'y'}), parsed_input=None, output=None, error="unsupported operand type(s) for +: 'int' and 'str'")


.. note::
    If users prefer to use Function, to incress the success rate, make sure your function arguments are dict based for class object. You can always convert it to a class from a dict.


FunctionExpression Output Format
**************************************************
We will adapt the above code easily using tool manager to use ``FunctionExpression`` as the output format.
We will use FunctionExpression this time in the parser. And we added the necessary context to handle the local variable `x`, `y`, and `np.array`.

.. code-block:: python

    tool_manager = ToolManager(
        tools=functions,
        additional_context={"x": x, "y": 0, "np.array": np.array, "np": np},
    )
    func_parser = JsonOutputParser(data_class=FunctionExpression)

Additionally, we can also pass the ``additional_context`` to LLM using the following prompt after the <TOOLS>

.. code-block:: python

    context = r"""<CONTEXT>
    Your function expression also have access to these context:
    {{context_str}}
    </CONTEXT>
    """

This time, let us try to execute all function concurrently and treating them all as async functions.

.. code-block:: python

    async def run_async_function_call(self, generator, tool_manager):
        answers = []
        start_time = time.time()
        tasks = []
        for idx, query in enumerate(queries):
            tasks.append(self.process_query(idx, query, generator, tool_manager))

        results = await asyncio.gather(*tasks)
        answers.extend(results)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time :.2f} seconds")
        return answers

    async def process_query(self, idx, query, generator, tool_manager: ToolManager):
        print(f"\n{idx} Query: {query}")
        print(f"{'-'*50}")
        try:
            result = generator(prompt_kwargs={"input_str": query})
            func_expr = FunctionExpression.from_dict(result.data)
            print(f"Function_expr: {func_expr}")
            func = tool_manager.parse_func_expr(func_expr)
            func_output = await tool_manager.execute_func_async(func)
            print(f"Function output: {func_output}")
            return func_output
        except Exception as e:
            print(
                f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
            )
            return None

In this case, we used :meth:`core.tool_manager.ToolManager.parse_func_expr` and :meth:`core.tool_manager.ToolManager.execute_func` to execute the function.
Or we can directly use :meth:`core.tool_manager.ToolManager.execute_func_expr` to execute the function expression. Both are equivalent.

From the output shown below, this time we get all function calls executed successfully.

.. code-block::

    0 Query: add 2 and 3
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='add(a=2, b=3)')

    1 Query: search for something
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='search(query="something")')

    2 Query: add points (1, 2) and (3, 4)
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='add_points(p1=Point(x=1, y=2), p2=Point(x=3, y=4))')

    3 Query: sum numpy array with arr = np.array([[1, 2], [3, 4]])
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='numpy_sum(arr=np.array([[1, 2], [3, 4]]))')

    4 Query: multiply 2 with local variable x
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='multiply(a=2, b=2)')

    5 Query: divide 2 by 3
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='divide(a=2.0, b=3.0)')

    6 Query: Add 5 to variable y
    --------------------------------------------------
    Function_expr: FunctionExpression(thought=None, action='add(a=0, b=5)')
    Function output: FunctionOutput(name='add_points', input=Function(thought=None, name='add_points', args=(), kwargs={'p1': Point(x=1, y=2), 'p2': Point(x=3, y=4)}), parsed_input=None, output=Point(x=4, y=6), error=None)
    Function output: FunctionOutput(name='numpy_sum', input=Function(thought=None, name='numpy_sum', args=(), kwargs={'arr': array([[1, 2],
        [3, 4]])}), parsed_input=None, output=10, error=None)
    Function output: FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 2, 'b': 3}), parsed_input=None, output=5, error=None)
    Function output: FunctionOutput(name='multiply', input=Function(thought=None, name='multiply', args=(), kwargs={'a': 2, 'b': 2}), parsed_input=None, output=4, error=None)
    Function output: FunctionOutput(name='search', input=Function(thought=None, name='search', args=(), kwargs={'query': 'something'}), parsed_input=None, output=['result1something', 'result2something'], error=None)
    Function output: FunctionOutput(name='divide', input=Function(thought=None, name='divide', args=(), kwargs={'a': 2.0, 'b': 3.0}), parsed_input=None, output=0.6666666666666666, error=None)
    Function output: FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 0, 'b': 5}), parsed_input=None, output=5, error=None)


Parallel Function Calls
-------------------------

We will slightly adapt the output format instruction to get it output json array, which can still be parsed with a json parser.

.. code-block:: python

    multple_function_call_template = r"""<SYS>You have these tools available:
    {% if tools %}
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}.
    {{tool}}
    ------------------------
    {% endfor %}
    </TOOLS>
    {% endif %}
    <OUTPUT_FORMAT>
    Here is how you call one function.
    {{output_format_str}}
    -Always return a List using `[]` of the above JSON objects, even if its just one item.
    </OUTPUT_FORMAT>
    <SYS>
    {{input_str}}
    You:
    """

As LLM has problem calling ``add_point``, we will add one example and we will generate it with :meth:`core.types.FunctionExpression.from_function`.
We will update our outputparser to use the example:

.. code-block:: python

    example = FunctionExpression.from_function(
            func=add_points, p1=Point(x=1, y=2), p2=Point(x=3, y=4)
    )
    func_parser = JsonOutputParser(
            data_class=FunctionExpression, examples=[example]
    )

Here is the updated output format in the prompt:

.. code-block::

    <OUTPUT_FORMAT>
    Here is how you call one function.
    Your output should be formatted as a standard JSON instance with the following schema:
    ```
    {
        "action": "FuncName(<kwargs>)                 Valid function call expression.                 Example: \"FuncName(a=1, b=2)\"                 Follow the data type specified in the function parameters.                e.g. for Type object with x,y properties, use \"ObjectType(x=1, y=2) (str) (required)"
    }
    ```
    Here is an example:
    ```
    {
        "action": "add_points(p1=Point(x=1, y=2), p2=Point(x=3, y=4))"
    }
    ```
    -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
    -Use double quotes for the keys and string values.
    -Follow the JSON formatting conventions.
    Awlays return a List using `[]` of the above JSON objects. You can have length of 1 or more.
    Do not call multiple functions in one action field.
    </OUTPUT_FORMAT>

This case, we will show the response from using `execute_func_expr_via_sandbox` to execute the function expression.

.. code-block:: python

    for idx in range(0, len(queries), 2):
        query = " and ".join(queries[idx : idx + 2])
        prompt_kwargs = {"input_str": query}
        print(f"\n{idx} Query: {query}")
        print(f"{'-'*50}")
        try:
            result = generator(prompt_kwargs=prompt_kwargs)
            # print(f"LLM raw output: {result.raw_response}")
            func_expr: List[FunctionExpression] = [
                FunctionExpression.from_dict(item) for item in result.data
            ]
            print(f"Function_expr: {func_expr}")
            for expr in func_expr:
                func_output = tool_manager.execute_func_expr_via_sandbox(expr)
                print(f"Function output: {func_output}")
        except Exception as e:
            print(
                f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
            )

By using an example to help with calling ``add_point``, we can now successfully execute all function calls.

.. code-block:: python

    0 Query: add 2 and 3 and search for something
    --------------------------------------------------
    Function_expr: [FunctionExpression(thought=None, action='add(a=2, b=3)'), FunctionExpression(thought=None, action='search(query="something")')]
    Function output: FunctionOutput(name='add(a=2, b=3)', input=FunctionExpression(thought=None, action='add(a=2, b=3)'), parsed_input=None, output=FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 2, 'b': 3}), parsed_input=None, output=5, error=None), error=None)
    Function output: FunctionOutput(name='search(query="something")', input=FunctionExpression(thought=None, action='search(query="something")'), parsed_input=None, output=FunctionOutput(name='search', input=Function(thought=None, name='search', args=(), kwargs={'query': 'something'}), parsed_input=None, output=['result1something', 'result2something'], error=None), error=None)

    2 Query: add points (1, 2) and (3, 4) and sum numpy array with arr = np.array([[1, 2], [3, 4]])
    --------------------------------------------------
    Function_expr: [FunctionExpression(thought=None, action='add_points(p1=Point(x=1, y=2), p2=Point(x=3, y=4))'), FunctionExpression(thought=None, action='numpy_sum(arr=[[1, 2], [3, 4]])')]
    Function output: FunctionOutput(name='add_points(p1=Point(x=1, y=2), p2=Point(x=3, y=4))', input=FunctionExpression(thought=None, action='add_points(p1=Point(x=1, y=2), p2=Point(x=3, y=4))'), parsed_input=None, output=FunctionOutput(name='add_points', input=Function(thought=None, name='add_points', args=(), kwargs={'p1': Point(x=1, y=2), 'p2': Point(x=3, y=4)}), parsed_input=None, output=Point(x=4, y=6), error=None), error=None)
    Function output: FunctionOutput(name='numpy_sum(arr=[[1, 2], [3, 4]])', input=FunctionExpression(thought=None, action='numpy_sum(arr=[[1, 2], [3, 4]])'), parsed_input=None, output=FunctionOutput(name='numpy_sum', input=Function(thought=None, name='numpy_sum', args=(), kwargs={'arr': [[1, 2], [3, 4]]}), parsed_input=None, output=10, error=None), error=None)

    4 Query: multiply 2 with local variable x and divide 2 by 3
    --------------------------------------------------
    Function_expr: [FunctionExpression(thought=None, action='multiply(a=2, b=x)'), FunctionExpression(thought=None, action='divide(a=2.0, b=3.0)')]
    Function output: FunctionOutput(name='multiply(a=2, b=x)', input=FunctionExpression(thought=None, action='multiply(a=2, b=x)'), parsed_input=None, output=FunctionOutput(name='multiply', input=Function(thought=None, name='multiply', args=(), kwargs={'a': 2, 'b': 2}), parsed_input=None, output=4, error=None), error=None)
    Function output: FunctionOutput(name='divide(a=2.0, b=3.0)', input=FunctionExpression(thought=None, action='divide(a=2.0, b=3.0)'), parsed_input=None, output=FunctionOutput(name='divide', input=Function(thought=None, name='divide', args=(), kwargs={'a': 2.0, 'b': 3.0}), parsed_input=None, output=0.6666666666666666, error=None), error=None)

    6 Query: Add 5 to variable y
    --------------------------------------------------
    Function_expr: [FunctionExpression(thought=None, action='add(a=y, b=5)')]
    Function output: FunctionOutput(name='add(a=y, b=5)', input=FunctionExpression(thought=None, action='add(a=y, b=5)'), parsed_input=None, output=FunctionOutput(name='add', input=Function(thought=None, name='add', args=(), kwargs={'a': 0, 'b': 5}), parsed_input=None, output=5, error=None), error=None)


MCP (Model Context Protocol) Tools
-------------------------

In addition to the above, AdalFlow also supports MCP tools [2]_, which enable interaction with LLM models and their environments through the MCP protocol.
Compared to the general function call defined above, MCP provides a unified interface between different tool providers and model vendors.
Importantly, you don't need to rewrite every tools. Instead, you can directly re-use the tools from the marketplace like `smithery.ai <https://smithery.ai/>`_.


**Configuring an MCP Client**. To configure a standard input/output (stdio) MCP client with your own server. An example of python server implementation is at `tutorials/mcp_agent/mcp_calculator_server.py`.

.. code-block:: python

    from mcp import StdioServerParameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_calculator_server.py"],
    )

To configure a Server-Sent Events (SSE) MCP client, you can use MCP server URLs from https://server.smithery.ai:

.. code-block:: python

    smithery_api_key = os.environ.get("SMITHERY_API_KEY")
    smithery_server_id = "@nickclyde/duckduckgo-mcp-server"
    server_params = MCPServerStreamableHttpParams(url=f"https://server.smithery.ai/{smithery_server_id}/mcp?api_key={smithery_api_key}")

**Executing MCP Tools**. You can execute an MCP tool as an asynchronous function. The `MCPFunctionTool` can be used in the same way as a general `FunctionTool`:

.. code-block:: python

    import asyncio
    from adalflow.core.mcp_tool import MCPFunctionTool
    # Get tools from the server
    async with mcp_session_context(server_params) as session:
        tools = await session.list_tools()
    async def call_async_function():
        tool = MCPFunctionTool(server_params, tools[0])
        output = await tool.acall()

    asyncio.run(call_async_function())


**Using MCP Tools from Marketplace**. You can reuse the tools from marketplace. Below is a step-by-step example.

1. Find a tool at smithery.ai. For example, https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server.
2. Load the server. A MCP server provides multiple tools. With `MCPToolManager`, you can easily load them as `MCPFunctionTool` instances all at once. First, create a client manager:

.. code-block:: python

    from adalflow.core.mcp_tool import MCPToolManager
    mcp_client_manager = MCPToolManager(server_params)

You can add servers to the manager in several ways. For example, we use the DuckDuckGo parameters (`@nickclyde/duckduckgo-mcp-server <https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server>`_), choose one of the following options:

.. code-block:: python

    # ======= Example 1: Add via stdio command. =======
    manager.add_server("duckduckgo-mcp-server", StdioServerParameters(
        command="npx",  # Command to run the server
        args=[
            "-y",
            "@smithery/cli@latest",
            "run",
            "@nickclyde/duckduckgo-mcp-server",
            "--key",
            "smithery-api-key"
        ],
    ))

    # ======= Example 2: Load servers from a JSON file. =======
    json_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
    manager.add_servers_from_json_file(json_path)

    # ======= Example 3: Load server from SSE URL. =======
    smithery_api_key = os.environ.get("SMITHERY_API_KEY")
    smithery_server_id = "@nickclyde/duckduckgo-mcp-server"
    mcp_server_url = f"https://server.smithery.ai/{smithery_server_id}/mcp?api_key={smithery_api_key}"
    manager.add_server("duckduckgo-mcp-server", mcp_server_url)

An example `mcp_servers.json` file:

.. code-block:: json

    {
        "mcpServers": {
            "duckduckgo-mcp-server": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@nickclyde/duckduckgo-mcp-server",
                    "--key",
                    "smithery-api-key"
                ]
            }
        }
    }

3. Use the manager to create `MCPFunctionTool` instances.
You can list all tools available on the server and use it with agent. A complete example of using MCP tools with agents can be found under the `tutorials/mcp_agent` folder.

.. code-block:: python

    tools = await mcp_client_manager.get_all_tools()  # List[MCPFunctionTool]
    for tool in tools:
        sig = tool.definition.func_desc.split('\n')[0]
        print(f"- Tool: {tool.definition.func_name}, Signature: {sig}")

Example outputs when we list the tools from DuckDuckGo

.. code-block:: python

    Listing tools for server: duckduckgo-mcp-server
    2025-06-03 17:15:22 - [mcp_tool.py:36:mcp_session_context] - 📡 Initializing connection...
    2025-06-03 17:15:23 - [mcp_tool.py:38:mcp_session_context] - ✅ Connection established!

    🗂️  Available Resources:

    🔧 Available Tools:
    • search:
        Search DuckDuckGo and return formatted results.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10)
            ctx: MCP context for logging

    • fetch_content:
        Fetch and parse content from a webpage URL.

        Args:
            url: The webpage URL to fetch content from
            ctx: MCP context for logging

4. Use or optimize prompts or agents with the function as if you are using the FunctionTool.
With the tool, we can create an agent and ask it to search for news.

.. code-block:: python

    agent = ReActAgent(
        tools=tools,
        ...
    )
    agent.call("Use DuckDuckGo to search for the winner on European Championship in 2025.")

The example output of a step execution is shown below.

.. code-block:: python

    2025-06-03 17:15:43 - [react.py:468:_execute_action_eval_mode] - Step 1:
    StepOutput(step=1, action=Function(thought='The search results indicate that the UEFA European Championship (Euros) was last held in 2024, with Spain winning. There are no results for a winner of the UEFA European Championship in 2025, which suggests it has not occurred yet. Therefore, I will conclude that there is no winner for the European Championship in 2025.', name='finish', args=[], kwargs={'answer': 'The UEFA European Championship in 2025 has not occurred yet, so there is no winner.', 'id': None}), function=None, observation='The UEFA European Championship in 2025 has not occurred yet, so there is no winner.')

.. admonition:: References
   :class: highlight

   .. [1] OpenAI tools API: https://beta.openai.com/docs/api-reference/tools
   .. [2] Anthropic MCP document: https://modelcontextprotocol.io/introduction

.. admonition:: API References
   :class: highlight

   - :class:`core.types.FunctionDefinition`
   - :class:`core.types.Function`
   - :class:`core.types.FunctionExpression`
   - :class:`core.types.FunctionOutput`
   - :class:`core.func_tool.FunctionTool`
   - :func:`core.mcp_tool.MCPFunctionTool`
   - :func:`core.mcp_tool.MCPToolManager`
   - :class:`core.tool_manager.ToolManager`
   - :func:`core.functional.get_fun_schema`
   - :func:`core.functional.parse_function_call_expr`
   - :func:`core.functional.sandbox_execute`
   - :func:`core.functional.generate_function_call_expression_from_callable`
