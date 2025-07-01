import pytest
import inspect
import asyncio
import time
from dataclasses import dataclass

from adalflow.core.func_tool import FunctionTool, FunctionType
from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import FunctionDefinition, Function, FunctionOutput
from adalflow.core.component import Component
from adalflow.core.container import ComponentList


@dataclass
class User:
    id: int
    name: str


# Define some dummy functions for testing purposes
def sync_add(x, y, user: User = User(id=1, name="John")):
    return x + y


async def async_add(x, y):
    return x + y

# sync generator
def sync_generator():
    yield "sync generator result"

# async generator
async def async_generator():
    yield "async generator result"


metadata = FunctionDefinition(func_desc="A simple addition tool", func_name="add")


# use normal functions, with only eval mode
def test_function_tool_sync():
    tool = FunctionTool(definition=metadata, fn=sync_add)
    print(
        f"tool: {tool}, tool.metadata: {tool.definition}, tool.fn: {tool.fn}, tool.async: {tool._is_async}"
    )

    output = tool(1, 2)  # Using __call__ which proxies to call()
    assert output.output == 3
    assert output.name == "add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == (1, 2)


def test_function_tool_async():
    # use default metadata
    tool = FunctionTool(fn=async_add)

    import asyncio

    output = asyncio.run(tool.acall(3, 4))
    assert output.output == 7
    assert output.name == "async_add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == (3, 4)

    # call with sync call should work too 
    output = tool.call(3, 4)
    assert output.output == 7
    assert output.name == "async_add", "The name should be set to the function name"
    assert hasattr(output.input, "args")
    assert output.input.args == (3, 4)


# =============== MERGED TESTS FROM dev_function_tool.py ===============

# Helper functions for comprehensive type and execution testing
def sync_fun():
    """I'm a sync function"""
    return "sync"


async def async_fun():
    """I'm an async function"""
    return "async"


def sync_yield_fun():
    """I'm a sync generator function"""
    yield "sync_yield"


async def async_yield_fun():
    """I'm an async generator function"""
    yield "async_yield"


def test_function_type_detection():
    """Test function type detection using inspect module and FunctionTool.detect_function_type"""
    print("=== Function Type Detection with inspect ===")
    
    functions = [
        ("sync_fun", sync_fun),
        ("async_fun", async_fun), 
        ("sync_yield_fun", sync_yield_fun),
        ("async_yield_fun", async_yield_fun)
    ]
    
    for name, func in functions:
        print(f"Function: {name}")
        print(f"  iscoroutinefunction: {inspect.iscoroutinefunction(func)}")
        print(f"  isgeneratorfunction: {inspect.isgeneratorfunction(func)}")
        print(f"  isasyncgenfunction: {inspect.isasyncgenfunction(func)}")
        print(f"  isfunction: {inspect.isfunction(func)}")
        print(f"  ismethod: {inspect.ismethod(func)}")
        print(f"  iscallable: {callable(func)}")
        print(f"  signature: {inspect.signature(func)}")
        
        # Test FunctionTool detection
        detected_type = FunctionTool.detect_function_type(func)
        print(f"  FunctionTool detected type: {detected_type}")
        print()
        
        # Verify expected types
        if name == "sync_fun":
            assert detected_type == FunctionType.SYNC
        elif name == "async_fun":
            assert detected_type == FunctionType.ASYNC
        elif name == "sync_yield_fun":
            assert detected_type == FunctionType.SYNC_GENERATOR
        elif name == "async_yield_fun":
            assert detected_type == FunctionType.ASYNC_GENERATOR


def test_function_tool_properties():
    """Test FunctionTool properties and initialization with different function types"""
    print("=== FunctionTool Properties Testing ===")
    
    # Create function tools
    sync_tool = FunctionTool(sync_fun)
    async_tool = FunctionTool(async_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    tools = [
        ("sync_tool", sync_tool, FunctionType.SYNC),
        ("async_tool", async_tool, FunctionType.ASYNC),
        ("sync_yield_tool", sync_yield_tool, FunctionType.SYNC_GENERATOR),
        ("async_yield_tool", async_yield_tool, FunctionType.ASYNC_GENERATOR)
    ]
    
    for name, tool, expected_type in tools:
        print(f"Testing {name}:")
        print(f"  Function type: {tool.function_type}")
        print(f"  Is async: {tool._is_async}")
        print(f"  Tool: {tool}")
        
        # Verify properties
        assert tool.function_type == expected_type
        assert tool._is_async == (expected_type in [FunctionType.ASYNC, FunctionType.ASYNC_GENERATOR])
        print()


def test_sync_execution_comprehensive():
    """Test comprehensive sync execution of all function types"""
    print("=== Comprehensive Sync Execution Testing ===")
    
    sync_tool = FunctionTool(sync_fun)
    async_tool = FunctionTool(async_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    tools_to_test = [
        ("sync_fun", sync_tool),
        ("async_fun", async_tool),
        ("sync_yield_fun", sync_yield_tool),
        ("async_yield_fun", async_yield_tool)
    ]
    
    for name, tool in tools_to_test:
        print(f"Testing {name}:")
        print(f"  Function type: {tool.function_type}")
        print(f"  Is async: {tool._is_async}")
        
        try:
            # Call the function using call()
            result = tool.call()
            
            # Check output type
            print(f"  Result type: {type(result).__name__}")
            
            # Check if it's a FunctionOutput
            if hasattr(result, 'output'):
                print(f"  Result.output type: {type(result.output).__name__}")
                print(f"  Result.output: {result.output}")
                
                # For generators, test iteration
                if tool.function_type.name in ['SYNC_GENERATOR', 'ASYNC_GENERATOR']:
                    print(f"  Testing generator iteration:")
                    
                    if tool.function_type.name == 'SYNC_GENERATOR':
                        # Sync generator - iterate normally
                        generator = result.output
                        for i, item in enumerate(generator):
                            print(f"    Yield {i}: {item}")
                            assert item == "sync_yield"
                    
                    elif tool.function_type.name == 'ASYNC_GENERATOR':
                        # Async generator - should be collected into a list by call()
                        items = result.output
                        print(f"    Collected items: {items}")
                        assert isinstance(items, list)
                        assert items == ["async_yield"]
                
                # For regular functions, check the value
                else:
                    if name == "sync_fun":
                        assert result.output == "sync"
                    elif name == "async_fun":
                        assert result.output == "async"
            
            # Check other FunctionOutput attributes
            if hasattr(result, 'name'):
                print(f"  Function name: {result.name}")
            if hasattr(result, 'input'):
                print(f"  Function input: {result.input}")
            if hasattr(result, 'error'):
                print(f"  Function error: {result.error}")
                
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Error type: {type(e).__name__}")
            pytest.fail(f"Sync execution failed for {name}: {e}")
        
        print("-" * 50)


@pytest.mark.asyncio
async def test_async_execution_comprehensive():
    """Test comprehensive async execution of all function types"""
    print("=== Comprehensive Async Execution Testing ===")
    
    sync_tool = FunctionTool(sync_fun)
    async_tool = FunctionTool(async_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    tools_to_test = [
        ("sync_fun", sync_tool),
        ("async_fun", async_tool),
        ("sync_yield_fun", sync_yield_tool),
        ("async_yield_fun", async_yield_tool)
    ]
    
    for name, tool in tools_to_test:
        print(f"Testing {name}:")
        print(f"  Function type: {tool.function_type}")
        print(f"  Is async: {tool._is_async}")
        
        try:
            # Call the function using acall
            result = await tool.acall()
            
            # Check output type
            print(f"  Result type: {type(result).__name__}")
            
            # Check if it's a FunctionOutput
            if hasattr(result, 'output'):
                print(f"  Result.output type: {type(result.output).__name__}")
                print(f"  Result.output: {result.output}")
                
                # For generators, test iteration
                if tool.function_type.name in ['SYNC_GENERATOR', 'ASYNC_GENERATOR']:
                    print(f"  Testing generator iteration:")
                    
                    if tool.function_type.name == 'SYNC_GENERATOR':
                        # Sync generator - iterate normally
                        generator = result.output
                        for i, item in enumerate(generator):
                            print(f"    Yield {i}: {item}")
                            assert item == "sync_yield"
                    
                    elif tool.function_type.name == 'ASYNC_GENERATOR':
                        # Async generator - iterate with async for
                        async_generator = result.output
                        i = 0
                        async for item in async_generator:
                            print(f"    Yield {i}: {item}")
                            assert item == "async_yield"
                            i += 1
                
                # For regular functions, check the value
                else:
                    if name == "sync_fun":
                        assert result.output == "sync"
                    elif name == "async_fun":
                        assert result.output == "async"
            
            # Check other FunctionOutput attributes
            if hasattr(result, 'name'):
                print(f"  Function name: {result.name}")
            if hasattr(result, 'input'):
                print(f"  Function input: {result.input}")
            if hasattr(result, 'error'):
                print(f"  Function error: {result.error}")
                
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Error type: {type(e).__name__}")
            pytest.fail(f"Async execution failed for {name}: {e}")
        
        print("-" * 50)


def test_execution_timing():
    """Test execution timing for different function types"""
    print("=== Testing Sequential Execution Timing ===")
    
    sync_tool = FunctionTool(sync_fun)
    async_tool = FunctionTool(async_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    tools_to_test = [
        ("sync_fun", sync_tool),
        ("async_fun", async_tool),
        ("sync_yield_fun", sync_yield_tool),
        ("async_yield_fun", async_yield_tool)
    ]
    
    start_time = time.time()
    
    try:
        # Run all tools sequentially
        results = []
        for name, tool in tools_to_test:
            print(f"Calling {name}...")
            result = tool.call()
            results.append(result)
            print(f"  Completed {name}")
        
        end_time = time.time()
        print(f"\nSequential execution completed in {end_time - start_time:.2f} seconds")
        
        print("\nSequential execution results:")
        for i, (name, _) in enumerate(tools_to_test):
            result = results[i]
            if hasattr(result, 'output'):
                print(f"  {name}: {result.output}")
            
    except Exception as e:
        print(f"Sequential execution error: {e}")
        pytest.fail(f"Sequential execution failed: {e}")


def test_generator_consumption_patterns():
    """Test different generator consumption patterns"""
    print("=== Testing Generator Consumption Patterns ===")
    
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    # Test sync generator
    print("Testing sync generator consumption:")
    sync_result = sync_yield_tool.call()
    sync_gen = sync_result.output
    print(f"  Generator type: {type(sync_gen).__name__}")
    
    # Consume the generator
    items = list(sync_gen)
    print(f"  Consumed items: {items}")
    assert items == ["sync_yield"]
    
    # Test async generator (should be collected into list in sync mode)
    print("\nTesting async generator consumption (sync mode):")
    async_result = async_yield_tool.call()
    async_items = async_result.output
    print(f"  Collected items type: {type(async_items).__name__}")
    print(f"  Collected items: {async_items}")
    assert isinstance(async_items, list)
    assert async_items == ["async_yield"]


def test_error_handling():
    """Test error handling for different function types"""
    print("=== Testing Error Handling ===")
    
    def error_sync_fun():
        """A sync function that raises an error"""
        raise ValueError("Sync function error")
    
    async def error_async_fun():
        """An async function that raises an error"""
        await asyncio.sleep(0.01)  # Small delay to ensure it's actually async
        raise ValueError("Async function error")
    
    error_sync_tool = FunctionTool(error_sync_fun)
    error_async_tool = FunctionTool(error_async_fun)
    
    # Test sync error - should be captured in FunctionOutput
    print("Testing sync function with error:")
    result = error_sync_tool.call()
    print(f"  Result: {result}")
    assert hasattr(result, 'error')
    assert result.error is not None
    print(f"  Error captured: {result.error}")
    
    # Test async error in sync mode - should be captured in FunctionOutput
    print("\nTesting async function with error (sync mode):")
    result = error_async_tool.call()
    print(f"  Result: {result}")
    assert hasattr(result, 'error')
    assert result.error is not None
    print(f"  Error captured: {result.error}")


@pytest.mark.asyncio
async def test_error_handling_async():
    """Test error handling in async mode"""
    print("=== Testing Error Handling (Async Mode) ===")
    
    def error_sync_fun():
        """A sync function that raises an error"""
        raise ValueError("Sync function error")
    
    async def error_async_fun():
        """An async function that raises an error"""
        await asyncio.sleep(0.01)
        raise ValueError("Async function error")
    
    error_sync_tool = FunctionTool(error_sync_fun)
    error_async_tool = FunctionTool(error_async_fun)
    
    # Test sync error in async mode
    print("Testing sync function with error (async mode):")
    result = await error_sync_tool.acall()
    print(f"  Result: {result}")
    assert hasattr(result, 'error')
    assert result.error is not None
    print(f"  Error captured: {result.error}")
    
    # Test async error in async mode
    print("\nTesting async function with error (async mode):")
    result = await error_async_tool.acall()
    print(f"  Result: {result}")
    assert hasattr(result, 'error')
    assert result.error is not None
    print(f"  Error captured: {result.error}")


# =============== END OF MERGED TESTS ===============


# =============== TOOL MANAGER EXECUTE TESTS ===============

def test_tool_manager_execute_func():
    """Test ToolManager.execute_func with different function types"""
    print("=== Testing ToolManager.execute_func ===")
    
    # Create test functions
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers"""
        return x + y
    
    def multiply_numbers(x: int, y: int) -> int:
        """Multiply two numbers"""
        return x * y
    
    # Create function tools
    add_tool = FunctionTool(add_numbers)
    multiply_tool = FunctionTool(multiply_numbers)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[add_tool, multiply_tool])
    
    # Test execute_func with different functions
    print("Testing add function:")
    add_func = Function(name="add_numbers", args=[3, 5], kwargs={})
    result = tool_manager.execute_func(add_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 8
    assert result.name == "add_numbers"
    
    print("\nTesting multiply function:")
    multiply_func = Function(name="multiply_numbers", args=[4, 6], kwargs={})
    result = tool_manager.execute_func(multiply_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 24
    assert result.name == "multiply_numbers"
    
    # Test with kwargs
    print("\nTesting function with kwargs:")
    add_func_kwargs = Function(name="add_numbers", args=[2], kwargs={"y": 7})
    result = tool_manager.execute_func(add_func_kwargs)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 9
    
    print("ToolManager.execute_func tests passed!")


@pytest.mark.asyncio
async def test_tool_manager_execute_func_async():
    """Test ToolManager.execute_func_async with sync and async functions"""
    print("=== Testing ToolManager.execute_func_async ===")
    
    # Create sync function
    def sync_divide(x: int, y: int) -> float:
        """Divide two numbers"""
        return x / y
    
    # Create async function
    async def async_power(x: int, y: int) -> int:
        """Raise x to the power of y"""
        await asyncio.sleep(0.01)  # Small delay to ensure it's async
        return x ** y
    
    # Create function tools
    divide_tool = FunctionTool(sync_divide)
    power_tool = FunctionTool(async_power)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[divide_tool, power_tool])
    
    # Test execute_func_async with sync function
    print("Testing sync function via async execution:")
    divide_func = Function(name="sync_divide", args=[10, 2], kwargs={})
    result = await tool_manager.execute_func_async(divide_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 5.0
    assert result.name == "sync_divide"
    
    # Test execute_func_async with async function
    print("\nTesting async function via async execution:")
    power_func = Function(name="async_power", args=[2, 3], kwargs={})
    result = await tool_manager.execute_func_async(power_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 8
    assert result.name == "async_power"
    
    # Test with kwargs
    print("\nTesting async function with kwargs:")
    power_func_kwargs = Function(name="async_power", args=[3], kwargs={"y": 2})
    result = await tool_manager.execute_func_async(power_func_kwargs)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.output == 9
    
    print("ToolManager.execute_func_async tests passed!")


def test_tool_manager_execute_func_streaming():
    """Test ToolManager.execute_func with streaming parameter"""
    print("=== Testing ToolManager.execute_func with streaming ===")
    
    # Create a generator function for streaming
    def stream_numbers(count: int):
        """Generate a sequence of numbers"""
        for i in range(count):
            yield i
        return count  # Final return value
    
    # Create function tool
    stream_tool = FunctionTool(stream_numbers)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[stream_tool])
    
    # Test without streaming
    print("Testing without streaming:")
    stream_func = Function(name="stream_numbers", args=[3], kwargs={})
    result = tool_manager.execute_func(stream_func, stream=False)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    
    # Test with streaming enabled
    print("\nTesting with streaming enabled:")
    result_stream = tool_manager.execute_func(stream_func, stream=True)
    print(f"  Result: {result_stream}")
    assert isinstance(result_stream, FunctionOutput)
    
    print("ToolManager.execute_func streaming tests passed!")


def test_tool_manager_execute_func_error_handling():
    """Test error handling in ToolManager.execute_func"""
    print("=== Testing ToolManager.execute_func Error Handling ===")
    
    # Create a function that raises an error
    def error_function(x: int):
        """A function that always raises an error"""
        raise ValueError(f"Test error with input: {x}")
    
    # Create function tool
    error_tool = FunctionTool(error_function)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[error_tool])
    
    # Test error handling
    print("Testing function that raises error:")
    error_func = Function(name="error_function", args=[42], kwargs={})
    
    # FunctionTool catches errors and wraps them in FunctionOutput.error
    result = tool_manager.execute_func(error_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.error is not None
    assert "Test error with input: 42" in str(result.error)
    print(f"  Error properly captured: {result.error}")
    
    print("ToolManager.execute_func error handling tests passed!")


@pytest.mark.asyncio
async def test_tool_manager_execute_func_async_error_handling():
    """Test error handling in ToolManager.execute_func_async"""
    print("=== Testing ToolManager.execute_func_async Error Handling ===")
    
    # Create sync function that raises error
    def sync_error_function(x: int):
        """A sync function that raises an error"""
        raise RuntimeError(f"Sync error with input: {x}")
    
    # Create async function that raises error
    async def async_error_function(x: int):
        """An async function that raises an error"""
        await asyncio.sleep(0.01)
        raise RuntimeError(f"Async error with input: {x}")
    
    # Create function tools
    sync_error_tool = FunctionTool(sync_error_function)
    async_error_tool = FunctionTool(async_error_function)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[sync_error_tool, async_error_tool])
    
    # Test sync function error in async execution - FunctionTool.acall catches errors
    print("Testing sync function error in async mode:")
    sync_error_func = Function(name="sync_error_function", args=[42], kwargs={})
    
    result = await tool_manager.execute_func_async(sync_error_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.error is not None
    assert "Sync error with input: 42" in str(result.error)
    print(f"  Error properly captured: {result.error}")
    
    # Test async function error in async execution - FunctionTool.acall catches errors
    print("\nTesting async function error in async mode:")
    async_error_func = Function(name="async_error_function", args=[99], kwargs={})
    
    result = await tool_manager.execute_func_async(async_error_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    assert result.error is not None
    assert "Async error with input: 99" in str(result.error)
    print(f"  Error properly captured: {result.error}")
    
    print("ToolManager.execute_func_async error handling tests passed!")


def test_tool_manager_execute_func_with_generators():
    """Test ToolManager.execute_func with generator functions"""
    print("=== Testing ToolManager.execute_func with Generators ===")
    
    # Create sync generator function
    def sync_generator_func(n: int):
        """Generate numbers from 0 to n-1"""
        for i in range(n):
            yield f"item_{i}"
    
    # Create async generator function  
    async def async_generator_func(n: int):
        """Async generate numbers from 0 to n-1"""
        for i in range(n):
            await asyncio.sleep(0.001)  # Small async delay
            yield f"async_item_{i}"
    
    # Create function tools
    sync_gen_tool = FunctionTool(sync_generator_func)
    async_gen_tool = FunctionTool(async_generator_func)
    
    # Create tool manager
    tool_manager = ToolManager(tools=[sync_gen_tool, async_gen_tool])
    
    # Test sync generator
    print("Testing sync generator function:")
    sync_gen_func = Function(name="sync_generator_func", args=[3], kwargs={})
    result = tool_manager.execute_func(sync_gen_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    
    # Test the generator output
    generator = result.output
    items = list(generator)
    print(f"  Generated items: {items}")
    assert items == ["item_0", "item_1", "item_2"]
    
    # Test async generator (should be handled in sync mode)
    print("\nTesting async generator function in sync mode:")
    async_gen_func = Function(name="async_generator_func", args=[2], kwargs={})
    result = tool_manager.execute_func(async_gen_func)
    print(f"  Result: {result}")
    assert isinstance(result, FunctionOutput)
    
    # In sync mode, async generators should be converted to lists
    if isinstance(result.output, list):
        print(f"  Async generator converted to list: {result.output}")
        assert result.output == ["async_item_0", "async_item_1"]
    
    print("ToolManager.execute_func generator tests passed!")


# =============== END OF TOOL MANAGER EXECUTE TESTS ===============


from adalflow.optim.grad_component import GradComponent


class GradAdd(GradComponent):
    def __init__(self):
        super().__init__(desc="A simple addition tool")
        print(f"training: {self.training}")

    def call(self, x, y):
        return x + y

    # def forward(self, x, y):
    #     print(f"training: {self.training}")
    #     return f"{x + y} + forward"


class GradSub(GradComponent):
    def __init__(self):
        super().__init__(desc="A simple subtraction tool")

    def call(self, x, y):
        return x - y

    # def forward(self, x, y):
    #     print(f"training: {self.training}")
    #     return f"{x - y} + forward"


class TestComponent(Component):
    def __init__(self):
        super().__init__()

        self.add = GradAdd()
        self.sub = GradSub()

        print(f"sub_component: {self.sub.training}")

        print(f"add_component: {self.add.training}")

        def add_as_tool(x, y):
            return self.add(x, y)

        # two ways to call a gradcomponent function

        self.tools_list = [
            FunctionTool(fn=add_as_tool),
            FunctionTool(fn=self.sub.__call__),
        ]
        # components can only be managed by component so that we can recursively set the training mode
        self.tools = ComponentList(self.tools_list)


add = GradAdd()
sub = GradSub()


class TestComponnetInstanceOutsideComponent(Component):
    def __init__(self):
        super().__init__()

        print(f"sub_component: {sub.training}")

        print(f"add_component: {add.training}")

        def add_as_tool(x, y):
            return add(x, y)

        self.tools = [
            FunctionTool(fn=add_as_tool),
            FunctionTool(fn=sub.__call__),
        ]


class TestToolManagerComponent(Component):

    def __init__(self):
        super().__init__()

        print(f"sub_component: {sub.training}")

        print(f"add_component: {add.training}")

        def add_as_tool(x, y):
            return add(x, y)

        # two ways to call a gradcomponent function

        self.tools = [
            FunctionTool(fn=add_as_tool),
            FunctionTool(fn=sub.__call__),
        ]
        print(f"tools: {self.tools}")
        self.tools = ComponentList(self.tools)

        # manag by tool manager, and since the component is passed to tools_manager which is also a component, it will be in training mode
        self.tools_manager = ToolManager(tools=self.tools)


def test_function_tool_with_grad_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert test_com.add.training
    # ensure it is the forward method that is called
    print(f"training: {test_com.add.training}")
    print(f"training: {test_com.sub.training}")
    print(f"function tool training: {test_com.tools[0].training}")
    assert test_com.tools[0].training
    assert test_com.tools_list[0].training
    output = test_com.tools[0](1, 2)
    print(f"output: {output}")
    assert output.data.output == 3


def test_component_instance_outside_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestComponnetInstanceOutsideComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert not add.training  # the subcomponent is no longer in training mode
    # ensure it is the forward method that is called
    output = test_com.tools[0](1, 2)
    assert output.output == 3


def test_tool_manager_with_grad_component():
    r"""When we set the training mode of the component, the subcomponents will change with it.
    Once the subcomponent change, it will adapt to training model too.
    """

    test_com = TestToolManagerComponent()
    assert not test_com.training
    # call the tools
    output = test_com.tools_manager.tools[0](1, 2)
    # ensure it is the call method that is called
    assert output.output == 3
    test_com.train()
    assert test_com.training
    assert test_com.tools.training
    assert test_com.tools_manager.training
    assert sub.training
  
    # ensure it is the forward method that is called
    output = test_com.tools_manager.tools[1](1, 2)
    assert output.data.output == -1
