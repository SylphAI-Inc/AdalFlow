from adalflow.core.func_tool import FunctionTool
import inspect
import asyncio

def sync_fun():
    """I'm a sync function"""
    # isfunction, #iscallable 
    return "sync"

async def async_fun():
    """I'm an async function"""
    # iscoroutinefunction #isfunction #iscallable
    return "async"

def sync_yield_fun():
    """I'm a sync generator function"""
    # isgeneratorfunction #isfunction #iscallable
    yield "sync_yield"

async def async_yield_fun():
    """I'm an async generator function"""
    # isasyncgenfunction #isfunction #iscallable
    yield "async_yield"


def inspect_function_types():
    """Demonstrate how inspect can detect different function types"""
    print("=== Function Type Detection with inspect ===\n")
    
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
        print()


def test_function_tools():
    """Test FunctionTool with different function types"""
    print("=== FunctionTool Testing ===\n")
    
    # Create function tools
    sync_tool = FunctionTool(sync_fun)
    async_tool = FunctionTool(async_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    for tool in [sync_tool, async_tool, sync_yield_tool, async_yield_tool]:
        print(tool)


async def test_async_execution():
    """Test async execution of function tools"""
    print("=== Async Execution Testing ===\n")

    sync_tool = FunctionTool(sync_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    
    async_tool = FunctionTool(async_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    # Test all function types with acall
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
            print(f"  Result: {result}")
            
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
                    
                    elif tool.function_type.name == 'ASYNC_GENERATOR':
                        # Async generator - iterate with async for
                        async_generator = result.output
                        i = 0
                        async for item in async_generator:
                            print(f"    Yield {i}: {item}")
                            i += 1
                
                # For regular functions, check the value
                else:
                    print(f"  Function returned: {result.output}")
            

                
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Error type: {type(e).__name__}")
        
        print("-" * 50)

    # Test concurrent execution
    print("\n=== Testing Concurrent Execution ===\n")
    
    try:
        # Run all tools concurrently
        results = await asyncio.gather(
            sync_tool.acall(),
            async_tool.acall(),
            sync_yield_tool.acall(),
            async_yield_tool.acall()
        )
        
        print("Concurrent execution results:")
        for i, (name, _) in enumerate(tools_to_test):
            result = results[i]
            print(f"  {name}: {result.output}")
            
    except Exception as e:
        print(f"Concurrent execution error: {e}")

    # Test generator consumption patterns
    print("\n=== Testing Generator Consumption Patterns ===\n")
    
    # Test sync generator
    print("Testing sync generator consumption:")
    sync_result = await sync_yield_tool.acall()
    sync_gen = sync_result.output
    print(f"  Generator type: {type(sync_gen).__name__}")
    
    # Consume the generator
    items = list(sync_gen)
    print(f"  Consumed items: {items}")
    
    # Test async generator
    print("\nTesting async generator consumption:")
    async_result = await async_yield_tool.acall()
    async_gen = async_result.output
    print(f"  Generator type: {type(async_gen).__name__}")
    
    # Consume the async generator
    items = []
    async for item in async_gen:
        items.append(item)
    print(f"  Consumed items: {items}")

    print("\n=== Async Execution Testing Complete ===\n")


def test_sync_execution():
    """Test sync execution of function tools"""
    print("=== Sync Execution Testing ===\n")
    
    sync_tool = FunctionTool(sync_fun)
    sync_yield_tool = FunctionTool(sync_yield_fun)
    
    async_tool = FunctionTool(async_fun)
    async_yield_tool = FunctionTool(async_yield_fun)
    
    # Test all function types with call
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
            # Call the function using call (sync method)
            result = tool.call()
            
            # Check output type
            print(f"  Result type: {type(result).__name__}")
            print(f"  Result: {result}")
            
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
                    
                    elif tool.function_type.name == 'ASYNC_GENERATOR':
                        # Async generator - should be collected into a list by call()
                        items = result.output
                        print(f"    Collected items: {items}")
                        for i, item in enumerate(items):
                            print(f"    Item {i}: {item}")
                
                # For regular functions, check the value
                else:
                    print(f"  Function returned: {result.output}")
            
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
        
        print("-" * 50)

    # Test sequential execution timing
    print("\n=== Testing Sequential Execution Timing ===\n")
    
    import time
    
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

    # Test generator consumption patterns in sync mode
    print("\n=== Testing Generator Consumption Patterns (Sync) ===\n")
    
    # Test sync generator
    print("Testing sync generator consumption:")
    sync_result = sync_yield_tool.call()
    sync_gen = sync_result.output
    print(f"  Generator type: {type(sync_gen).__name__}")
    
    # Consume the generator
    items = list(sync_gen)
    print(f"  Consumed items: {items}")
    
    # Test async generator (should be collected into list)
    print("\nTesting async generator consumption (sync mode):")
    async_result = async_yield_tool.call()
    async_items = async_result.output
    print(f"  Collected items type: {type(async_items).__name__}")
    print(f"  Collected items: {async_items}")

    # Test error handling
    print("\n=== Testing Error Handling ===\n")
    
    def error_sync_fun():
        """A sync function that raises an error"""
        raise ValueError("Sync function error")
    
    async def error_async_fun():
        """An async function that raises an error"""
        await asyncio.sleep(0.1)
        raise ValueError("Async function error")
    
    error_sync_tool = FunctionTool(error_sync_fun)
    error_async_tool = FunctionTool(error_async_fun)
    
    # Test sync error
    print("Testing sync function with error:")
    try:
        result = error_sync_tool.call()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Caught error: {e}")
    
    # Test async error in sync mode
    print("\nTesting async function with error (sync mode):")
    try:
        result = error_async_tool.call()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Caught error: {e}")

    print("\n=== Sync Execution Testing Complete ===\n")


if __name__ == "__main__":
    # Show how inspect detects function types
    inspect_function_types()
    
    # Show FunctionTool properties
    test_function_tools()
    
    # Test sync execution
    test_sync_execution()
    
    # # Test async execution
    # print("\n" + "="*50)
    # asyncio.run(test_async_execution())
    