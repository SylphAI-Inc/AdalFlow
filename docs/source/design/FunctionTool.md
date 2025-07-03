# Designing of AdalFlow FunctionTool

`FunctionTool` is a core component in AdalFlow that provides a standardized interface for wrapping functions as tools that can be used by LLMs and agents. It extends `Component` and supports both synchronous and asynchronous functions, generators, and trainable components.

## Overview

The `FunctionTool` class wraps any callable function and provides a standardized interface for:
- Function metadata generation (name, description, parameters schema)
- Execution with proper error handling
- Support for both sync and async functions
- Generator function support
- Integration with trainable components

## Key Features

- **Universal Function Support**: Handles regular functions, async functions, generators, and async generators
- **Automatic Type Detection**: Automatically detects function type using `FunctionType` enum
- **Dual Execution Modes**: Supports both synchronous (`call`) and asynchronous (`acall`) execution
- **Error Handling**: Captures and wraps errors in `FunctionOutput` objects
- **Training Integration**: Seamlessly integrates with trainable components for optimization
- **Metadata Generation**: Automatically creates function definitions with parameter schemas

## Function Types

The `FunctionType` enum defines four supported function types:

```python
class FunctionType(Enum):
    SYNC = auto()              # Regular sync function: def func(): return value
    ASYNC = auto()             # Async function: async def func(): return value
    SYNC_GENERATOR = auto()    # Sync generator: def func(): yield value
    ASYNC_GENERATOR = auto()   # Async generator: async def func(): yield value
```

## Basic Usage

### Simple Function Wrapping

```python
from adalflow.core.func_tool import FunctionTool

def add_numbers(x: int, y: int) -> int:
    """Add two numbers together"""
    return x + y

# Create a function tool
tool = FunctionTool(add_numbers)

# Execute the function
result = tool.call(3, 5)
print(result.output)  # 8

# Test Output:
# Output: 8
# Full result: FunctionOutput(name='add_numbers', input=Function(thought=None, name='add_numbers', args=(3, 5), kwargs={}), parsed_input=None, output=8, error=None)
```

### Async Function Support

```python
import asyncio

async def async_add(x: int, y: int) -> int:
    """Asynchronously add two numbers"""
    return x + y

# Create async function tool
async_tool = FunctionTool(async_add)

# Execute synchronously (blocks until complete)
result = async_tool.call(3, 5)
print(result.output)  # 8

# Execute asynchronously
async def main():
    result = await async_tool.acall(3, 5)
    print(result.output)  # 8

asyncio.run(main())

# Test Output:
# Sync call output: 8
# Async call output: 8
```

### Generator Functions

```python
def number_generator(count: int):
    """Generate a sequence of numbers"""
    for i in range(count):
        yield i

# Create generator tool
gen_tool = FunctionTool(number_generator)

# Execute and get generator
result = gen_tool.call(3)
numbers = list(result.output)
print(numbers)  # [0, 1, 2]

# Test Output:
# Numbers: [0, 1, 2]
# Full result: FunctionOutput(name='number_generator', input=Function(thought=None, name='number_generator', args=(3,), kwargs={}), parsed_input=None, output=<generator object number_generator at 0x111287d30>, error=None)
```

## Advanced Usage

### Class Methods as Tools

```python
class Calculator:
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    def multiply(self, x: float) -> float:
        """Multiply input by the multiplier"""
        return x * self.multiplier

calc = Calculator(2.5)
multiply_tool = FunctionTool(calc.multiply)

result = multiply_tool.call(4)
print(result.output)  # 10.0

# Test Output:
# Output: 10.0
# Full result: FunctionOutput(name='Calculator_multiply', input=Function(thought=None, name='Calculator_multiply', args=(4,), kwargs={}), parsed_input=None, output=10.0, error=None)
```

### Integration with Trainable Components

```python
from adalflow.core.func_tool import FunctionTool
from adalflow.core.component import Component

class AgenticRAG(Component):
    def __init__(self):
        super().__init__()
        self.retriever = Retriever()  # Your retriever implementation
        self.llm = Generator()        # Your generator implementation

        def retriever_as_tool(query: str) -> str:
            """Retrieve relevant documents"""
            return self.retriever(query)

        # Create tools with component references for training
        self.tools = [
            FunctionTool(retriever_as_tool, component=self.retriever),
            FunctionTool(self.llm.__call__, component=self.llm)
        ]
```

## API Reference

### Constructor

```python
FunctionTool(
    fn: Union[Callable, FunGradComponent],
    definition: Optional[FunctionDefinition] = None
)
```

**Parameters:**
- `fn`: The function to wrap (can be sync, async, generator, or trainable component)
- `definition`: Optional custom function definition (auto-generated if not provided)

### Methods

#### `call(*args, **kwargs) -> FunctionOutput`

Execute the function synchronously. Works with all function types:
- **SYNC**: Calls directly
- **ASYNC**: Runs in event loop (blocks until complete)
- **SYNC_GENERATOR**: Returns generator object
- **ASYNC_GENERATOR**: Collects all values into a list

#### `acall(*args, **kwargs) -> FunctionOutput`

Execute the function asynchronously:
- **SYNC**: Executes in thread pool
- **ASYNC**: Awaits the coroutine
- **SYNC_GENERATOR**: Returns generator object
- **ASYNC_GENERATOR**: Returns async generator object

#### `detect_function_type(fn: Callable) -> FunctionType`

Static method to detect the type of a given function.

### Properties

- `fn`: The wrapped function
- `definition`: The function definition with metadata
- `function_type`: The detected function type
- `class_instance`: The class instance if the function is a bound method

## Function Output

All executions return a `FunctionOutput` object containing:

```python
@dataclass
class FunctionOutput:
    name: str                    # Function name
    input: Function             # Input arguments (name, args, kwargs)
    output: Any                 # Function result
    error: Optional[str]        # Error message if execution failed
```

## Error Handling

`FunctionTool` provides robust error handling:

```python
def error_function():
    raise ValueError("Something went wrong")

tool = FunctionTool(error_function)
result = tool.call()

print(result.error)  # "Error at calling error_function: Something went wrong"
print(result.output)  # None

# Test Output:
# Error: Error at calling <function error_function at 0x102fe0180>: Something went wrong
# Output: None
# Full result: FunctionOutput(name='error_function', input=Function(thought=None, name='error_function', args=(), kwargs={}), parsed_input=None, output=None, error='Error at calling <function error_function at 0x102fe0180>: Something went wrong')
```

## Training Mode

When used with trainable components, `FunctionTool` supports training mode:

```python
# In evaluation mode
tool.eval()
result = tool.call(args)
assert isinstance(result, FunctionOutput)

# In training mode
tool.train()
result = tool.call(args)
assert isinstance(result, Parameter)  # Wrapped for gradient computation
```

## Best Practices

1. **Function Documentation**: Provide clear docstrings as they're used for function descriptions
2. **Type Hints**: Use type hints for better parameter schema generation
3. **Error Handling**: Let `FunctionTool` handle errors rather than catching them in your function
4. **Async Usage**: Use `acall()` for async functions when possible for better performance
5. **Generator Consumption**: Remember that generators returned by `call()` need to be consumed

## Examples

### Complete Example with ToolManager

```python
from adalflow.core.func_tool import FunctionTool
from adalflow.core.tool_manager import ToolManager

def add(x: int, y: int) -> int:
    """Add two numbers"""
    return x + y

def multiply(x: int, y: int) -> int:
    """Multiply two numbers"""
    return x * y

# Create tools
add_tool = FunctionTool(add)
multiply_tool = FunctionTool(multiply)

# Create tool manager
manager = ToolManager(tools=[add_tool, multiply_tool])

# Execute functions
add_result = manager.execute_func(Function(name="add", args=[2, 3], kwargs={}))
multiply_result = manager.execute_func(Function(name="multiply", args=[4, 5], kwargs={}))

print(f"Add result: {add_result.output}")      # 5
print(f"Multiply result: {multiply_result.output}")  # 20

# Test Output:
# Add result: 5
# Multiply result: 20
# Full add result: FunctionOutput(name='add', input=Function(thought=None, name='add', args=(2, 3), kwargs={}), parsed_input=None, output=5, error=None)
# Full multiply result: FunctionOutput(name='multiply', input=Function(thought=None, name='multiply', args=(4, 5), kwargs={}), parsed_input=None, output=20, error=None)
```

## Related Components

- [`ToolManager`](./ToolManager.md): Manages collections of function tools
- [`Component`](./Component.md): Base class for trainable components
- [`FunctionDefinition`](./types.md#functiondefinition): Function metadata structure
- [`FunctionOutput`](./types.md#functionoutput): Function execution result structure
