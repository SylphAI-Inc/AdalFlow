"""
Tutorial: Using Runner.forward() Method to Get Trainable Results

This tutorial demonstrates how to use the Runner's forward() method to get 
trainable outputs that can be used for optimization. The key difference between
runner.call() and runner.forward() is that forward() returns trainable Parameter 
objects that maintain gradient information for optimization.
"""

import adalflow as adal
from adalflow.core.types import RunnerResult
from adalflow.optim.parameter import OutputParameter
from adalflow.utils import setup_env

setup_env()


def create_simple_agent():
    """Create a simple agent for demonstration."""
    # Configure model client
    model_config = {
        "model_client": adal.AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
        }
    }
    
    # Create agent with simple tools
    agent = adal.Agent(
        name="demo_agent",
        add_llm_as_fallback=True,
        max_steps=3,
        **model_config
    )
    
    return agent


def example_1_basic_forward_usage():
    """Example 1: Basic usage of runner.forward() method."""
    print("=== Example 1: Basic Forward Usage ===")
    
    agent = create_simple_agent()
    runner = adal.Runner(agent=agent, training=True)  # Important: set training=True
    
    # Use forward() instead of call() to get trainable results
    prompt_kwargs = {
        "input_str": "What is 2+2? Use the finish action to provide your final answer."
    }
    
    # This returns an OutputParameter with gradient information
    trainable_result = runner.forward(prompt_kwargs=prompt_kwargs)
    
    print(f"Result type: {type(trainable_result)}")
    print(f"Is trainable (requires_opt): {trainable_result.requires_opt}")
    print(f"Has gradient function: {trainable_result.grad_fn is not None}")
    
    # Access the actual RunnerResult from the parameter
    runner_result: RunnerResult = trainable_result.data
    print(f"Final answer: {runner_result.answer}")
    print(f"Number of steps: {len(runner_result.step_history)}")
    
    return trainable_result


def example_2_extracting_final_output():
    """Example 2: How to extract the final output from trainable result."""
    print("\n=== Example 2: Extracting Final Output ===")
    
    agent = create_simple_agent()
    runner = adal.Runner(agent=agent, training=True)
    
    prompt_kwargs = {
        "input_str": "Calculate 15 * 3 and explain your reasoning. Use finish to provide your final answer."
    }
    
    # Get trainable result
    trainable_result = runner.forward(prompt_kwargs=prompt_kwargs)
    
    # Method 1: Access through .data attribute
    final_answer = trainable_result.data.answer
    print(f"Method 1 - Final answer via .data: {final_answer}")
    
    # Method 2: Access step history for detailed breakdown
    step_history = trainable_result.data.step_history
    print(f"Method 2 - Step history length: {len(step_history)}")
    for i, step in enumerate(step_history):
        print(f"  Step {i}: {step.function.name if step.function else 'None'} -> {step.observation}")
    
    # Method 3: Check if there were any errors
    if trainable_result.data.error:
        print(f"Method 3 - Error occurred: {trainable_result.data.error}")
    else:
        print("Method 3 - No errors occurred")
    
    return trainable_result


def example_3_comparison_call_vs_forward():
    """Example 3: Comparing runner.call() vs runner.forward()."""
    print("\n=== Example 3: Call vs Forward Comparison ===")
    
    agent = create_simple_agent()
    runner = adal.Runner(agent=agent)
    
    prompt_kwargs = {
        "input_str": "What is the capital of France? Use finish to provide your final answer."
    }
    
    # Using call() - returns RunnerResult directly (not trainable)
    runner.training = False  # Inference mode
    call_result = runner(prompt_kwargs=prompt_kwargs)
    
    print("Call Result:")
    print(f"  Type: {type(call_result)}")
    print(f"  Answer: {call_result.answer}")
    print(f"  Is trainable: No (RunnerResult object)")
    
    # Using forward() - returns trainable OutputParameter
    runner.training = True  # Training mode
    forward_result = runner.forward(prompt_kwargs=prompt_kwargs)
    
    print("\nForward Result:")
    print(f"  Type: {type(forward_result)}")
    print(f"  Answer: {forward_result.data.answer}")
    print(f"  Is trainable: {forward_result.requires_opt}")
    print(f"  Has predecessors: {len(forward_result.predecessors) > 0}")
    
    return call_result, forward_result

def example_4_gradient_tracking():
    """Example 5: Understanding gradient tracking in trainable results."""
    print("\n=== Example 4: Gradient Tracking ===")
    
    agent = create_simple_agent()
    runner = adal.Runner(agent=agent, training=True)
    
    prompt_kwargs = {
        "input_str": "Solve: If John has 5 apples and gives away 2, how many does he have left?"
    }
    
    # Get trainable result
    trainable_result = runner.forward(prompt_kwargs=prompt_kwargs)
    
    print("Gradient Information:")
    print(f"  Has grad_fn: {trainable_result.grad_fn is not None}")
    print(f"  Requires optimization: {trainable_result.requires_opt}")
    print(f"  Parameter name: {trainable_result.name}")
    print(f"  Number of predecessors: {len(trainable_result.predecessors)}")
    
    # The grad_fn contains backward context for optimization
    if trainable_result.grad_fn:
        backward_ctx = trainable_result.grad_fn
        print(backward_ctx)
        print(f"  Backward function available: {backward_ctx.backward_fn is not None}")
        print(f"  Template available: {backward_ctx.kwargs['template'] is not None}")
        print(f"  Prompt string length: {len(backward_ctx.kwargs['prompt_str']) if backward_ctx.kwargs['prompt_str'] else 0}")
    
    return trainable_result


def main():
    """Run all examples."""
    print("Tutorial: Runner.forward() for Trainable Results")
    print("=" * 50)
    
    try:
        # Run examples
        example_1_basic_forward_usage()
        example_2_extracting_final_output()
        example_3_comparison_call_vs_forward()
        example_4_gradient_tracking()
        
        print("\n" + "=" * 50)
        print("Tutorial completed successfully!")
        print("\nKey takeaways:")
        print("1. Use runner.forward() instead of runner.call() for trainable results")
        print("2. Set training=True when creating Runner for optimization")
        print("3. Forward() returns OutputParameter with gradient information")
        print("4. Trainable results can be used with AdalFlow's optimization system")
        
    except Exception as e:
        print(f"Error occurred during tutorial: {e}")
        print("Make sure you have:")
        print("1. Set up your API key (ANTHROPIC_API_KEY)")
        print("2. Installed AdalFlow with all dependencies")
        return False
    
    return True


if __name__ == "__main__":
    # Set up environment (you may need to configure your API key)
    from adalflow.utils import setup_env
    setup_env()
    
    main()