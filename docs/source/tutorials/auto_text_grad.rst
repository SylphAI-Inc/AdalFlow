Auto Text-Diff
===============================================
Show a DAG with parameter nodes and edges.

To make a task pipeline trainable.

Textual Gradient Operators
--------------------------
"Textual gradient Operators" are the operators that are capable of backpropagation, this including operator for LLM calls, for evaluate function, and for llm as a judge function.
Think of the LLM calls as model layer in pytorch, such as nn.Linear, nn.Conv2d, or transformer layers.
Think of the evaluation function (normally you have gt) and LLM as judge (normall you have no gt reference but you rely on llm to give an evaluation score) as
a loss function in pytorch, such as nn.CrossEntropyLoss, nn.MSELoss, or nn.BCELoss.
These operators need to be capable of backpropagation to get "feedback"/"gradients" for the auto-diff optimizer.
We introduce ``GradComponent`` class which consists of two must-have abstract methods: ``forward`` and ``backward``.

- ``forward``: The forward pass of the operator. It will return a `Prameter` with the backward function set to the backward function of the operator.
- ``backward``: The backward pass of the operator. It will compute the response's predecessor's gradient with regard to the response. (The ``Parameter`` object returned by the ``forward`` method)

We currently have the following operators:
- ``Generator`` is adapted as a ``GradComponent``.

Generator Adaptation
~~~~~~~~~~~~~~~~~~~~~~

In auto-text grad, generator needs to be adapted as an operator that supports backpropagation to get "feedback"/"gradients" for the auto-diff optimizer.
So, it inherits from ``GradComponent`` class, adding ``forward``, ``backward`` and ``set_backward_engine`` methods.

Note:

 (1) When in forward mode, we need to parse the ``GeneratorOutput`` to ``Parameter`` object. Often we can use ``data`` to be ``data`` of the ``Parameter``.
 But if the generator is structured output, we might need to do a data map.

 (2) The generator can fail, and an optimizer should capture this failure message as part of the direct feedback. We have `failure_message_to_backward`.
 Here is one failure example: `data=Error: None, Raw response: Sure, I'm ready to help. What's the reasoning question?`.


Retriever Adaptation
~~~~~~~~~~~~~~~~~~~~~~
For now, we dont set up persistent parameters for retriever, the role of the retriever is to relay  any intermediate parameters back to its predecessors if they happen to be a generator.
The backward function for now has no effect, but it is a placeholder for future implementation.

For demo optimizer, it does not need the whole pipeline to be propogatable, which means it can be a
DAG of parameters. And the later is the condition to do text-grad for any generator in a task pipeline.
..
    TODO: if we set the top_k as a parameter (hyperparameter along with the data type int)
    text_grad can be used to optimize the hyperparametr to replace the human intelligence.
    will it work better than hyperparameter sweep? This is a future research project.

To optimize any task pipeline
------------------------------

For generators: prompt_kwargs are the leaf nodes to optimize.
It takes [str, Parameter] as value.

GradComponent handles the predecessors which form a DAG of parameters.
So all arguments in the input_args if they are of type parameters, they are all predecessors.

A user subclass GradComponent will automatically make the component trainable (at least for the default behaviors).
Just like in pytorch, if you subclass nn.Module, you can use the model to train.


Question: there might no need to have the concept of Component, so we have simplier library apis and one less abstract layer.


EvalFunction As Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

**Gradient engine template**


Here is one example of d_(1) / d_g_output.

```
The response from the generator was accurate according to the ObjectCountingEvalFn.
The output correctly matched the ground truth, resulting in a perfect score of 1.0.
There is no need for improvement as the generator's output was correct.


Textual Gradient Optimizer
----------------------------



AdalComponent to organize code
------------------------------


Trainer to put all together
----------------------------

..
    TODO:
    1. clearity on self.tracing
