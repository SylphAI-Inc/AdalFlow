.. _parameter:

Parameter
====================


There are two types of parameters:

* once-off, such as loss, y_pred, intermedia response where each run it will create a new one, and they are temporary and will be released after the run.
* persistent, the parameters that we are optimizing, such as those with an actual type assigned `param_type` in the `Parameter` class.


TODO: a DAG to show this.


All our targing parameter to train will end up being a leaf node in the auto-diff DAG.

In each run, the persistent parameters can be used by multiple successors if batch_size > 1. This results it to accumulate all traces.

auto-diff
-----------
To support auto-diff, we added `predecessors` , and adalflow created `peers` concept to ensure training parameters are aware of each other to not conflicting while optimizing.

For instance, if no peers context, the system instruction can generate examples or enforce output format while users want to train each of them separately.

These characteristics are important. this means for passing the score


teacher
-----------

Teacher generator should not use any parameter as it will only be used in call.


Generator Prediction parameters
--------------------------------

We will track:

.. code-block:: python

    input_args {"Prompt_kwargs", "model_kwargs", ...}
    raw_response
    data (processed response)

Potentially we can make the data generator output

Or any component output parameter, it should have a forward

.. code-block:: python

    response = Parameter(
            data=retriever_reponse,
            alias=self.name + "_output",
            role_desc="Retriever response",
            predecessors=predecessors,
            input_args=input_args,
        )


Demo Parameter
----------------

Demo parameter and demo optimizers does not require loss backpropagatin.

It needs a loss function to compute the eval score. but we dont need to run
`loss.backward()` if we are only doing few-shot optimization.

loss.backward will do two things:
* backpropagate score to prececessors as a context
* backpropagate the gradients to text prompts.
