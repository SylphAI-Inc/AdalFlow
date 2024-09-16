.. _parameter:

Parameter
====================

.. figure:: /_static/images/tensor_parameter.png
    :align: center
    :alt: AdalFlow Tensor and Parameter
    :width: 620px


    AdalFlow Tensor and Parameter

Designing an auto-diff system for LLM task pipeline is actually quite challenging.
First we made effort to make everything a component and it made the interactions between each components easier and more transparent at visualization.
But every component takes in any type of data as input, just as in pytorch, input to compute in the auto-diff needs to be a tensor, and trinable parameters will be Parametr (a type of mutable tensor),
this can be applied to the LLM task pipeline as well.
LLM task pipeline and in-context learning will add new trainable parameters such as few-shot demos and instruction tuning.

There are two types of parameters:

* once-off, such as loss, y_pred, intermediate response where each run it will create a new one, and they are temporary and will be released after the run.
* persistent, the parameters that we are optimizing, such as those with an actual type assigned `param_type` in the `Parameter` class.
   For the persistent parameters, the data type will be string.


Each parameter has a:
id: unique identifier
name:

Intermediate parameters
------------------------
intermediate parameters data = Componnet.call output.

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
