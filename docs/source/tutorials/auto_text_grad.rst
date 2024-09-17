Auto Text-Grad
===============================================
Show a DAG with parameter nodes and edges.

To make a task pipeline trainable.

Auto text grad system is similr to pytorch autograd. Here are how we also differs:

1. Torch.Tensor & Torch.Parameter vs AdalFlow.Parameter : AdalFlow.Tensor can save any type of data and Tensor is mainly numerical array and matrices.
This means that the backward is not the gradient function from the math operations applied on the tensor, but customized towards the operators.
The operators here are components like Generator, Retriever, Loss function, etc.
We have defined the backward function for the generator which genreates the textual feedback for Parameter of prompt type.
For Retriever, right now, it does not have its parameter types that we optimize but it can very much change and be improved in the future.

In adalflow, we use the parameter types to differentiate instead of separately create a Tensor and its subclass Parameter.
We have the follow parameter type:

- trainable parameters to generator
   prompt
   demos

- intermediate parameters
  - input to the component
  - output from the component

- gradient

To be able to pass parameters around to the whole pipeline.



Torch.no_grad() vs AdalFlow.GradComponent.

Torch.no_grad() is a context manager that disables gradient calculation.
(1) It stops tracking the operations that are performed to build the computation-graph. In AdalFlow, we use Adal.Component call or the subclass adal.GradComponent
(2) Save and handles intermediate values(eg. activations, inputs) needed for the backward pass.
(3) Stores the computation graph for later backpropagation.

In pytorch you do this for inference:

.. code-block:: python

    import torch

    model = MyModel()
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking
        output = model(input_data)  # Forward pass only

In AdalFlow, you do this for inference:

.. code-block:: python

    import adalflow as adal

    task_pipeline = MyTaskPipeline()
    task_pipeline.eval()  # Set model to evaluation mode
    task_pipeline(input_data)  # similar to torch.no_grad() or
    # task_pipeline.call(input_data)  # Forward pass only
    # task_pipeline.acall(input_data)  # Forward pass only

Just like pytorch has tensor and parameter, which are a special type of tensor, the gradcomponent is a special type of component capable of auto-text-grad.

**How to connect the output-input between components?**

In pytorch, this is a earsier problem as they are all matrices within tensors.
But in LLM applications, (1) each component's output can be very different in terms of form.
For generator we have ``GeneratorOutput`` and for retriever, we have ``List[RetrieverOutput]``
. To connect retriever output to generator output, we need special handling of the `("\n\n".join(ro.[0].documents)` .
For langgraph, this is done inside of each manually defined node. And the whole pipeline uses GraphState (global accessible) to the whole graph to access and to store the data.
(2) we need robust error handling in our output structure too.


.. code-block:: python

    class GraphState(BaseModel):

        question: Optional[str] = None
        generation: Optional[str] = None
        documents: List[str] = []

    def retriever_node(state: GraphState):
        new_documents = retriever.invoke(state.question)
        new_documents = [d.page_content for d in new_documents]
        state.documents.extend(new_documents)
        return {"documents": state.documents}

    def generation_node(state: GraphState):
        generation = rag_chain.invoke({
            "context": "\n\n".join(state.documents),
            "question": state.question,
        })
        return {"generation": generation}

When we are doing training, both outputs are parameters, but the way to connect data is the same.
We use a successor_map_fn of type `Dict[str, Callable]` to connect the output of one component to the input of another component.
str will be `id(successor)`. This is only needed in the forward function of any Component or GradComponent.

Here is our example:

.. code-block:: python

    def foward(self, question: str, id: str = None) -> adal.Parameter:
        retriever_out = self.retriever.forward(input=question)
        successor_map_fn = lambda x: (
            "\n\n".join(x.data[0].documents)
            if x.data and x.data[0] and x.data[0].documents
            else ""
        )
        retriever_out.add_successor_map_fn(successor=self.llm, map_fn=successor_map_fn)
        generator_out = self.llm.forward(
            prompt_kwargs={"question": question, "context": retriever_out}, id=id
        )
        return generator_out

#TODO: save the trace_graph
And here is our trace_graph:

Textual Gradient Operators
--------------------------
"Textual gradient Operators" are the operators that are capable of backpropagation, this including operator for LLM calls, for evaluate function, and for llm as a judge function.
Think of the LLM calls as model layer in pytorch, such as nn.Linear, nn.Conv2d, or transformer layers.
Think of the evaluation function (normally you have gt) and LLM as judge (normall you have no gt reference but you rely on llm to give an evaluation score) as
a loss function in pytorch, such as nn.CrossEntropyLoss, nn.MSELoss, or nn.BCELoss.


These operators need to be capable of backpropagation to get "feedback"/"gradients" for the auto-diff optimizer.
We introduce ``GradComponent`` class which consists of two must-have abstract methods: ``forward`` and ``backward``.
``GradComponent`` has default ``forward`` that wraps a normal function call inside of the ``forward`` method to return a Parameter and builds the computation graph.

- ``forward``: The forward pass of the operator. It will return a `Prameter` with the backward function set to the backward function of the operator.
- ``backward``: The backward pass of the operator. It will compute the response's predecessor's gradient with regard to the response. (The ``Parameter`` object returned by the ``forward`` method)

We currently have the following operators:
- ``Generator`` is adapted as a ``GradComponent``.
.. TODO:
  - remove the __call__ and call method, use only forward and backward to simplify the understanding
  - forward will be able to track the predecessors to form a DAG of parameters, this will always be helpful.
  - # a forward will

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
