.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
      <a href="https://colab.research.google.com/drive/1_YnD4HshzPRARvishoU4IA-qQuX9jHrT?usp=sharing" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
   </div>


Core Concepts
=================

LLM-AutoDiff
-----------------
AdalFlow mainly relies on `LLM-AutoDiff <https://arxiv.org/abs/2501.16673>`__ to do automatic prompt engineering(APE).

Similar to Auto-differentiation in PyTorch, LLM-AutoDiff works by forming a runtime computation graph of prompts, hyperparameters, intermediate outputs, and losses in the `forward` pass.
In the `backward` pass, the `backward engine` LLM we put at each node will work together to identify which prompts are the cause of errors, so that a `feedback`-driven LM optimizer can leverage it to propose new prompts.


Components
-----------------
:class:`Component<core.component>` is to LM task pipelines what `nn.Module` is to PyTorch models. It is the base class for components such as ``Prompt``, ``ModelClient``, ``Generator``, ``Retriever`` in AdalFlow.
Your task pipeline should also subclass from ``Component``.
A component can recursively contain and register other components, allow easy control of (1) `training` and `inference` modes, (2) visualization of the workflow structure, and (3) serialization and deserialization of the component.


We require a component to have (1) a `call` method, which will be called during the `inference` time,
and (2) a `forward` method, which will be called during the `training` time and output a `Parameter` object which has the output of the component wrapped in the `data` field.

There are four main types of components in AdalFlow:

1. `Component`: the base class of all components. Used for LLM workflows. Supports both training and evaluation modes with `forward` (train mode), `call` (eval mode), and `bicall` (supports both eval and train modes).

.. code-block:: python

    class ObjectCountTaskPipeline(adal.Component):
        def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
            super().__init__()

            self.llm_counter = adal.Generator(
                model_client=model_client,
                model_kwargs=model_kwargs,
                template="User: {{input_str}}",
                output_processors=parse_integer_answer,
            )

        def bicall(self, question: str, id: str = None):
            return self.llm_counter(prompt_kwargs={"input_str": question}, id=id)

2. `GradComponent`: a subclass of `Component` that has a `backward` method. It defines a unit of computation that are capable of backpropagation. Our `Generator` and `Retriever` are GradComponents.

.. code-block:: python

    generator = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
        template="User: {{question}}"
    )

    output = generator(prompt_kwargs={"question": "How many apples do I have?"})

3. `DataComponent`: a subclass of `Component` that only has `call` method and does not handle any `Parameter` object. Examples include `Prompt`, `DataClassParser` which only handles the data formatting but rather the transformation.

.. code-block:: python

    @adal.func_to_data_component
    def parse_integer_answer(answer: str):
        import re
        numbers = re.findall(r"\d+", answer)
        return int(numbers[-1]) if numbers else -1

4. `LossComponent`: a subclass of `Component` that likely takes an evaluation metric function and has a `backward` method. When it is attached to your LM workflow's output, the whole training pipeline is capable of backpropagation.

.. code-block:: python

    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    loss_fn = adal.EvalFnToTextLoss(
        eval_fn=eval_fn,
        eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0"
    )


For a full walkthrough of the component system, please refer to :ref:`Full Tutorial <question_answering>`.

Parameters
-----------------
:class:`Parameter<optim.parameter.Parameter>` are used to save (1) intermediate forward data and (2) gradients/feedback, and (3) graph information such as `predecessors`.
It also has function like `draw_graph` to help you visualize the structure of your computation graph.



DataClass and Structured Output
----------------------------------
:class:`DataClass<core.base_data_class.DataClass>` is used for developers to define a data model.
Similar to `Pydantic`, it has methods like `to_yaml_signature`, `to_json_signature`, `to_yaml`, `to_json` to help you generate the data model schema and to generate the json/yaml data representation as strings.
It can be best used together with :class:`DataClassParser<components.output_parsers.dataclass_parser.DataClassParser>` for structured output.

.. code-block:: python

    from dataclasses import dataclass, field

    @dataclass
    class TrecData:
        question: str = field(
            metadata={"desc": "The question asked by the user"}
        )
        label: int = field(
            metadata={"desc": "The label of the question"}, default=0
        )

``DataClass`` covers the following:

1. Generate the class ``schema`` and ``signature`` (less verbose) to describe the data format to LLMs.
2. Convert the data instance to a json or yaml string to show the data example to LLMs.
3. Load the data instance from a json or yaml string to get the data instance back to be processed in the program.

Checkout :ref:`Developer Notes - DataClass <core-base_data_class_note>` for more details.
