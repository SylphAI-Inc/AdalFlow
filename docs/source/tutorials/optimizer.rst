.. _optimizer:

Optimizer
==========================================================


Optimizing strategy
--------------------
In general, its good to start with multi-stage training. First, find a good zer-shot prompt, and then do few-shot training.

You can start with zero-shot ICL, training only prompts arguments/templates. With the option to
use ``instruction_to_optimizer`` to create synthetic examples.

.. code-block:: python

    system_prompt = Parameter(
            alias="task_instruction",
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=ParameterType.NONE,
            instruction_to_optimizer="You can show some examples if you think that will help.",
        )

Optionally, you can do both prompt and few-shot training together.

How we implemented the boostrap few-shot training
--------------------------------------------------

Demo parameter:
Import fields

.. code-block:: python

    param_type = ParameterType.DEMOS
    requires_opt = True
    alias = "few_shot_demos"
    data = None
    role_desc = "To provide few shot demos to the language model"
    _traces: Dict[str, DataClass] = {} # teacher mode traces
    _score: float = 0.0  # end to end score, used for once-off parameter such as y_pred, or any intermedia component output to have the score




1. we add `use_teacher` method and `teacher_mode`(bool) attribute to `Component` which works recursively similar to `.train()` method.
2. We add `set_teacher_generatpr` method to `Generator` class which whenever the `teacher_mode` is True, it will do the forward pass with teacher's `call` method.
   Additionally, it works with `demo_class`, `input_mapping`, and `output_mapping` to create a demo instance, along with `id` we passed optionally to the generator.
   It will add the demo instance to its parameters that are of `ParameterType.DEMOS` types via the `_traces` attributes in the `Parameter` class.
   So far, we can find the input, output and via being ``DataClass``, we can easily convert it to string. However, we still miss the `score` that will be used to sample bootstrap samples.
   This means if we pass the `demo_class`, `input_mapping`, and `output_mapping` to the `Generator` class, we can trace all inputs and outputs while it is in `teacher_mode` to the
   parameter you defined as `ParameterType.DEMOS` type.

Here is an example for passing `DEMOs`:

.. code-block:: python

    @dataclass
    class ObjectCountSimple(DataClass):
        """Dataclass for string output"""
        id: str = field(
            default=None,
            metadata={"desc": "The unique identifier of the example"},
        )

        question: str = field(
            default=None,
            metadata={"desc": "The question to be answered"},
        )

        answer: str = field(
            default=None,
            metadata={"desc": "The raw answer to the question"},
        )
        score: float = field(
            default=None,
            metadata={
                "desc": "The score of the answer, in range [0, 1]. The higher the better"
            },
        )

    _few_shot_demos = Parameter(
            alias="few_shot_demos",
            data=None,
            role_desc="To provide few shot demos to the language model",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

    self.llm_counter = Generator(
        model_client=model_client,
        model_kwargs=model_kwargs,
        template=few_shot_template,
        prompt_kwargs={
            "system_prompt": system_prompt,
            "few_shot_demos": _few_shot_demos,
        },
        output_processors=parse_integer_answer,  # transform data field
        use_cache=True,
        demo_data_class=ObjectCountSimple,  # for output format
        demo_data_class_input_mapping={"question": "input_str"},
        demo_data_class_output_mapping={"answer": lambda x: x.raw_response},
    )

3. We need to pass eval score to those traces. We leverage the loss function in text-grad, in the backpropogation, on the loss parameter, we will pass back the score to
   its predecessors such as `y_pred`. So in ``EvalFnToTextLoss``, we have

.. code-block:: python

    pred._score = respose.data


When in teacher mode, we should only have the demo backpropagation and without text-gradient backpropogation (so this will not end up consuming llm calls).
It becomes a way to trace runs.


Tracing
--------------------
The ``Trainer`` additionally will provide a one round trace for each generator in the task pipeline
if we set up an empty demo parameter. and run one round of forward, eval, and backward.
This will be useful to gather training data or even bootstrap a training dataset.

**Backpropagate in student mode. **

generator will have
This will backpropagate the eval response



Bootstrap samples




Implementation
--------------------

.. code-block:: python

    self.task.train() # ensure we use forward that will return a parameter and then we can attach the backward engine for gradients, and if it has a teacher, we will attach a demo propose function.
