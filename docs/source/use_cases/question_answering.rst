.. _question_answering:

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/qas/adalflow_object_count_auto_optimization.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/tree/main/use_cases/question_answering/bbh/object_count" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

Question Answering
===============================


AdalFlow provides token-efficient and high-performing prompt optimization within a unified framework.

This will be our first tutorial on end to end task pipeline optimization with AdalFlow.

Overview
----------------
In this tutorial, we will build and optimize a question-answering task pipeline.
Specifically, the task is to count the total number of objects.
Here is an example from the dataset:

.. code-block:: python

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

LLM has to understand the type of objects to count and provide the correct answer.

For optimization, we will demonstrate both the instruction/prompt optimization [1]_ using text-grad and few-shot In-context Learning(ICL) [2]_.

**Instruction/prompt Optimization**

We especially want to see how the optimizer performs with both good and bad starting prompts.

With a low-performing starting prompt, our zero-shot optimizer can achieve a 90% accuracy on the validation and test sets, a 36% and 25% improvement, respectively.
It converged within 5 steps, with each batch containing only 4 samples.


.. list-table:: Scores by Method and Split On Low-performing Starting Prompt (gpt-3.5-turbo)
   :header-rows: 1
   :widths: 20 20 20 20

   * - Method
     - Train
     - Val
     - Test
   * - Start (manual prompt)
     - N/A (50 samples)
     - 0.54 (50 samples)
     - 0.65 (100 samples)
   * - Optimized Zero-shot
     - N/A
     - 0.9 (**+36%**)
     - 0.9 (**+25%**)



.. list-table:: Manual Prompt vs Optimized Prompt (gpt-3.5-turbo)
   :header-rows: 1
   :widths: 20 20

   * - Method
     - Prompt
   * - Manual
     - You will answer a reasoning question. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
   * - Optimized (zero-shot) (90% on val, 90% on test)
     - You will answer a reasoning question by performing detailed and careful counting of each item. Ensure no items, particularly those in plural form, are miscounted. The last line of your response should be formatted as follows: 'Answer: $VALUE' where VALUE is a numerical value.


We will also demonstrate how to optimize an already high-performing task pipeline (~90% accuracy) to achieve even better results—a process that would be very challenging with manual prompt optimization.

.. list-table:: Scores by Method and Split On High-performing Starting Prompt (gpt-3.5-turbo)
   :header-rows: 1
   :widths: 20 20 20 20

   * - Method
     - Train
     - Val
     - Test
   * - Start (manual prompt)
     - 0.88 (50 samples)
     - 0.90 (50 samples)
     - 0.87 (100 samples)
   * - Optimized Zero-shot
     - N/A
     - 0.98 (**+8%**)
     - 0.91 (**+4%**)


.. list-table:: Manual Prompt vs Optimized Prompt
   :header-rows: 1
   :widths: 20 20

   * - Method
     - Prompt
   * - Manual
     - You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
   * - Optimized (zero-shot) (92% on val, 91% on test)
     - You will answer a reasoning question. Think step by step, and make sure to convert any numbers written in words into numerals. Double-check your calculations. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
   * - Optimized (plus generated examples by itself) (98% on val, 91% on test)
     - You will answer a reasoning question. Think step by step and double-check each calculation you make. Pay close attention to any numerical quantities in the text, converting written numbers into their numerical equivalents. Additionally, re-verify your final answer before concluding. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value. Here are some examples: 1. I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have? Answer: 8

**Bootstrap Few-shot**

We achieved 94% accuracy on the test split with just one bootstrap shot, using only the demonstration of the teacher model's response, surpassing the performance of all existing libraries.

Here is one example of the demonstrated reasoning from the teacher model:

.. code-block:: python

    "Example: 'Let''s count the fruits one by one:\n\n\n  1. Orange: 1\n\n  2. Strawberries: 3\n\n  3. Apple: 1\n\n  4. Bananas: 3\n\n  5. Raspberries: 3\n\n  6. Peach: 1\n\n  7. Blackberry: 1\n\n  8. Grape: 1\n\n  9. Plum: 1\n\n  10. Nectarines: 2\n\n\n  Now, we sum them up:\n\n  \\[ 1 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 2 = 17 \\]\n\n\n  Answer: 17'",

**Overall**

.. list-table:: Optimized Scores comparison on the same prompt on test set (gpt-3.5-turbo)
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Test
   * - Text-grad (start)
     - 0.72
   * - Text-grad (optimized)
     - 0.89
   * - AdalFlow (start)
     - 0.87
   * - AdalFlow(text-grad optimized)
     - 0.91
   * - AdalFlow ("Learn-to-reason" one-shot)
     - **0.94**

Now, let's get started on how to implement and achieve the results mentioned above together.


Build the task pipeline
--------------------------
As we can leverage the optimizer to automatically optimize our task pipeline, we offer a quick way to build it.
We'll instruct the LLM to respond with a chain of thought and end the response with the format Answer: $VALUE. We will use the following code to process it:

.. code-block:: python

    import adalflow as adal
    import re

    @adal.func_to_data_component
    def parse_integer_answer(answer: str):
        """A function that parses the last integer from a string using regular expressions."""
        try:
            # Use regular expression to find all sequences of digits
            numbers = re.findall(r"\d+", answer)
            if numbers:
                # Get the last number found
                answer = int(numbers[-1])
            else:
                answer = -1
        except ValueError:
            answer = -1

        return answer

``adal.func_to_component`` is a decorator that converts a function to a component so that we can pass it to the generator as a output processor.

For the task, we will use a simple template taking three arguments: ``system_prompt``, ``few_shot_demos``, and ``input_str``.

.. code-block:: python

    few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
    {{system_prompt}}
    {# Few shot demos #}
    {% if few_shot_demos is not none %}
    Here are some examples:
    {{few_shot_demos}}
    {% endif %}
    <END_OF_SYSTEM_PROMPT>
    <START_OF_USER>
    {{input_str}}
    <END_OF_USER>
    """


We will create two parameters for training the model: ``system_prompt`` and ``few_shot_demos``.
We will initialize the ``Parameter`` with a ``role_desc`` and ``requires_opt`` to inform the ``backward_engine`` (for feedback/textual gradients) and
the optimizer about the purpose of the parameter.
Additionally, we need to set the ``param_type`` to ``ParameterType.PROMPT`` and ``ParameterType.DEMOS`` so that our trainer can configure the appropriate optimizer to optimize these parameters.

Here is our task pipeline:

.. code-block:: python

    from typing import Dict, Union
    import adalflow as adal


    class ObjectCountTaskPipeline(adal.Component):
        def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
            super().__init__()

            system_prompt = adal.Parameter(
                data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
                role_desc="To give task instruction to the language model in the system prompt",
                requires_opt=True,
                param_type=ParameterType.PROMPT,
            )
            few_shot_demos = adal.Parameter(
                data=None,
                role_desc="To provide few shot demos to the language model",
                requires_opt=True,
                param_type=ParameterType.DEMOS,
            )

            self.llm_counter = adal.Generator(
                model_client=model_client,
                model_kwargs=model_kwargs,
                template=few_shot_template,
                prompt_kwargs={
                    "system_prompt": system_prompt,
                    "few_shot_demos": few_shot_demos,
                },
                output_processors=parse_integer_answer,
                use_cache=True,
            )

        def call(
            self, question: str, id: str = None
        ) -> Union[adal.GeneratorOutput, adal.Parameter]:
            output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
            return output



Here are a few points to keep in mind:

1. Our task pipeline operates in both evaluation and training modes. By default, it will be in evaluation mode and will output a ``GeneratorOutput`` object.
   When in training mode, it will output a ``Parameter`` object where the data attribute contains the raw output from ``GeneratorOutput``.
   The entire GeneratorOutput object will be saved in the ``full_response`` attribute, allowing it to be used later for evaluation.
   To specify which input should be passed to the evaluation function, we will assign it to the ``eval_input`` attribute.

2. If we want to train using few-shot in-context learning, we need to assign an ``id`` to our LLM call. This ``id`` will be used to trace the few-shot examples automatically.

Now, let's pass a ``gpt-3.5-turbo`` model to our task pipeline and test both training and evaluation modes.

.. code-block:: python

    from adalflow.components.model_client.openai_client import OpenAIClient

    adal.setup_env()

    gpt_3_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 2000,
            "temperature": 0.0,
            "top_p": 0.99,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        },
    }

Here is the code to test the task pipeline:

.. code-block:: python

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"
    task_pipeline = ObjectCountTaskPipeline(**gpt_3_model)
    print(task_pipeline)

    answer = task_pipeline(question)
    print(answer)

    # set it to train mode
    task_pipeline.train()
    answer = task_pipeline(question, id="1")
    print(answer)
    print(f"full_response: {answer.full_response}")

The answer for the eval mode:

.. code-block:: python

    GeneratorOutput(id="1", data=8, error=None, usage=CompletionUsage(completion_tokens=113, prompt_tokens=113, total_tokens=226), raw_response='To find the total number of musical instruments you have, you simply need to count the individual instruments you listed. \n\nCounting the instruments:\n1 flute\n1 piano\n1 trombone\n1 violin\n1 accordion\n1 clarinet\n1 drum\n1 trumpet\n\nAdding the number of stoves and lamps, which are not musical instruments:\n4 stoves\n2 lamps\n\nTotal number of musical instruments = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 8\n\nAnswer: 8', metadata=None)

The answer for the train mode:

.. code-block:: python

    Parameter(name=Generator_output, requires_opt=True, param_type=generator_output (The output of the generator.), role_desc=Output from (llm) Generator, data=To find the total number of musical instruments you have, you simply need to count the individual instruments you listed.

    Counting the instruments:
    1 flute
    1 piano
    1 trombone
    1 violin
    1 accordion
    1 clarinet
    1 drum
    1 trumpet

    Adding the number of stoves and lamps, which are not musical instruments:
    4 stoves
    2 lamps

    Total number of musical instruments = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 8

    Answer: 8, predecessors={Parameter(name=To_give_ta, requires_opt=True, param_type=prompt (Instruction to the language model on task, data, and format.), role_desc=To give task instruction to the language model in the system prompt, data=You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value., predecessors=set(), gradients=set(),            raw_response=None, input_args=None, traces={}), Parameter(name=To_provide, requires_opt=True, param_type=demos (A few examples to guide the language model.), role_desc=To provide few shot demos to the language model, data=None, predecessors=set(), gradients=set(),            raw_response=None, input_args=None, traces={})}, gradients=set(),            raw_response=None, input_args={'prompt_kwargs': {'system_prompt': Parameter(name=To_give_ta, requires_opt=True, param_type=prompt (Instruction to the language model on task, data, and format.), role_desc=To give task instruction to the language model in the system prompt, data=You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value., predecessors=set(), gradients=set(),            raw_response=None, input_args=None, traces={}), 'few_shot_demos': Parameter(name=To_provide, requires_opt=True, param_type=demos (A few examples to guide the language model.), role_desc=To provide few shot demos to the language model, data=None, predecessors=set(), gradients=set(),            raw_response=None, input_args=None, traces={}), 'input_str': 'I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?'}, 'model_kwargs': {'model': 'gpt-3.5-turbo', 'max_tokens': 2000, 'temperature': 0.0, 'top_p': 0.99, 'frequency_penalty': 0, 'presence_penalty': 0, 'stop': None}}, traces={})

**Visualize the computation graph**

When in training mode, we are able to visualize the computation graph easily with the following code:

.. code-block:: python

    answer.draw_graph()

Here is the :doc:`computation graph for this task pipeline <../use_cases/qa_computation_graph>`

So far, we have completed the task pipeline and ensured it works in both evaluation and training modes. Of course, if the performance is already perfect, there may be no need for further training, but evaluation is still essential.

Our training pipeline can assist with both training and evaluation.


Evaluate the task pipeline
----------------------------

Before we start the training, we should prepare three datasets: train, validation, and test datasets. An initial evaluation is necessary to check two things:

1. **Overall Performance on Each Data Split:** We need to assess the performance on each data split. If the accuracy does not meet the required standards, we must plan for further evaluation and adjustments.

2. **Performance Consistency Across Datasets:** We need to ensure that each split (train, validation, and test) performs comparably. This consistency is crucial so that the train and validation sets can serve as reliable indicators of test performance.

Datasets
~~~~~~~~~~~~

We have prepared the dataset at ``adalflow.datasets.big_bench_hard``.
We can load it with the following code:

.. code-block:: python

    from adalflow.datasets.big_bench_hard import BigBenchHard
    from adalflow.utils.data import subset_dataset

    def load_datasets(max_samples: int = None):
        """Load the dataset"""
        train_data = BigBenchHard(split="train")
        val_data = BigBenchHard(split="val")
        test_data = BigBenchHard(split="test")

        # Limit the number of samples
        if max_samples:
            train_data = subset_dataset(train_data, max_samples)
            val_data = subset_dataset(val_data, max_samples)
            test_data = subset_dataset(test_data, max_samples)

        return train_data, val_data, test_data

We have 50, 50, 100 samples in the train, val, and test datasets, respectively. Here is one example of the loaded data sample:

.. code-block:: python

    Example(id='b0cffa3e-9dc8-4d8e-82e6-9dd7d34128df', question='I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?', answer='8')

The data sample is already of type ``DataClass`` and each sample is assigned with an ``id``, a ``question``, and an ``answer``.
To note that the answer is in `str` format.


Diagnose the task pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate the task pipeline using the :meth:`diagnose<optim.trainer.trainer.Trainer>` method provided by our trainer,
we can take advantage of the :class:`AdalComponent<optim.trainer.adal.AdalComponent>` interface.
This interface class should be subclassed, allowing us to leverage its parallel processing capabilities, callback configuration, optimizer configuration, and built-in support for the teacher/backward engine.
The AdalComponent works similarly to how PyTorch Lightning's LightningModule interacts with its Trainer.

Here’s the minimum code required to get started on evaluating the task pipeline:

.. code-block:: python

    from adalflow.datasets.types import Example
    from adalflow.eval.answer_match_acc import AnswerMatchAcc


    class ObjectCountAdalComponent(adal.AdalComponent):
        def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
            task = ObjectCountTaskPipeline(model_client, model_kwargs)
            eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
            super().__init__(task=task, eval_fn=eval_fn)

        def prepare_task(self, sample: Example):
            return self.task.call, {"question": sample.question, "id": sample.id}

        def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput) -> float:
            y_label = -1
            if (y_pred is not None and y_pred.data is not None):  # if y_pred and y_pred.data: might introduce bug when the data is 0
                y_label = y_pred.data
            return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

We needed one `eval_fn`, one `task`, and two methods: `prepare_task` and `prepare_eval` that tells `Trainer` how to call the task and how to call the eval function.

Now, lets use the trainer.


.. code-block:: python

    def diagnose(
        model_client: adal.ModelClient,
        model_kwargs: Dict,
    ) -> Dict:
        from use_cases.question_answering.bhh_object_count.data import load_datasets

        trainset, valset, testset = load_datasets()

        adal_component = ObjectCountAdalComponent(model_client, model_kwargs)
        trainer = adal.Trainer(adaltask=adal_component)
        trainer.diagnose(dataset=trainset, split="train")
        trainer.diagnose(dataset=valset, split="val")
        trainer.diagnose(dataset=testset, split="test")

File structure:

.. code-block:: bash

    .adalflow/
    ├── ckpt/
    │   └── ObjectCountAdalComponent/
    │       ├── diagnose_{train, val, test}/  # Directory for training data diagnostics
    │       │   ├── llm_counter_call.jsonl    # Sorted by score from lowest to highest
    │       │   ├── logger_metadata.jsonl
    │       │   ├── llm_counter_diagnose.json # Contains samples with score < 0.5, sorted by score
    │       │   └── stats.json



.. note::

   As we save all data in default at `~/.adalflow`, you can create a soft link to the current directory to access the data easily
   in your code editor.

The `llm_counter_call.jsonl` file will contain 6 keys:

1. "prompt_kwargs": the prompt_kwargs used in the call of ``llm_counter``.
2. "model_kwargs": the model_kwargs used in the call of ``llm_counter``.
3. "input": Everything that passed to the model_client (LLM).
4. "output": GeneratorOutput object.
5. "score": the performance score of the model on the dataset split.
6. "time_stamp": the time stamp of the call.

The items are ranked from the lowest to the highest score. The score is the performance score of the model on the dataset split.
If you have passed the ``id`` to the call, you will find it in the ``output``.

In the ``{}_diagnose.json`` file, we save what can be used to manually diagnose the errors:
- "id": the id of the sample.
- "score": the performance score of the model on the dataset split.
- "prompt_kwargs": the prompt_kwargs used in the call of ``llm_counter``.
- "raw_response": the raw_response of the model.
- "answer": the answer of the sample.
- "dataset_item": the dataset item where you can find sample to compare with.


Here is the stats:

.. list-table:: Scores by Split
   :header-rows: 1

   * - Split
     - Train
     - Val
     - Test
   * - Score
     - 0.88 (50)
     - 0.90 (50)
     - 0.87 (100)

The model already performs quite well on the dataset.
Let's see if we can optimize it further with either few-shot or zero-shot prompt optimization or even both.


Train Setup
------------------------------

Prepare AdalComponent for training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To be able to train, we will add a few attributes and define a few methods in our ``ObjectCountAdalComponent`` class.

First, ``loss_fn`` where we use ``ada.EvalFnToTextLoss`` to compute the loss(``Parameter``) where it takes the ``eval_fn`` and the ``eval_fn_desc`` at the initialization.
This loss function will pass whatever user set at ``kwargs`` to the ``eval_fn`` and compute the loss and handle the ``textual gradient`` for the loss function.
If you intent to train ``ParameterType.PROMPT``, you need to configure the `backward_engine` which is a subclass of `Generator` with its own `template`, along with a `text_optimizer_model_config` which will be used as the optimizer that proposes the new prompt.
If you also want to train ``ParameterType.DEMOS``, you need to configure the `teacher_generator` which is exactly the same setup as your `llm_counter` but with your configured `model_client` and `model_kwargs` that potentially will be a strong teacher model to guide your target model to learn from.

.. code-block:: python

    class ObjectCountAdalComponent(adal.AdalComponent):
        def __init__(
            self,
            model_client: adal.ModelClient,
            model_kwargs: Dict,
            backward_engine_model_config: Dict,
            teacher_model_config: Dict,
            text_optimizer_model_config: Dict,
        ):
            task = ObjectCountTaskPipeline(model_client, model_kwargs)
            eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
            loss_fn = adal.EvalFnToTextLoss(
                eval_fn=eval_fn,
                eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
            )
            super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

            self.backward_engine_model_config = backward_engine_model_config
            self.teacher_model_config = teacher_model_config
            self.text_optimizer_model_config = text_optimizer_model_config



Second, :meth:`prepare_loss` where we will return the loss function and the ``kwargs`` to the loss function.
We need to convert the the ground truth into a ``Parameter`` and set the ``eval_input`` that will be used as value to the ``eval_fn``
when we evaluate the model.

.. code-block:: python

    def prepare_loss(
        self, sample: Example, pred: adal.Parameter
    ) -> Tuple[Callable, Dict[str, Any]]:
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}

Optional[Under the hood]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Under the hood, `AdalComponent` already has three methods to configure the backward engine, the teacher generator, the text optimizer, and the demo optimizer.

.. We provided a ``configure_backward_engine_helper`` method to smooth this setup; it requires only the ``model_client`` and the ``model_kwargs``.

.. code-block:: python

    def configure_backward_engine(self):
        super().configure_backward_engine_helper(
            **self.backward_engine_model_config
        )

.. If we also need to train the ``ParameterType.DEMOS``, we will need to set the ``teacher_generator`` which is exactly the same setup as your ``llm_counter`` but
.. with your configured ``model_client`` and ``model_kwargs``.

.. code-block:: python

    def configure_teacher_generator(self):
        super().configure_teacher_generator_helper(
            **self.teacher_generator_model_config
        )


.. Finally, we need to configure the optimizer. We will use both the ``DemoOptimizer`` (in default configured with ``adal.optim.few_shot.few_shot_optimizer.BootstrapFewShot``) and the ``PromptOptimizer`` (in default configured with ``adal.optim.text_grad.tgd_optimizer.TGDOptimizer``).

.. code-block:: python

    def configure_optimizers(self):
        to = super().configure_text_optimizer_helper(**self.text_optimizer_model_config)
        do = super().configure_demo_optimizer_helper()
        return to  + do

Use the trainer
~~~~~~~~~~~~~~~~~~~~

Now, we can use the trainer to train the model.

.. code-block:: python

    def train(
        train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
        raw_shots: int = 1,
        bootstrap_shots: int = 1,
        max_steps=1,
        num_workers=4,
        strategy="random",
        debug=False,
    ):
        adal_component = ObjectCountAdalComponent(
            **gpt_3_model,
            teacher_model_config=gpt_4o_model,
            text_optimizer_model_config=gpt_4o_model,
            backward_engine_model_config=gpt_4o_model
        )
        print(adal_component)
        trainer = Trainer(
            train_batch_size=train_batch_size,
            strategy=strategy,
            max_steps=max_steps,
            num_workers=num_workers,
            adaltask=adal_component,
            raw_shots=raw_shots,
            bootstrap_shots=bootstrap_shots,
            debug=debug,
            weighted_sampling=True,
        )
        print(trainer)

        train_dataset, val_dataset, test_dataset = load_datasets()
        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            debug=debug,
        )



Train in Debug mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    train(debug=True, max_steps=12, strategy="constrained")

Using the ``debug`` will show us two samples: one successful and one failed sample.
And it will not only check all necessary steps/methods to try its best to ensure you
have implemented all parts correctly before the training on the whole dataset which can be expensive.
Also, it is important to make sure the ``backward_engine`` is giving the right feedback and the ``optimizer`` is
following the instruction to make correct proposal.

When you need more detailed logging, you can add this setup:

.. code-block:: python

    from adalflow.utils import get_logger

    get_logger(level="DEBUG")

.. Debug mode will turn on the log and set it to ``DEBUG`` level.

If everything is fine, you will see the following debug report:

.. figure:: /_static/images/adalflow_debug_report.png
    :align: center
    :alt: AdalFlow debug report
    :width: 620px


    AdalFlow debug report


student_graph

.. code-block:: bash

    .adalflow/
    ├── ckpt/
    │   └── ObjectCountAdalComponent/
    │       ├── diagnose_{train, val, test}/  # Directory for training data diagnostics
    │       │   ├── llm_counter_call.jsonl    # Sorted by score from lowest to highest
    │       │   ├── logger_metadata.jsonl
    │       │   ├── llm_counter_diagnose.json # Contains samples with score < 0.5, sorted by score
    │       │   └── stats.json
    │       ├── debug_text_grads                          # Directory for debug mode with text optimizer
    │       │   ├── lib.log                    # Log file
    │       │   ├── trace_graph_sum.png       # Trace graph with textual feedback and new proposed value
    │       │   ├── trace_graph_sum_root.json # Json representation of the root loss node (sum of the success and fail loss)
    │       |-- debug_demos                           # Directory for debug mode with demo optimizer
    │       │   ├── student_graph
    │       │   │   ├── trace_graph_EvalFnToTextLoss_output_id_6ea5da3c-d414-4aae-8462-75dd1e09abab.png # Trace graph with textual feedback and new proposed value
    │       │   │   ├── trace_graph_EvalFnToTextLoss_output_id_6ea5da3c-d414-4aae-8462-75dd1e09abab_root.json # Json representation of the root loss node (sum of the success and fail loss)

Here is how our trace_graph with text gradients looks like: :doc:`QA text-grad trace graph <qa_text_grad_trace_graph>`.
Here is how our trace_graph with demos looks like: :doc:`QA demos trace graph <qa_demo_trace_graph>`.


Train with Text-Gradient Descent
-----------------------------------
To train, we simply set the ``debug`` to ``False``.

To do textual-gradient descent training for our task pipeline, we will go back to the task pipeline to set the `requires_opt` to `False` for the `few_shot_demos` parameter and
`requires_opt=True` for the `system_prompt` parameter.

.. code-block:: python

    system_prompt = adal.Parameter(
                data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
                role_desc="To give task instruction to the language model in the system prompt",
                requires_opt=True,
                param_type=ParameterType.PROMPT,
            )
    few_shot_demos = adal.Parameter(
        data=None,
        role_desc="To provide few shot demos to the language model",
        requires_opt=False,
        param_type=ParameterType.DEMOS,
    )

For the text optimizer, we have two training strategy: ``random`` and ``constrained``.
The ``random`` strategy runs a batch of loss and backward propagation and then validate it on the ``validation`` and ``test`` dataset at each step.
This is a standard training strategy, and it is used by libraries like ``Dspy`` and ``Text-grad``.
You can refer :meth:`optim.trainer.Trainer.fit` for more details.

The ``constrained`` strategy is unique to AdalFlow library where it runs a moving batch capped at maximum 20 samples, and it subsample the correct and failed samples (each maximum at 4).
Before it runs the validations on the full ``validation`` and ``test`` dataset, it will run a validation on the moving sampled subset and the moving batch. It will try 5 proposals on the moving batch and only let a proposal that can beat the current subset and moving batch performance before it can be validated on the full dataset.
We find it often more effective than the ``random`` strategy.

Additionally, we estimate the maximum validataion score each validation can get. Once we know the maximum score is below our minimum requirement (the last highest validation score), we stop the evaluation to save time and cost.

After the training, we will all information saved in ``.adalflow/ckpt/ObjectCountAdalComponent/``.
With file names like:

.. code-block:: bash

    .adalflow/
    ├── ckpt/
    │   └── ObjectCountAdalComponent/
    │       random_max_steps_8_bb908_run_1.json # The last training run for random strategy
    │       constrained_max_steps_8_a1754_run_1.json # The last training run for constrained strategy


Here is an example of how our ckpt file looks like: :doc:`ckpt_file <../tutorials/ckpt_file>`.
This file is a direct `to_dict`  (json) representation of :class:`TrainerResult<optim.types.TrainerResult>`.


Train with Few-shot Bootstrap
------------------------------
As we have defined a ``ParameterType.DEMOS`` in our ``ObjectCountAdalComponent``, we can train the model with few-shot bootstrap.
We will set ``raw_shots=0`` and ``bootstrap_shots=1`` in the ``train`` method.
In default, our demonstrations use the teacher's direct raw response, with the purpose to teach the weaker model how to reason the answer.
We call this "Learn to reason" few-shot bootstrap.

Note: before we start the training, it will be worth to check if the teacher model is performing better so that the student can learn from the teacher.
We can achieve this using the diagnose method while setting the `model_client` and `model_kwargs` to the teacher model.
Additionally, ensure you set the `split` to `train_teacher` etc to ensure the previous diagnose on the student model is not overwritten.
Here is the teach model performance on the zero-shot prompt:

.. list-table:: Scores by teacher mode (gpt-4o) on the same high-performing starting prompt
   :header-rows: 1
   :widths: 20 20 20 20

   * - Method
     - Train
     - Val
     - Test
   * - Start (manual prompt)
     - 0.98 (50 samples)
     - 1.0 (50 samples)
     - 0.98 (100 samples)


We will show how a single demonstration can help push the model performance to 92% on validation and 97% on test.

To do few-shot for our task pipeline, we will go back to the task pipeline to set the `requires_opt` to `True` for the `few_shot_demos` parameter and
turn off the `requires_opt` for the `system_prompt` parameter.

.. code-block:: python

    system_prompt = adal.Parameter(
                data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
                role_desc="To give task instruction to the language model in the system prompt",
                requires_opt=False,
                param_type=ParameterType.PROMPT,
            )
    few_shot_demos = adal.Parameter(
        data=None,
        role_desc="To provide few shot demos to the language model",
        requires_opt=True,
        param_type=ParameterType.DEMOS,
    )


Here is our top performing few-shot example:

.. list-table:: Scores for One-shot Bootstrap
   :header-rows: 1
   :widths: 10 40 25 25

   * - Method
     - Prompt
     - Val
     - Test
   * - Start
     - None
     - 0.90
     - 0.87
   * - Optimized One-shot
     - """Example: 'To find the total number of objects you have, you need to count each individual\n  item. In this case, you have:\n\n  1 microwave\n\n  1 lamp\n\n  4 cars\n\n  1 stove\n\n  1 toaster\n\n  1 bed\n\n\n  Adding these together:\n\n  1 + 1 + 4 + 1 + 1 + 1 = 9\n\n\n  Therefore, you have 9 objects in total.\n\n  Answer: 9'""
     - 0.96 (**+6%**, 4% < teacher)
     - 0.94 (**+7%**, 4% < teacher)





Benchmarking
------------------------------
We compared our performance with text-grad. Here are our stats:
The same prompt, text-grad gets 0.72 on the validation set. and it optimized it to 0.89.
But text-grad use more lengthy prompt, where it takes more than 80s to run a backpropagation on a batch size of 4.
Yet, we only take 12s.
Also AdalFlow has better converage rate in general.
We also leverage single message prompt, sending the whole template to the model's system message, making this whole development process easy.

.. list-table:: Optimized Scores comparison on the same prompt on test set (gpt-3.5-turbo)
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Test
   * - Text-grad (start)
     - 0.72
   * - Text-grad (optimized)
     - 0.89
   * - AdalFlow (start)
     - 0.87
   * - AdalFlow(text-grad optimized)
     - 0.91
   * - AdalFlow ("Learn-to-reason" one-shot)
     - **0.94**

.. note::
    In the start we use same prompt but we use a single template which achieves much better zero-shot performance than text-grad which sends the system prompt to system message and the input to user message.

.. admonition:: References
   :class: highlight

   .. [1] Text-grad: https://arxiv.org/abs/2406.07496
   .. [2] DsPy: https://arxiv.org/abs/2310.03714
   .. [3] OPRO: https://arxiv.org/abs/2309.03409
