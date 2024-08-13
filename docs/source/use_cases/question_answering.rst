Question Answering optimization
===============================

In this tutorial, we will implement and optimize a question answering task pipeline using both string output
and structued output. In particular, it is to count the total objects.

Here is the one example:

.. code-block:: python

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

For optimization, we will demonstrate both few-shot In-context Learning(ICL) and the instruction/prompt optimization.

Build the task pipeline
--------------------------
As we can leverage the optimizer to auto-optimize our task pipeline, we provide a quick way to build the task pipeline.

We will instruct the LLM to response with chain_of_thought and end the response with format 'Answer: $VALUE'.
We will use the following code to process it:

.. code-block:: python

    import adalflow as adal
    import re

    @adal.fun_to_component
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

``adal.fun_to_component`` is a decorator that converts a function to a component so that we can pass it to the generator as a output processor.

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
We will init the ``Parameter`` with a ``role_desc`` and ``requires_opt`` to let the ``backward_engine`` (for feedback/textual gradients) and the optimizer know what the parameter is for.
Also, we have to set the ``param_type`` to ``ParameterType.PROMPT`` and ``ParameterType.DEMOS`` so that the our ``trainer``
can configure the right optimizer to optimize the parameters.
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

1. Our task pipeline has eval and train mode. In default, it will be in eval mode, and it will output a ``GeneratorOutput`` object.
   When in train mode, it will output a ``Parameter`` object where the ``data`` attribute will be the raw output from ``GeneratorOutput``
   and it will save the whole ``GeneratorOutput`` object in the ``full_response`` attribute in case to be used for evaluation.
   To indicate the specific input to the evaluation function, we will pass it to ``eval_input`` attribute.

2. If we want to train few-shot in-context learning, we will have assign an ``id`` to our LLM call.
   The ``id`` will be used to trace the few-shot examples.

Now, lets pass a ``gpt-3.5-turbo`` model to our task pipeline and test both training and evaluation mode.

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

    GeneratorOutput(id=None, data=8, error=None, usage=CompletionUsage(completion_tokens=113, prompt_tokens=113, total_tokens=226), raw_response='To find the total number of musical instruments you have, you simply need to count the individual instruments you listed. \n\nCounting the instruments:\n1 flute\n1 piano\n1 trombone\n1 violin\n1 accordion\n1 clarinet\n1 drum\n1 trumpet\n\nAdding the number of stoves and lamps, which are not musical instruments:\n4 stoves\n2 lamps\n\nTotal number of musical instruments = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 8\n\nAnswer: 8', metadata=None)

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

So far, we have completed the task pipeline and made sure it is working in both eval and train mode.
Of course, if the performance is perfect here, there is no need to train. But we need to evaluate it.
Our train pipeline can help you with both training and evaluation.

# TODO: rerun the example as the id is just added to the call.

Evaluate the task pipeline
----------------------------
Before we start the training, we should prepare three datasets: train, validation, and test datasets.
We need to do intitial evaluation to check two things:

1. The overall performance, maybe an average accross the datasets. If it does not meet the accuracy requirements, we need to plan on evaluation.

2. The performance on each dataset: we need to ensure that each split has a comparable performance so that the train and validation set can be a good indicator to the test performance.

Datasets
~~~~~~~~~~~~

We have prepared the dataset at ``adalflow.datasets.big_bench_hard``.
We can load it:

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
We have a `diagnose` method in our ``trainer`` to help users run evaluation on a dataset split using callbacks and call logger automatically setup by our trainer.
In this case, we will find the following datastructure.

Starting from here, we will use ``AdalComponent`` which is an interface class that we should subclass from.
The ``AdalComponent`` provides parallel processing to run the pipeline, handles call back config, optimizer config, or even teacher/backward engine out of box.
It opens up a few attributes and methods for users to complete by subclass ``AdalComponent``.
This is similar to how ``PyTorch Lightning``'s ``LightningModule`` works with its ``Trainer``.

This minimum code will get us started on evaluating the task pipeline.

.. code-block:: python

    from adalflow.datasets.types import Example
    from adalflow.eval.answer_match_acc import AnswerMatchAcc


    class ObjectCountAdalComponent(adal.AdalComponent):
        def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
            task = ObjectCountTaskPipeline(model_client, model_kwargs)
            eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
            super().__init__(task=task, eval_fn=eval_fn)

        def handle_one_task_sample(self, sample: Example):
            return self.task.call, {"question": sample.question, "id": sample.id}

        def evaluate_one_sample(
            self, sample: Example, y_pred: adal.GeneratorOutput
        ) -> float:
            y_label = -1
            if y_pred and y_pred.data:
                y_label = y_pred.data
            return self.eval_fn(y_label, sample.answer)

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



Tips:
   As we save all data in default at ~/.adalflow, you can create a soft link to the current directory to access the data easily
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


Train-Debug mode
------------------------------

Using the ``debug`` will show us two samples: one successful and one failed sample.
And it will not only check all necessary steps/methods to try its best to ensure you
have implemented all parts correctly before the training on the whole dataset which can be expensive.
Also, it is important to make sure the ``backward_engine`` is giving the right feedback and the ``optimizer`` is
following the instruction to make correct proposal.

Train
------------------------------
