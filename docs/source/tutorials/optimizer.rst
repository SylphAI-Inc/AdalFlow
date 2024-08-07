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


Weigted sampling [test negative case]
--------------------------------------
Assume an example input x with y_pred.
WE will have the teacher with y_pred_t.
When we sample from [(x_i, y_pred_t)], we assign weight w_i = (score_i_t - score_i)
Then we can sample from the weighted samples.
When `score_i_t` is 1, and `score_i` is 0, it has a high chance to be sampled.
When `score_i_t` is 1, and `score_i` is 1, it has a low chance to be sampled.
When `score_i_t` is 0, and `score_i` is 1, and there is another case (1), -1 + 1 = 0+delta, -1/delta = -inf, 1/delta = inf, so we will still sample 1.
When `score_i_t` is 0, and `score_i` is 0, it has a low chance to be sampled.

For samples T-A, all of them will have weights 0, if they are in the traces, their weights will be 1-score_i,
still give higher chance to sample the negative cases.

Here shows a subset of students traces:

.. code-block:: json

    {
        "2d72e6e7-71a2-428d-90ff-6986ba52a0d3": {
            "id": "2d72e6e7-71a2-428d-90ff-6986ba52a0d3",
            "question": "I have a yam, a cauliflower, a bed, two cabbages, a garlic, an oven, a carrot, a head of broccoli, a potato, a stalk of celery, a lettuce head, and a toaster. How many vegetables do I have?",
            "answer": "To determine the total number of vegetables, we need to count each type of vegetable and add them together.\n\n1. Yam: 1\n2. Cauliflower: 1\n3. Cabbages: 2\n4. Garlic: 1\n5. Carrot: 1\n6. Broccoli: 1\n7. Potato: 1\n8. Celery: 1\n9. Lettuce: 1\n\nNow, we sum these values:\n\n\\[ 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 1 = 10 \\]\n\nSo, the total number of vegetables is:\n\nAnswer: 10",
            "score": 1.0
        },
        "37da1fbf-402c-44fc-a8b8-a45ad4690a47": {
            "id": "37da1fbf-402c-44fc-a8b8-a45ad4690a47",
            "question": "I have an apple, three bananas, a strawberry, a peach, three oranges, a plum, a raspberry, two grapes, a nectarine, and a blackberry. How many fruits do I have?",
            "answer": "First, list each fruit mentioned:\n\n1. Apple\n2. Three bananas\n3. Strawberry\n4. Peach\n5. Three oranges\n6. Plum\n7. Raspberry\n8. Two grapes\n9. Nectarine\n10. Blackberry\n\nNow, count each item:\n\n1. Apple\n2. Three bananas\n3. Strawberry\n4. Peach\n5. Three oranges\n6. Plum\n7. Raspberry\n8. Two grapes\n9. Nectarine\n10. Blackberry\n\nAdd these numbers:\n\n1 + 3 + 1 + 1 + 3 + 1 + 1 + 2 + 1 + 1 = 14\n\nAnswer: 14",
            "score": 0.0
        },
        "38404a09-0191-4f6f-8a12-4a392e464fe7": {
            "id": "38404a09-0191-4f6f-8a12-4a392e464fe7",
            "question": "I have a cauliflower, a stalk of celery, a cabbage, and a garlic. How many vegetables do I have?",
            "answer": "First, list each vegetable mentioned:\n\n1. Cauliflower\n2. Stalk of celery\n3. Cabbage\n4. Garlic\n\nNow, count each item:\n\n1. Cauliflower\n2. Stalk of celery\n3. Cabbage\n4. Garlic\n\nAdd these numbers:\n\n1 + 1 + 1 + 1 = 4\n\nAnswer: 4",
            "score": 1.0
        },
        "5869e6cd-bd3a-4682-bf52-417ab8b4e3cc": {
            "id": "5869e6cd-bd3a-4682-bf52-417ab8b4e3cc",
            "question": "I have an orange, three strawberries, an apple, three bananas, three raspberries, a peach, a blackberry, a grape, a plum, and two nectarines. How many fruits do I have?",
            "answer": "First, list each fruit mentioned:\n\n1. Orange\n2. Three strawberries\n3. Apple\n4. Three bananas\n5. Three raspberries\n6. Peach\n7. Blackberry\n8. Grape\n9. Plum\n10. Two nectarines\n\nNow, count each item:\n\n1. Orange\n2. Three strawberries\n3. Apple\n4. Three bananas\n5. Three raspberries\n6. Peach\n7. Blackberry\n8. Grape\n9. Plum\n10. Two nectarines\n\nAdd these numbers:\n\n1 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 2 = 16\n\nAnswer: 16",
            "score": 0.0
        }
    }

Here is the teacher traces:

.. code-block:: json
    {

        "2d72e6e7-71a2-428d-90ff-6986ba52a0d3": {
            "id": "2d72e6e7-71a2-428d-90ff-6986ba52a0d3",
            "question": "I have a yam, a cauliflower, a bed, two cabbages, a garlic, an oven, a carrot, a head of broccoli, a potato, a stalk of celery, a lettuce head, and a toaster. How many vegetables do I have?",
            "answer": "Let's identify which items from the given list are vegetables:\n\n1. Yam - vegetable\n2. Cauliflower - vegetable\n3. Bed - not a vegetable\n4. Two cabbages - vegetables (counted as two)\n5. Garlic - vegetable\n6. Oven - not a vegetable\n7. Carrot - vegetable\n8. Head of broccoli - vegetable\n9. Potato - vegetable\n10. Stalk of celery - vegetable\n11. Lettuce head - vegetable\n12. Toaster - not a vegetable\n\nNow, let's count the vegetables:\n\n1. Yam\n2. Cauliflower\n3. Two cabbages (counted as two)\n4. Garlic\n5. Carrot\n6. Head of broccoli\n7. Potato\n8. Stalk of celery\n9. Lettuce head\n\nAdding these up, we get:\n\n1 (yam) + 1 (cauliflower) + 2 (cabbages) + 1 (garlic) + 1 (carrot) + 1 (broccoli) + 1 (potato) + 1 (celery) + 1 (lettuce) = 10\n\nAnswer: 10",
            "score": 1.0
        },
        "37da1fbf-402c-44fc-a8b8-a45ad4690a47": {
            "id": "37da1fbf-402c-44fc-a8b8-a45ad4690a47",
            "question": "I have an apple, three bananas, a strawberry, a peach, three oranges, a plum, a raspberry, two grapes, a nectarine, and a blackberry. How many fruits do I have?",
            "answer": "To solve this problem, we need to count the total number of fruits listed.\n\n1. One apple: 1\n2. Three bananas: 3\n3. One strawberry: 1\n4. One peach: 1\n5. Three oranges: 3\n6. One plum: 1\n7. One raspberry: 1\n8. Two grapes: 2\n9. One nectarine: 1\n10. One blackberry: 1\n\nNext, we add these quantities together:\n\n1 + 3 + 1 + 1 + 3 + 1 + 1 + 2 + 1 + 1 = 15\n\nThus, the total number of fruits is 15.\n\nAnswer: 15",
            "score": 1.0
        },
        "38404a09-0191-4f6f-8a12-4a392e464fe7": {
            "id": "38404a09-0191-4f6f-8a12-4a392e464fe7",
            "question": "I have a cauliflower, a stalk of celery, a cabbage, and a garlic. How many vegetables do I have?",
            "answer": "To determine the total number of vegetables, let's identify and count each item mentioned:\n\n1. Cauliflower\n2. Stalk of celery\n3. Cabbage\n4. Garlic\n\nEach item listed is a vegetable. Now, we simply count them:\n\n1 (Cauliflower) + 1 (Stalk of celery) + 1 (Cabbage) + 1 (Garlic) = 4\n\nAnswer: 4",
            "score": 1.0
        },
        "5869e6cd-bd3a-4682-bf52-417ab8b4e3cc": {
            "id": "5869e6cd-bd3a-4682-bf52-417ab8b4e3cc",
            "question": "I have an orange, three strawberries, an apple, three bananas, three raspberries, a peach, a blackberry, a grape, a plum, and two nectarines. How many fruits do I have?",
            "answer": "To determine the total number of fruits, we need to count each type of fruit and add them together.\n\n1. Orange: 1\n2. Strawberries: 3\n3. Apple: 1\n4. Bananas: 3\n5. Raspberries: 3\n6. Peach: 1\n7. Blackberry: 1\n8. Grape: 1\n9. Plum: 1\n10. Nectarines: 2\n\nNow, we sum these values:\n\\[ 1 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 2 = 17 \\]\n\nSo, the total number of fruits is:\n\nAnswer: 17",
            "score": 1.0
        }
    }

When we allow 2 augmented samples, here is what it has sampled:

.. code-block:: json

    [
        {
            "id": "37da1fbf-402c-44fc-a8b8-a45ad4690a47",
            "question": "I have an apple, three bananas, a strawberry, a peach, three oranges, a plum, a raspberry, two grapes, a nectarine, and a blackberry. How many fruits do I have?",
            "answer": "To solve this problem, we need to count the total number of fruits listed.\n\n1. One apple: 1\n2. Three bananas: 3\n3. One strawberry: 1\n4. One peach: 1\n5. Three oranges: 3\n6. One plum: 1\n7. One raspberry: 1\n8. Two grapes: 2\n9. One nectarine: 1\n10. One blackberry: 1\n\nNext, we add these quantities together:\n\n1 + 3 + 1 + 1 + 3 + 1 + 1 + 2 + 1 + 1 = 15\n\nThus, the total number of fruits is 15.\n\nAnswer: 15",
            "score": 1.0
        },
        {
            "id": "5869e6cd-bd3a-4682-bf52-417ab8b4e3cc",
            "question": "I have an orange, three strawberries, an apple, three bananas, three raspberries, a peach, a blackberry, a grape, a plum, and two nectarines. How many fruits do I have?",
            "answer": "To determine the total number of fruits, we need to count each type of fruit and add them together.\n\n1. Orange: 1\n2. Strawberries: 3\n3. Apple: 1\n4. Bananas: 3\n5. Raspberries: 3\n6. Peach: 1\n7. Blackberry: 1\n8. Grape: 1\n9. Plum: 1\n10. Nectarines: 2\n\nNow, we sum these values:\n\\[ 1 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 2 = 17 \\]\n\nSo, the total number of fruits is:\n\nAnswer: 17",
            "score": 1.0
        }
    ]






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
