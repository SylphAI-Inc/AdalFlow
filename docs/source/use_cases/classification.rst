Classification Optimization
=============================

Classification is one of the widely used tasks in NLP.
Be able to optimize the GenAI based classification can help developers to quickly develop a well-performing model.
In the longer term, this model can help bootstrap the training of a cheaper and classification model.

Here is what you  will learn from this tutorial:

1. Build a classification task pipeline with structured output
2. Learn the ``mixed`` and ``sequential`` training when we explore both``TextOptimizer``
and ``DemoOptimizer`` to optimize the classification task.
3. Handle the case where the val dataset is not a good indicator to the test accuracy.


Performance Hightlight
-----------------------
Here is the peroformance result, where our optimizers
.. list-table:: Top2 best Zero-shot Optimized Classification on GPT-3.5-turbo
   :header-rows: 1
   :widths: 20 20 20 20

   * - Method
     - Train
     - Val
     - Test
   * - Start (manual prompt)
     - 67.5% (20*6 samples)
     - 69.4% (6*6 samples)
     - 82.64% (144 samples)
   * - Start (GPT-4o/Teacher)
     - 77.5%
     - 77.78%
     - 86.11%
   * - DsPy (Start)
     - 57.5%
     - 61.1%
     - 60.42%
   * - DsPy (bootstrap 4-shots + raw 36-shots)
     - N/A
     - 86.1%
     - 82.6%
   * - AdalFlow (Optimized Zero-shot)
     - N/A
     - 77.78%, 80.5% (**+8.4%**)
     - 86.81%, 89.6% (**+4.2%**)
   * - AdalFlow (Optimized Zero-shot + bootstrap 1-shot)
     - N/A
     - N/A
     - 88.19%
   * - AdalFlow (Optimized Zero-shot + bootstrap 1-shot + 40 raw shots)
     - N/A
     - **86.1%**
     - **90.28%**
   * - AdalFlow (Optimized Zero-shot on GPT-4o)
     - 77.8%
     - 77.78%
     - 84.03%


In this case, Text-Grad 2.0 is able to close the gap to the teacher model, leaving no space for the DemoOptimizer to improve as it learns to boost its reasoning from a teacher model's reasoning.
Even though the many-shots (as many as 40) can still improve the performance for a bit, but it will adds a lot more tokens.


Here is the DsPy's Signature (similar to the prompt) where its task description is a direct copy our AdalFlow's starting prompt:

.. code-block:: python

   class GenerateAnswer(dspy.Signature):
        """You are a classifier. Given a question, you need to classify it into one of the following classes:
        Format: class_index. class_name, class_description
        1. ABBR, Abbreviation
        2. ENTY, Entity
        3. DESC, Description and abstract concept
        4. HUM, Human being
        5. LOC, Location
        6. NUM, Numeric value
        - Do not try to answer the question:"""

        question: str = dspy.InputField(desc="Question to be classified")
        answer: str = dspy.OutputField(
            desc="Select one from ABBR, ENTY, DESC, HUM, LOC, NUM"
        )

AdalFlow starting prompt and data class:

.. code-block:: python

   template = r"""<START_OF_SYSTEM_MESSAGE>
    {{system_prompt}}
    {% if output_format_str is not none %}
    {{output_format_str}}
    {% endif %}
    {% if few_shot_demos is not none %}
    Here are some examples:
    {{few_shot_demos}}
    {% endif %}
    <END_OF_SYSTEM_MESSAGE>
    <START_OF_USER_MESSAGE>
    {{input_str}}
    <END_OF_USER_MESSAGE>
    """

    task_desc_template = r"""You are a classifier. Given a question, you need to classify it into one of the following classes:
    Format: class_index. class_name, class_description
    {% if classes %}
    {% for class in classes %}
    {{loop.index-1}}. {{class.label}}, {{class.desc}}
    {% endfor %}
    {% endif %}
    - Do not try to answer the question:
    """

    @dataclass
    class TRECExtendedData(TrecData):
        rationale: str = field(
            metadata={
                "desc": "Your step-by-step reasoning to classify the question to class_name"
            },
            default=None,
        )
        __input_fields__ = ["question"]
        __output_fields__ = ["rationale", "class_name"]

    # for context, TrecData has the following fields:
    @dataclass
    class TrecData(BaseData):
        __doc__ = """A dataclass for representing examples in the TREC dataset."""
        question: str = field(
            metadata={"desc": "The question to be classified"},
            default=None,
        )
        class_name: str = field(
            metadata={"desc": "One of {ABBR, ENTY, DESC, HUM, LOC, NUM}"},
            default=None,
        )
        class_index: int = field(
            metadata={"desc": "The class label, in range [0, 5]"},
            default=-1,
        )

        __input_fields__ = ["question"]  # follow this order too.
        __output_fields__ = ["class_name", "class_index"]


We can see that being able to flexibly control the prompt instead of delegate to a fixed ``Signature`` is advantageous.
We use ``yaml`` format for the output in this case, and be able to use template to control which part we want to train.

We eventually find that ``TextOptimizer`` works better on smaller instruction prompt.
Here is our Parameters:

.. code-block:: python

           prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=self.parser.get_task_desc_str(),
                role_desc="Task description",
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),
            "output_format_str": adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="Output format requirements",
                requires_opt=False,
                param_type=adal.ParameterType.PROMPT,
            ),
            "few_shot_demos": adal.Parameter(
                data=None,
                requires_opt=True,
                role_desc="Few shot examples to help the model",
                param_type=adal.ParameterType.DEMOS,
            ),
        }

Being able to train each part of the prompt gives us more granular control and in this case, only train ``system_prompt`` instead of training both or train a joined prompt has gained better performance.
And it is also cheaper to propose a smaller prompt.

:note::
    Your can find all our code at ``use_cases/classification`` and the Dspy's implementation at ``benchmarks/trec_classification``.
