.. <a href="https://colab.research.google.com/drive/1n3mHUWekTEYHiBdYBTw43TKlPN41A9za?usp=sharing" target="_blank" style="margin-right: 10px;">
..          <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
..       </a>

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">

      <a href="https://github.com/SylphAI-Inc/AdalFlow/tree/main/benchmarks/hotpot_qa/" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

RAG optimization
====================

In this tutorial, we will cover the auto-optimization of a standard RAG:

- Introducing `HotPotQA` dataset and `HotPotQAData` class.
- Convert `Dspy`'s Retriever to AdalFlow's `Retriever` to easy comparison.
- Build the standard RAG with `Retriever` and `Generator` components.
- Learn how to connect the output-input between components to enable auto-text-grad optimization.


HotPotQA dataset
------------------
:class:`datasets.hotpotqa.HotPotQA` dataset is widely used in research community to benchmark QA and RAG tasks.
Here is one example of the dataset:

.. code-block:: python

    from adalflow.datasets.hotpot_qa import HotPotQA, HotPotQAData

    dataset = HotPotQA(split="train", size=20)
    print(dataset[0], type(dataset[0]))

The output will be:

.. code-block:: python

    HotPotQAData(id='5a8b57f25542995d1e6f1371', question='Were Scott Derrickson and Ed Wood of the same nationality?', answer='yes', gold_titles="{'Scott Derrickson', 'Ed Wood'}")

The dataset comes with more text that you can retrieve from. But in this tutorial,
we will use the `retriever` from DsPy library which allows you to retrieve relevant passages from a large wikipedia corpus using BERT embedding model.

Retriever
------------------
With the following code, we can easily convert the `Dspy`'s Retriever to AdalFlow's `Retriever`:

.. code-block:: python

    import adaflow as adal
    import dspy

    class DspyRetriever(adal.Retriever):
        def __init__(self, top_k: int = 3):
            super().__init__()
            self.top_k = top_k
            self.dspy_retriever = dspy.Retrieve(k=top_k)

        def call(self, input: str, top_k: Optional[int] = None) -> List[adal.RetrieverOutput]:

            k = top_k or self.top_k

            output = self.dspy_retriever(query_or_queries=input, k=k)
            final_output: List[RetrieverOutput] = []
            documents = output.passages

            final_output.append(
                RetrieverOutput(
                    query=input,
                    documents=documents,
                    doc_indices=[],
                )
            )
            return final_output

Let's try one example:

.. code-block:: python

    def test_retriever():
        question = "How many storeys are in the castle that David Gregory inherited?"
        retriever = DspyRetriever(top_k=3)
        retriever_out = retriever(input=question)
        print(f"retriever_out: {retriever_out}")

The output will be:

.. code-block:: python

    [RetrieverOutput(doc_indices=[], doc_scores=None, query='How many storeys are in the castle that David Gregory inherited?', documents=['David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory\'s use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.', 'St. Gregory Hotel | The St. Gregory Hotel is a boutique hotel located in downtown Washington, D.C., in the United States. Established in 2000, the nine-floor hotel has 155 rooms, which includes 54 deluxe rooms, 85 suites with kitchens, and 16 top-floor suites with balconies. The hotel, which changed hands in June 2015, has a life-size statue of Marilyn Monroe in the lobby.', 'Karl D. Gregory Cooperative House | Karl D. Gregory Cooperative House is a member of the Inter-Cooperative Council at the University of Michigan. The structure that stands at 1617 Washtenaw was originally built in 1909 for the Tau Gamma Nu fraternity, but was purchased by the ICC in 1995. Gregory House is the only house in the organization that is expressly substance free. No tobacco, alcohol, or illicit drugs are allowed on the property. Gregory House has a maximum capacity of 29 members (by way of 13 single and 8 double capacity rooms) as of June 2008.'])]

Trainable RAG
------------------

In other tutorials, we used only one component - `Generator`, so there is no need to connect the output-input between components.
That was why we had written our task pipeline with a single `call` method that has both `training` and `inference` mode.

The previous task pipeline `call` method will return `GeneratorOutput` in inference mode and `Parameter` in training mode.

.. code-block:: python

    def call(
         self, question: str, id: Optional[str] = None
     ) -> Union[adal.GeneratorOutput, adal.Parameter]:
         prompt_kwargs = self._prepare_input(question)
         output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
         return output

The above code works as the `__call__` method of the `Generator` component supports `training` and `inference` mode already.

**Inference mode**

We can separate the `inference mode` and `training mode` by using `call` and `forward` methods separately.
Implementing `call` method is just free-form pythonic coding.
Assume this class already have `retriever` and `llm` configured in the `__init__` method.

Here is our call method:

.. code-block:: python

    def call(self, question: str, id: str = None) -> adal.GeneratorOutput:
        if self.training:
            raise ValueError(
                "This component is not supposed to be called in training mode"
            )

        retriever_out = self.retriever.call(input=question)

        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x[0].documents) if x and x[0] and x[0].documents else ""
        )
        retrieved_context = successor_map_fn(retriever_out)

        prompt_kwargs = {
            "context": retrieved_context,
            "question": question,
        }

        output = self.llm.call(
            prompt_kwargs=prompt_kwargs,
            id=id,
        )
        return output

Between the `retriever` and `llm` components, we converted the retriever output to a string and passed it to the `llm` component via the `prompt_kwargs`.


**Training mode**

In this case, we need to create a trainable RAG pipeline with both `Retriever` and `Generator` components.
Here in particular, we show how to write the `forward`(training) method of the `RAG` component.
To make a pipeline trainable, we need to pass `Parameter` as input and output between components.

The `foward` method of `Generator` will use `Parameter` to build a dynamic computation graph, and we need a way to convert the output of the `Retriever` to the input of the `Generator`.
We achieve this via using the `successor_map_fn` of `Parameter` class.
In this case, the `data` field of the `Parameter` saved the output of the `Retriever`.
The `successor_map_fn` will apply a mapping function to convert the `Parameter` output from the Retriever to the string format used in the `Generator` prompt.

Here is our `RAG` component's `forward` method:

.. code-block:: python

    def forward(self, question: str, id: str = None) -> adal.Parameter:
        if not self.training:
            raise ValueError("This component is not supposed to be called in eval mode")
        retriever_out = self.retriever.forward(input=question)
        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x.data[0].documents)
            if x.data and x.data[0] and x.data[0].documents
            else ""
        )
        retriever_out.add_successor_map_fn(successor=self.llm, map_fn=successor_map_fn)
        generator_out = self.llm.forward(
            prompt_kwargs={"question": question, "context": retriever_out}, id=id
        )
        return generator_out

**Both modes in the same method**

You also still has the option to put two methods together. Here is one example:

.. code-block:: python

    def bicall(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        """You can also combine both the forward and call in the same function.
        Supports both training and eval mode by using __call__ for GradComponents
        like Retriever and Generator
        """
        retriever_out = self.retriever(input=question)
        if isinstance(retriever_out, adal.Parameter):
            successor_map_fn = lambda x: (  # noqa E731
                "\n\n".join(x.data[0].documents)
                if x.data and x.data[0] and x.data[0].documents
                else ""
            )
            retriever_out.add_successor_map_fn(
                successor=self.llm, map_fn=successor_map_fn
            )
        else:
            successor_map_fn = lambda x: (  # noqa E731
                "\n\n".join(x[0].documents) if x and x[0] and x[0].documents else ""
            )
            retrieved_context = successor_map_fn(retriever_out)
        prompt_kwargs = {
            "context": retrieved_context,
            "question": question,
        }
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output

**Trainable Parameters**

In this task, we will define two trainable parameters: one to optimize the task description and one to do few-shot learning.

Here is our `Task` class:

.. code-block:: python

    task_desc_str = r"""Answer questions with short factoid answers.

    You will receive context(may contain relevant facts) and a question.
    Think step by step."""


    class VanillaRAG(adal.GradComponent):
        def __init__(self, passages_per_hop=3, model_client=None, model_kwargs=None):
            super().__init__()

            self.passages_per_hop = passages_per_hop

            self.retriever = DspyRetriever(top_k=passages_per_hop)
            self.llm_parser = adal.DataClassParser(
                data_class=AnswerData, return_data_class=True, format_type="json"
            )
            self.llm = Generator(
                model_client=model_client,
                model_kwargs=model_kwargs,
                prompt_kwargs={
                    "task_desc_str": adal.Parameter(
                        data=task_desc_str,
                        role_desc="Task description for the language model",
                        param_type=adal.ParameterType.PROMPT,
                    ),
                    "few_shot_demos": adal.Parameter(
                        data=None,
                        requires_opt=True,
                        role_desc="To provide few shot demos to the language model",
                        param_type=adal.ParameterType.DEMOS,
                    ),
                    "output_format_str": self.llm_parser.get_output_format_str(),
                },
                template=answer_template,
                output_processors=self.llm_parser,
                use_cache=True,
            )


Prepare for Training
---------------------

First, we need to create a `AdalComponent` to help configs the `Trainer`.

* In the `__init__:  `eval_fn` and `loss_fn`, `task`, and configure `teacher generator` for few-shot learning,
and `backward_engine` and `optimizer` for text_grad optimization.

* Minimumly, we need to let the `Trainer` know (1) how to call the task pipeline in both modes.
(2) in inference/eval mode, how to parse the last output (GenereratorOutput) to the input of the `eval_fn`.
(3) in training mode, how to parse the last output (Parameter) to the input of the `loss_fn`.

Here is the `AdalComponent` class for the `VanillaRAG` task:

.. code-block:: python

    class VallinaRAGAdal(adal.AdalComponent):
        def __init__(
            self,
            model_client: adal.ModelClient,
            model_kwargs: Dict,
            backward_engine_model_config: Dict | None = None,
            teacher_model_config: Dict | None = None,
            text_optimizer_model_config: Dict | None = None,
        ):
            task = VanillaRAG(
                model_client=model_client,
                model_kwargs=model_kwargs,
                passages_per_hop=3,
            )
            eval_fn = AnswerMatchAcc(type="fuzzy_match").compute_single_item
            loss_fn = adal.EvalFnToTextLoss(
                eval_fn=eval_fn, eval_fn_desc="fuzzy_match: 1 if str(y) in str(y_gt) else 0"
            )
            super().__init__(
                task=task,
                eval_fn=eval_fn,
                loss_fn=loss_fn,
                backward_engine_model_config=backward_engine_model_config,
                teacher_model_config=teacher_model_config,
                text_optimizer_model_config=text_optimizer_model_config,
            )

        # tell the trainer how to call the task
        def handle_one_task_sample(
            self, sample: HotPotQAData
        ) -> Tuple[Callable[..., Any], Dict]:
            if self.task.training:
                return self.task.forward, {"question": sample.question, "id": sample.id}
            else:
                return self.task.call, {"question": sample.question, "id": sample.id}


        # eval mode: get the generator output, directly engage with the eval_fn
        def evaluate_one_sample(
            self, sample: HotPotQAData, y_pred: adal.GeneratorOutput
        ) -> float:
            y_label = ""
            if y_pred and y_pred.data and y_pred.data.answer:
                y_label = y_pred.data.answer
            return self.eval_fn(y=y_label, y_gt=sample.answer)

        # train mode: get the loss and get the data from the full_response
        def handle_one_loss_sample(self, sample: HotPotQAData, pred: adal.Parameter):
            y_gt = adal.Parameter(
                name="y_gt",
                data=sample.answer,
                eval_input=sample.answer,
                requires_opt=False,
            )

            pred.eval_input = (
                pred.full_response.data.answer
                if pred.full_response
                and pred.full_response.data
                and pred.full_response.data.answer
                else ""
            )
            return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}

Diagnose
---------------------
Before we start training, we decided to diagnose the pipeline and to analyze the current performance before the optimization.
Here is the code:

.. code-block:: python


    def train_diagnose(
        model_client: adal.ModelClient,
        model_kwargs: Dict,
    ) -> Dict:

        trainset, valset, testset = load_datasets()

        adal_component = VallinaRAGAdal(
            model_client,
            model_kwargs,
            backward_engine_model_config=gpt_4o_model,
            teacher_model_config=gpt_3_model,
            text_optimizer_model_config=gpt_3_model,
        )
        trainer = adal.Trainer(adaltask=adal_component)
        trainer.diagnose(dataset=trainset, split="train")
        # trainer.diagnose(dataset=valset, split="val")
        # trainer.diagnose(dataset=testset, split="test")

From this, I have discovered that the inital evaluation will think `Yes` and `yes` are different answers.
We fixed the evaluator to use the lower case of the prediction and the ground truth to compare.
The pipeline without optimizaton achieved around :math:`0.6` accuracy on the test set.

Training
---------------------
First, we train with only the supervision on the final generation answer, without the retriever supervision.





.. admonition:: API reference
   :class: highlight

   - :class:`datasets.hotpotqa.HotPotQA`
