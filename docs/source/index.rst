
*"Say Goodbye to Manual Prompting and No More Vendor Lock-in"*

Getting Started: Install AdalFlow and Run Your First Query
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

   pip install -U adalflow

.. tabs::


   .. tab:: OpenAI

      Set up `OPENAI_API_KEY` in your `.env` file or pass the `api_key` directly to the client.

      .. code-block:: python

         import adalflow as adal
         from adalflow.utils import setup_env

         setup_env()

         openai_llm = adal.Generator(
            model_client=adal.OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
         )
         resopnse = openai_llm(prompt_kwargs={"input_str": "What is LLM?"})


   .. tab:: Groq

      Set up `GROQ_API_KEY` in your `.env` file or pass the `api_key` directly to the client.


      .. code-block:: python

         import adalflow as adal
         from adalflow.utils import setup_env

         setup_env()

         groq_llm = adal.Generator(
            model_client=adal.GroqAPIClient(), model_kwargs={"model": "llama3-8b-8192"}
         )
         resopnse = groq_llm(prompt_kwargs={"input_str": "What is LLM?"})

   .. tab:: Anthropic

      Set up `ANTHROPIC_API_KEY` in your `.env` file or pass the `api_key` directly to the client.

      .. code-block:: python

         import adalflow as adal
         from adalflow.utils import setup_env

         setup_env()

         anthropic_llm = adal.Generator(
            model_client=adal.AnthropicAPIClient(), model_kwargs={"model" "claude-3-opus-20240229"}
         )
         resopnse = anthropic_llm(prompt_kwargs={"input_str": "What is LLM?"})

   .. tab:: Local

      Ollama is one option. You can also use `vllm` or HuggingFace `transformers`.

      First, install `ollama` and prepare the model.

      .. code-block:: python

         # Download Ollama command line tool
         curl -fsSL https://ollama.com/install.sh | sh

         # Pull the model to use
         ollama pull llama3

      Then, use the model in the same way as the cloud-based LLMs.

      .. code-block:: python

         import adalflow as adal

         llama_llm = adal.Generator(
            model_client=adal.OllamaClient(), model_kwargs={"model": "llama3"}
         )
         resopnse = llama_llm(prompt_kwargs={"input_str": "What is LLM?"})

   .. tab:: Other


      For other providers, check the :ref:`All Integrations <get_started-integrations>` page.



Try AdalFlow experience end to end in 15 minutes with the `Colab Notebook <https://colab.research.google.com/drive/1_YnD4HshzPRARvishoU4IA-qQuX9jHrT?usp=sharing>`__.


Build your LLM workflow with full control over Prompt, Model, and Output Data Parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Prompt` is the new programming language. All LLM app patterns, from RAG to agents, are implemented via subprompts. AdalFlow leverages `jinja2` template engine to help developers define the overall prompt structure for various applications.
* `DataClass` helps developers define the input and output data structure for the LLM pipeline.
* `Component` is where we define the LLM workflow, which supports both train and eval modes via `forward` and `call` methods.



.. tabs::

   .. tab:: Question Answering

      With `template`, you know exactly what is passed to the LLM. You also have full control over the output parser by defining it yourself.
      `system_prompt` is claimed as a `adal.Parameter` to help training. You can also directly pass `string`.

      .. code-block:: python

         template = r"""<START_OF_SYSTEM_PROMPT>
         {{system_prompt}}
         <END_OF_SYSTEM_PROMPT>
         <START_OF_USER>
         {{input_str}}
         <END_OF_USER>"""

         @adal.func_to_data_component
         def parse_integer_answer(answer: str):
               numbers = re.findall(r"\d+", answer)
               return int(numbers[-1])

         class ObjectCountTaskPipeline(adal.Component):
            def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
               super().__init__()
               system_prompt = adal.Parameter(
                     data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
                     role_desc="To give task instruction to the language model in the system prompt",
                     requires_opt=True,
                     param_type=ParameterType.PROMPT,
               )
               self.llm_counter = adal.Generator(
                     model_client=model_client,
                     model_kwargs=model_kwargs,
                     template=template,
                     prompt_kwargs={
                        "system_prompt": system_prompt,
                     },
                     output_processors=parse_integer_answer,
               )

            def bicall(
               self, question: str, id: str = None
            ) -> Union[adal.GeneratorOutput, adal.Parameter]:
               output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
               return output

      Running the workflow:

      .. code-block:: python

         object_count_pipeline = ObjectCountTaskPipeline(**model_config)

         question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"
         response = object_count_pipeline(question)

      Check out the :ref:`Full Tutorial <question_answering>` for more details.


   .. tab:: Classification


      We use `jinja2` to programmatically formulate our classification description. We use `DataClassParser` for structured data output.


      .. code-block:: python

         template = r"""<START_OF_SYSTEM_PROMPT>;
         {{system_prompt}}
         {% if output_format_str is not none %}
         {{output_format_str}}
         {% endif %}
         <END_OF_SYSTEM_PROMPT>
         <START_OF_USER>
         {{input_str}}
         <END_OF_USER>"""

         task_desc_template = r"""You are a classifier. Given a question, you need to classify it into one of the following classes:
         Format: class_index. class_name, class_description
         {% if classes %}
         {% for class in classes %}
         {{loop.index-1}}. {{class.label}}, {{class.desc}}
         {% endfor %}
         {% endif %}
         - Do not try to answer the question."""

         @dataclass
         class TRECExtendedData(adal.DataClass):
            question: str = field(
               metadata={"desc": "The question to be classified"}, default=None)
            rationale: str = field(
               metadata={
                     "desc": "Your step-by-step reasoning to classify the question to class_name"
               }, default=None)
            class_name: Literal["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"] = field(
               metadata={"desc": "The class name"}, default=None)

            __input_fields__ = ["question"]
            __output_fields__ = ["rationale", "class_name"]


         class TRECClassifierStructuredOutput(adal.Component):
            def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
               super().__init__()
               label_desc = [
                     {"label": label, "desc": desc}
                     for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
               ]
               task_desc_str = adal.Prompt(
                     template=task_desc_template, prompt_kwargs={"classes": label_desc}
               )()
               parser = adal.DataClassParser(
                     data_class=TRECExtendedData, return_data_class=True, format_type="yaml"
               )
               prompt_kwargs = {
                     "system_prompt": adal.Parameter(
                        data=task_desc_str,
                        role_desc="Task description",
                        requires_opt=True,
                        param_type=adal.ParameterType.PROMPT,
                     ),
                     "output_format_str": parser.get_output_format_str(),
               }
               self.llm = adal.Generator(
                     model_client=model_client,
                     model_kwargs=model_kwargs,
                     prompt_kwargs=prompt_kwargs,
                     template=template,
                     output_processors=self.parser,
               )

            def bicall(
               self, question: str, id: Optional[str] = None
            ) -> Union[adal.GeneratorOutput, adal.Parameter]:
               output = self.llm(prompt_kwargs={"input_str": question}, id=id)
               return output

      Check out the :ref:`Full Tutorial <classification_end_to_end>` for more details.

   .. tab:: RAG


      RAG consists of two parts: (1) A data pipeline run off-line to prepare the database with indexces for search and (2) The RAG component that receives a query, retrieves and responds.

      .. code-block:: python

          # Part 1: Data Pipeline using AdalFlow DataComponents
         def prepare_data_pipeline():
            splitter = TextSplitter(**configs["text_splitter"])
            embedder = adal.Embedder(
               model_client=configs["embedder"]["model_client"](),
               model_kwargs=configs["embedder"]["model_kwargs"],
            )
            embedder_transformer = ToEmbeddings(
               embedder=embedder, batch_size=configs["embedder"]["batch_size"]
            )
            data_transformer = adal.Sequential(
               splitter, embedder_transformer
            )  # sequential will chain together splitter and embedder
            return data_transformer

         def transform_documents_and_save_to_db(
            documents: List[Document], db_path: str
         ) -> adal.LocalDB:
            data_transformer = prepare_data_pipeline()

            db = LocalDB()
            db.register_transformer(transformer=data_transformer, key="split_and_embed")
            db.load(documents)
            db.transform(key="split_and_embed")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db.save_state(filepath=db_path)
            return db

         # Part 2: RAG Component using AdalFlow Components

         class RAG(adal.Component):
            def __init__(self):
               super().__init__()
               self.embedder = adal.Embedder(
                     model_client=configs["embedder"]["model_client"](),
                     model_kwargs=configs["embedder"]["model_kwargs"],
               )
               self.initialize_db_manager()
               data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)
               self.generator = adal.Generator(
                     template=RAG_TEMPLATE,
                     prompt_kwargs={
                        "output_format_str": data_parser.get_output_format_str(),
                        "system_prompt": system_prompt,
                     },
                     model_client=configs["generator"]["model_client"](),
                     model_kwargs=configs["generator"]["model_kwargs"],
                     output_processors=data_parser,
               )

            def initialize_db_manager(self):
               self.db_manager = DatabaseManager()
               self.transformed_docs = []

            def prepare_retriever(self, repo_url_or_path: str):
               self.initialize_db_manager()
               self.transformed_docs = self.db_manager.prepare_database(repo_url_or_path)
               self.retriever = FAISSRetriever(
                     **configs["retriever"],
                     embedder=self.embedder,
                     documents=self.transformed_docs,
               )

            def call(self, query: str) -> Any:
               retrieved_documents = self.retriever(query)
               prompt_kwargs = {
                     "input_str": query,
                     "contexts": retrieved_documents[0].documents,
               }
               response = self.generator(
                     prompt_kwargs=prompt_kwargs,
               )
               return response.data, retrieved_documents

      Check out a real-world RAG project `GithubChat <https://github.com/SylphAI-Inc/GithubChat>`__ or this `colab notebook <https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/use_cases/adalflow_rag_optimization.ipynb>`__.

   .. tab:: Agent

      We can use both another `component`(trainable) or another function as a tool. Here is an example of an AgenticRAG.

      .. code-block:: python

         class AgenticRAG(adal.Component):
            def __init__(self, model_client, model_kwargs):
               super().__init__()
               self.dspy_retriever = DspyRetriever(top_k=2)

               def dspy_retriever_as_tool(
                     input: str,
                     id: Optional[str] = None,
               ) -> List[str]:
                     r"""Retrieves the top 2 passages from using input as the query.
                     Ensure you get all the context to answer the original question.
                     """
                     output = self.dspy_retriever(input=input, id=id)
                     parsed_output = output
                     if isinstance(output, adal.Parameter):
                        parsed_output = output.data.documents
                        return parsed_output
                     documents = parsed_output.documents
                     return documents

               tools = [
                     FunctionTool(dspy_retriever_as_tool, component=self.dspy_retriever),
               ]

               self.agent = ReActAgent(
                     max_steps=3,
                     add_llm_as_fallback=False,
                     tools=tools,
                     model_client=model_client,
                     model_kwargs=model_kwargs,
               )

            def bicall(self, input: str, id: str = None) -> str:
               out = self.agent(input=input, id=id)
               return out

      Check out the `source code <https://github.com/SylphAI-Inc/AdalFlow/blob/main/benchmarks/hotpot_qa/adal_exp/build_multi_hop_rag.py>`__.



Auto-optimize your LLM workflow with both Prompt Tuning and Few-shot Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* When a `Parameter` is defined as `PROMPT`, AdalFlow will automatically optimize the prompt on your training dataset via `LLM-AutoDiff <https://arxiv.org/abs/2501.16673>`__.
* When a `Parameter` is defined as `DEMOS`, `Few-Shot Bootstrap Learning <https://arxiv.org/abs/2310.03714>`__ is applied.
* When both are defined, AdalFlow will use both to find the best performing prompts.


.. tabs::

   .. tab:: AdalComponent

      The `Trainer` requires the following key elements from your workflow:

      1. **Task Workflow** – Defined as a `Component` from the previous step.
      2. **Model Configuration** – Includes settings for the optimizer and bootstrap teacher. It is recommended to use high-performing LLM models such as `4o`, `o1`, `o3-mini`, or `r1`.
      3. **Evaluation Metrics** – The criteria used to assess model performance.
      4. **LLM Workflow Execution** – Instructions on how to invoke your LLM workflow in both evaluation and training modes.


      Developers can organize the above information in `AdalComponent`, similar to how `PyTorch LightningModule` is structured.


      .. code-block:: python

         class TrecClassifierAdal(adal.AdalComponent):
            def __init__(
               self,
               model_client: adal.ModelClient,
               model_kwargs: Dict,
               teacher_model_config: Dict,
               backward_engine_model_config: Dict,
               text_optimizer_model_config: Dict,
            ):
               task = TRECClassifierStructuredOutput(model_client, model_kwargs)
               eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
               loss_fn = adal.EvalFnToTextLoss(
                     eval_fn=eval_fn,
                     eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0. When the LLM prediction failed with format parsing which results with errors, we set y_pred = -1",
               )
               super().__init__(
                     task=task,
                     eval_fn=eval_fn,
                     loss_fn=loss_fn,
                     backward_engine_model_config=backward_engine_model_config,
                     text_optimizer_model_config=text_optimizer_model_config,
                     teacher_model_config=teacher_model_config,
               )

            def prepare_task(self, sample: TRECExtendedData):
               return self.task.bicall, {"question": sample.question, "id": sample.id}

            def prepare_eval(
               self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
            ) -> float:
               y_label = -1
               if y_pred and y_pred.data is not None and y_pred.data.class_name is not None:
                     y_label = y_pred.data.class_name
               return self.eval_fn, {"y": y_label, "y_gt": sample.class_name}

            def prepare_loss(
               self, sample: TRECExtendedData, y_pred: adal.Parameter, *args, **kwargs
            ) -> Tuple[Callable[..., Any], Dict]:
               full_response = y_pred.data
               y_label = -1
               if (full_response and full_response.data is not None
                     and full_response.data.class_name is not None):
                     y_label = full_response.data.class_name

               y_pred.eval_input = y_label
               y_gt = adal.Parameter(
                     name="y_gt",
                     data=sample.class_name,
                     eval_input=sample.class_name,
                     requires_opt=False,
               )
               return self.loss_fn, {
                     "kwargs": {"y": y_pred, "y_gt": y_gt},
                     "id": sample.id,
               }

      Check out the :ref:`Full Tutorial <question_answering>` for more details.

   .. tab:: Load Data and Train

      Trainer takes `AdalComponent` and datasets as input. It trains the prompts and returns the best checkpoint.

      .. code-block:: python

         def load_datasets():
            train_data = TrecDataset(split="train")
            val_data = TrecDataset(split="val")ßß
            test_data = TrecDataset(split="test")
            return train_data, val_data, test_data

         adal_component = TrecClassifierAdal(
               model_client=model_client,
               model_kwargs=model_kwargs,
               text_optimizer_model_config=deepseek_r1_model,
               backward_engine_model_config=gpt_4o_model,
               teacher_model_config=gpt_4o_model,
         )
         trainer = adal.Trainer(
               adaltask=adal_component,
               max_steps=12,
               raw_shots=1,
               bootstrap_shots=1,
         )

         train_dataset, val_dataset, test_dataset = load_datasets()
         ckpt, _ = trainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
         )

      :ref:`Full Tutorial here <question_answering>`.





.. .. raw:: html

..     <h3 style="text-align: left; font-size: 1.5em; margin-top: 50px;">
..     Light, Modular, and Model-agnositc Task Pipeline

..     </h3>

.. Light, Modular, and Model-agnositc Task Pipeline
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. LLMs are like water; AdalFlow help developers quickly shape them into any applications, from GenAI applications such as chatbots, translation, summarization, code generation, RAG, and autonomous agents to classical NLP tasks like text classification and named entity recognition.


.. Only two fundamental but powerful base classes: `Component` for the pipeline and `DataClass` for data interaction with LLMs.
.. The result is a library with bare minimum abstraction, providing developers with *maximum customizability*.

.. You have full control over the prompt template, the model you use, and the output parsing for your task pipeline.


.. .. figure:: /_static/images/AdalFlow_task_pipeline.png
..    :alt: AdalFlow Task Pipeline
..    :align: center



.. .. raw:: html

..     <h3 style="text-align: left; font-size: 1.5em; margin-top: 10px;">
..     Unified Framework for Auto-Optimization
..     </h3>


.. AdalFlow provides token-efficient and high-performing prompt optimization within a unified framework.
.. To optimize your pipeline, simply define a ``Parameter`` and pass it to our ``Generator``.
.. Whether you need to optimize task instructions or few-shot demonstrations,
.. our unified framework offers an easy way to **diagnose**, **visualize**, **debug**, and **train** your pipeline.

.. This trace graph demonstrates how our auto-differentiation works: :doc:`trace_graph <../tutorials/trace_graph>`

.. **Trainable Task Pipeline**

.. Just define it as a ``Parameter`` and pass it to our ``Generator``.


.. .. figure:: /_static/images/Trainable_task_pipeline.png
..    :alt: AdalFlow Trainable Task Pipeline
..    :align: center


.. **AdalComponent & Trainer**

.. ``AdalComponent`` acts as the `interpreter`  between task pipeline and the trainer, defining training and validation steps, optimizers, evaluators, loss functions, backward engine for textual gradients or tracing the demonstrations, the teacher generator.


.. .. figure:: /_static/images/trainer.png
..    :alt: AdalFlow AdalComponent & Trainer
..    :align: center




Unites Research and Production
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Our team has experience in both AI research and production.
We are building a library that unites the two worlds, forming a healthy LLM application ecosystem.

- To resemble the PyTorch library makes it easier for LLM researchers to use the library.
- Researchers building on AdalFlow enable production engineers to easily adopt, test, and iterate on their production data.
- Our 100% control and clarity of the source code further make it easy for product teams to build on and for researchers to extend their new methods.


Community
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Learn, share and collaborate with the AdalFlow AI community*

.. You can join our community on various platforms:

.. * `Discord <https://discord.com/invite/ezzszrRZvT>`_
.. * `GitHub Discussion <https://github.com>`_

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px;">
      <a href="https://github.com/SylphAI-Inc/AdalFlow/discussions" style="display: inline-block; margin-left: 10px;">
         <img src="https://img.shields.io/badge/GitHub Discussions-AdalFlow-blue?logo=github&style=flat-square" alt="GitHub Repo">
      </a>
        <a href="https://github.com/SylphAI-Inc/AdalFlow/issues" style="display: inline-block; margin-left: 10px;">
         <img src="https://img.shields.io/badge/Bugs/Feature Requests-AdalFlow-blue?logo=github&style=flat-square" alt="GitHub Repo">
      </a>
      <a href="https://discord.gg/ezzszrRZvT" style="display: inline-block; margin-left: 10px;">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/ezzszrRZvT?style=flat">
      </a>

   </div>



.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   integrations/index

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   new_tutorials/index
   .. :caption: Tutorials - How each part works
   .. :hidden:


.. .. Hide the use cases for now
.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :hidden:

   use_cases/index


      .. :caption: Benchmarks

      .. Manually add documents for the code in benchmarks


..    :glob:
..    :maxdepth: 1
..    :caption: Resources

..    resources/index

.. hide the for contributors now

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: For Contributors
   :hidden:

   contributor/index


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   tutorials/index



.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   apis/index
