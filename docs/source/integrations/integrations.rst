.. _get_started-integrations:

All Providers
==================

AdalFlow integrates with many popular AI and database platforms to provide a comprehensive solution for your LM applications.

Model Providers
-------------
AdalFlow supports a wide range of model providers, each offering unique capabilities and models:

.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.deepseek_client.html#module-components.model_client.deepseek_client" target="_blank">
            <img src="../_static/images/deepseek.png" alt="Deepseek">
            <span>Deepseek</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.anthropic_client.html#module-components.model_client.anthropic_client" target="_blank">
            <img src="../_static/images/anthropic.png" alt="Anthropic">
            <span>Anthropic</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://mistral.ai/" target="_blank">
            <img src="../_static/images/mistral.png" alt="Mistral">
            <span>Mistral</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.ollama_client.html#module-components.model_client.ollama_client" target="_blank">
            <img src="../_static/images/ollama.png" alt="Ollama">
            <span>Ollama</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.openai_client.html#module-components.model_client.openai_client" target="_blank">
            <img src="../_static/images/openai.png" alt="OpenAI">
            <span>OpenAI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.transformers_client.html#module-components.model_client.transformers_client" target="_blank">
            <img src="../_static/images/huggingface.png" alt="Transformers">
            <span>Transformers</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.groq_client.html#module-components.model_client.groq_client" target="_blank">
            <img src="../_static/images/groq.png" alt="Groq">
            <span>Groq</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.azureai_client.html#module-components.model_client.azureai_client" target="_blank">
            <img src="../_static/images/azure.png" alt="Azure OpenAI">
            <span>Azure OpenAI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.bedrock_client.html" target="_blank">
            <img src="../_static/images/bedrock.png" alt="Amazon Bedrock">
            <span>Amazon Bedrock</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.together_client.html" target="_blank">
            <img src="../_static/images/together.png" alt="Together AI">
            <span>Together AI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.xai_client.html" target="_blank">
            <img src="../_static/images/xai.png" alt="xAI">
            <span>xAI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.fireworks_client.html" target="_blank">
            <img src="../_static/images/fireworks.png" alt="Fireworks">
            <span>Fireworks</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.sambanova_client.html" target="_blank">
            <img src="../_static/images/sambanova.png" alt="Sambanova">
            <span>Sambanova</span>
         </a>
      </div>
   </div>

.. list-table:: LLM + VLLM
   :widths: 25 55 20
   :header-rows: 1

   * - **Major Class**
     - **Description**
     - **Tutorial**
   * - :class:`Generator <core.generator.Generator>`
     - A user-facing orchestration component that handles LLM predictions. It includes a prompt template, model client, and output parser.
     - :ref:`Generator <generator>`
   * - :class:`ReActAgent <components.agent.react.ReActAgent>`
     - An agent that uses large language model reasoning (Re) and actions (Act) to solve queries.
     - :ref:`Agent <tutorials-agent>`
   * - :class:`ModelClient <core.model_client.ModelClient>`
     - The low-level component managing the actual calls to a chosen LLM (OpenAI, Anthropic, VLLM, etc.).
     - :ref:`ModelClient <tutorials-model_client>`

Vector Databases
--------------
.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.retriever.qdrant_retriever.html#module-components.retriever.qdrant_retriever" target="_blank">
            <img src="../_static/images/qdrant.png" alt="Qdrant">
            <span>Qdrant</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.retriever.lancedb_retriver.html#module-components.retriever.lancedb_retriver" target="_blank">
            <img src="../_static/images/lancedb.png" alt="LanceDB">
            <span>LanceDB</span>
         </a>
      </div>
   </div>



Embedding and Reranking Models
---------------------------
.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.transformers_client.html#module-components.model_client.transformers_client" target="_blank">
            <img src="../_static/images/huggingface.png" alt="Hugging Face">
            <span>Hugging Face</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.openai_client.html#module-components.model_client.openai_client" target="_blank">
            <img src="../_static/images/openai.png" alt="OpenAI">
            <span>OpenAI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://adalflow.sylph.ai/apis/components/components.model_client.cohere_client.html#module-components.model_client.cohere_client" target="_blank">
            <img src="../_static/images/cohere.png" alt="Cohere">
            <span>Cohere</span>
         </a>
      </div>
   </div>


.. list-table:: Embeddings, Reranking, and Vector Databases
   :widths: 25 55 20
   :header-rows: 1

   * - **Major Class**
     - **Description**
     - **Tutorial**
   * - :class:`Embedder <core.embedder.Embedder>`
     - A user-facing component that orchestrates embedding models via ``ModelClient`` and ``output_processors``
     - :ref:`Embedder <tutorials-embedder>`
   * - :class:`Retriever <core.retriever.Retriever>`
     - Each subclass can be a local, a vector-db, a retranker, or an LLM-turned retriever to handle retrieval tasks in RAG.
     - :ref:`Retriever <tutorials-retriever>`
   * - :class:`TextSplitter <components.data_process.TextSplitter>`
     - Chunking large text into smaller segments for more efficient and accurate embedding, retrieval, and LLM context processing.
     - :ref:`TextSplitter <tutorials-text_splitter>`


.. raw:: html

   <style>
      .integration-grid {
         display: grid;
         grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
         gap: 2rem;
         margin: 2rem 0;
      }
      .integration-item {
         text-align: center;
         padding: 1rem;
         border: 1px solid #eee;
         border-radius: 8px;
         transition: transform 0.2s, box-shadow 0.2s;
      }
      .integration-item:hover {
         transform: translateY(-5px);
         box-shadow: 0 5px 15px rgba(0,0,0,0.1);
      }
      .integration-item img {
         max-width: 100px;
         height: auto;
         margin-bottom: 1rem;
      }
      .integration-item a {
         text-decoration: none;
         color: inherit;
         display: flex;
         flex-direction: column;
         align-items: center;
      }
      .integration-item span {
         font-weight: 500;
      }
   </style>
