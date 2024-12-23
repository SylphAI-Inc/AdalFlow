.. _get_started-integrations:

Integrations
===========

AdalFlow integrates with many popular AI and database platforms to provide a comprehensive solution for your LLM applications.

Model Providers
-------------

.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://platform.openai.com/" target="_blank">
            <img src="../_static/logos/openai.png" alt="OpenAI">
            <span>OpenAI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://console.anthropic.com/" target="_blank">
            <img src="../_static/logos/anthropic.png" alt="Anthropic">
            <span>Anthropic</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://mistral.ai/" target="_blank">
            <img src="../_static/logos/mistral.png" alt="Mistral AI">
            <span>Mistral AI</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://aws.amazon.com/bedrock/" target="_blank">
            <img src="../_static/logos/aws-bedrock.png" alt="Amazon Bedrock">
            <span>Amazon Bedrock</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://groq.com/" target="_blank">
            <img src="../_static/logos/groq.png" alt="Groq">
            <span>Groq</span>
         </a>
      </div>
   </div>

Vector Databases
--------------

.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://qdrant.tech/" target="_blank">
            <img src="../_static/logos/qdrant.png" alt="Qdrant">
            <span>Qdrant</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://lancedb.com/" target="_blank">
            <img src="../_static/logos/lancedb.png" alt="LanceDB">
            <span>LanceDB</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://www.pinecone.io/" target="_blank">
            <img src="../_static/logos/pinecone.png" alt="Pinecone">
            <span>Pinecone</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://www.milvus.io/" target="_blank">
            <img src="../_static/logos/milvus.png" alt="Milvus">
            <span>Milvus</span>
         </a>
      </div>
   </div>

Embedding Models
--------------

.. raw:: html

   <div class="integration-grid">
      <div class="integration-item">
         <a href="https://huggingface.co/" target="_blank">
            <img src="../_static/logos/huggingface.png" alt="Hugging Face">
            <span>Hugging Face</span>
         </a>
      </div>
      <div class="integration-item">
         <a href="https://platform.openai.com/docs/guides/embeddings" target="_blank">
            <img src="../_static/logos/openai.png" alt="OpenAI Embeddings">
            <span>OpenAI Embeddings</span>
         </a>
      </div>
   </div>

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

Quick Start
----------

To use any of these integrations, first install AdalFlow with the appropriate extras:

.. code-block:: bash

   # For model providers
   pip install "adalflow[openai,anthropic,mistral,bedrock,groq]"

   # For vector databases
   pip install "adalflow[qdrant,lancedb]"

See the :ref:`installation guide <get_started-installation>` for more details.

Usage Examples
------------

Check out our tutorials for detailed examples of using these integrations:

- :ref:`Model Clients <tutorials-model_client>`
- :ref:`Vector Databases <tutorials-database>`
- :ref:`Embeddings <tutorials-embedder>`
