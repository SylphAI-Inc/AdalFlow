Install AdalFlow and Run your LM
=========================


.. _Installation:


.. raw:: html

    <div class="setup-container">
        <!-- Header -->
        <div class="info-header">
            <span>ℹ️</span> <strong>Getting Started: Install AdalFlow and set up your LM</strong>
        </div>

        <!-- Installation Box -->
        <div class="install-box">
            <code> pip install -U adalflow</code>
            <button class="copy-btn" onclick="copyToClipboard()">📋</button>
        </div>

        <!-- Tabs Section -->
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab(event, 'openai')">OpenAI</button>
            <button class="tab-button" onclick="switchTab(event, 'groq')">Groq</button>
            <button class="tab-button" onclick="switchTab(event, 'anthropic')">Anthropic</button>
            <button class="tab-button" onclick="switchTab(event, 'local')">Local LMs</button>
            <button class="tab-button" onclick="switchTab(event, 'other')">Other providers</button>
        </div>

        <!-- Content Sections -->
        <div id="openai" class="tab-content active">
            <p>Setup `OPENAI_API_KEY` in your `.env` file or pass the `api_key` to the client.</p>
            <div class="code-box">
                <button class="copy-btn" onclick="copyCode('code-openai')">📋</button>
                <pre><code id="code-openai">
   import adalflow as adal

   # setup env or pass the api_key to client
   from adalflow.utils import setup_env

   setup_env()

   openai_llm = adal.Generator(
      model_client=adal.OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
   )
   resopnse = openai_llm(prompt_kwargs={"input_str": "What is LLM?"})
                </code></pre>
            </div>
        </div>

        <div id="groq" class="tab-content">
            <p>Setup `GROQ_API_KEY` in your `.env` file or pass the `api_key` to the client.</p>

            <div class="code-box">
                <button class="copy-btn" onclick="copyCode('code-groq')">📋</button>
                <pre><code id="code-groq">
   import adalflow as adal

   # setup env or pass the api_key to client
   from adalflow.utils import setup_env

   setup_env()

   llama_llm = adal.Generator(
      model_client=adal.GroqAPIClient(), model_kwargs={"model": "llama3-8b-8192"}
   )
   resopnse = llama_llm(prompt_kwargs={"input_str": "What is LLM?"})


                </code></pre>
            </div>
        </div>

        <div id="anthropic" class="tab-content">
            <p>Setup `ANTHROPIC_API_KEY` in your `.env` file or pass the `api_key` to the client.</p>
            <div class="code-box">
                <button class="copy-btn" onclick="copyCode('code-anthropic')">📋</button>
                <pre><code id="code-anthropic">
   import adalflow as adal

   # setup env or pass the api_key to client
   from adalflow.utils import setup_env

   setup_env()

   anthropic_llm = adal.Generator(
      model_client=adal.AnthropicAPIClient(), model_kwargs={"model": "claude-3-opus-20240229"}
   )
   resopnse = anthropic_llm(prompt_kwargs={"input_str": "What is LLM?"})

                </code></pre>
            </div>
        </div>

        <div id="local" class="tab-content">
            <p>Ollama is one option. You can also use `vllm` or HuggingFace `transformers`.</p>
            <div class="code-box">
                <button class="copy-btn" onclick="copyCode('code-local')">📋</button>
                <pre><code id="code-local">
   # Download Ollama command line tool
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the model to use
   ollama pull llama3
                </code></pre>
            </div>
            <p>Use it in the same way as other providers.</p>
            <div class="code-box">
                <button class="copy-btn" onclick="copyCode('code-local-use')">📋</button>
                <pre><code id="code-local-use">
   import adalflow as adal

   llama_llm = adal.Generator(
      model_client=adal.OllamaClient(), model_kwargs={"model": "llama3"}
   )
   resopnse = llama_llm(prompt_kwargs={"input_str": "What is LLM?"})
                </code></pre>
            </div>
        </div>



        <div id="other" class="tab-content">
      <p>For other providers, check the <a href="https://adalflow.sylph.ai/integrations/integrations.html" target="_blank" class="doc-link">official documentation</a>.</p>
      </div>
    </div>

    <!-- JavaScript for Tab Functionality -->
    <script>
        function switchTab(event, tabName) {
            let i, tabcontent, tabbuttons;

            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }

            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].classList.remove("active");
            }

            document.getElementById(tabName).classList.add("active");
            event.currentTarget.classList.add("active");
        }

        function copyCode(codeId) {
            let codeText = document.getElementById(codeId).innerText;
            navigator.clipboard.writeText(codeText);
            alert("Code copied to clipboard!");
        }

        function copyToClipboard() {
            navigator.clipboard.writeText("pip install -U adalflow");
            alert("Copied to clipboard!");
        }
    </script>

    <!-- CSS Styling -->
    <style>
        .setup-container {
            border: 2px solid #D0E2F2;
            padding: 20px;
            border-radius: 10px;
            background-color: #F8FCFF;
            font-family: Arial, sans-serif;
        }

        .info-header {
            display: flex;
            align-items: center;
            background: #EAF5FF;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.8em;
            color: #0078D7;
        }

        .install-box {
            background: #F4F4F4;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .copy-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            margin-left: 10px;
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid #DDD;
            margin-top: 20px;
        }

        .tab-button {
            flex-grow: 1;
            background: none;
            border: none;
            padding: 10px;
            font-size: 1em;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }

        .tab-button.active {
            font-weight: bold;
            border-bottom: 3px solid #0078D7;
        }

        .tab-content {
            display: none;
            padding: 20px;
            background: #FAFAFA;
            border-radius: 5px;
            margin-top: 10px;
        }

        .tab-content.active {
            display: block;
        }

        .code-box {
            position: relative;
            background: #F4F4F4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: "Courier New", monospace;
        }

        .code-box pre {
            margin: 0;
        }

        .code-box code {
            font-size: 0.95em;
            display: block;
            white-space: pre-wrap;
        }

        .code-box .copy-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 1em;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 5px;
        }

        .doc-link {
    color: #0078D7;  /* Matches the UI theme */
    text-decoration: none;
    font-weight: bold;
      }

      .doc-link:hover {
         text-decoration: underline;
      }

        @media (max-width: 768px) {
            .setup-container {
                padding: 15px;
            }
            .code-box {
                font-size: 0.85em;
            }
        }
    </style>







.. Or, you can load it yourself with ``python-dotenv``:

.. .. code-block:: python

..    from dotenv import load_dotenv
..    load_dotenv()  # This loads the environment variables from `.env`.

.. This setup ensures that AdalFlow can access all necessary configurations during runtime.

.. 4. Install Optional Packages
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. AdalFlow currently has built-in support for (1) OpenAI, Groq, Anthropic, Google, and Cohere, and (2) FAISS and Transformers.
.. You can find all optional packages at :class:`OptionalPackages<utils.lazy_import.OptionalPackages>`.
.. Make sure to install the necessary SDKs for the components you plan to use.
.. Here is the list of our tested versions:



..    openai = "^1.12.0"
..    groq = "^0.5.0"
..    faiss-cpu = "^1.8.0"
..    sqlalchemy = "^2.0.30"
..    pgvector = "^0.3.1"
..    torch = "^2.3.1"
..    anthropic = "^0.31.1"
..    google-generativeai = "^0.7.2"
..    cohere = "^5.5.8"

.. You can install the optional packages with either ``pip install package_name`` or ``pip install adalflow[package_name]``.






.. Poetry Installation
.. --------------------------

.. Developers and contributors who need access to the source code or wish to contribute to the project should set up their environment as follows:

.. 1. **Clone the Repository:**

..    Start by cloning the AdalFlow repository to your local machine:

..    .. code-block:: bash

..       git clone https://github.com/SylphAI-Inc/AdalFlow
..       cd AdalFlow

.. 2. **Configure API Keys:**

..    Copy the example environment file and add your API keys:

..    .. code-block:: bash

..       cp .env.example .env
..       # Open .env and fill in your API keys

.. 3. **Install Dependencies:**

..    Use Poetry to install the dependencies and set up the virtual environment:

..    .. code-block:: bash

..       poetry install
..       poetry shell

.. 4. **Verification:**

..    Now, you should be able to run any file within the repository or execute tests to confirm everything is set up correctly.
