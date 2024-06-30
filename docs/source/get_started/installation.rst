Installation
============

LightRAG is available in Python.
To install the package, run:

.. code-block:: bash

   pip install lightrag



1. **Set up API keys via Environment Variables**

``.env`` file is recommended.
You can have it at your project root directory.
Here are an example:

.. code-block:: bash

    OPENAI_API_KEY=YOUR_API_KEY_IF_YOU_USE_OPENAI
    GROQ_API_KEY=YOUR_API_KEY_IF_YOU_USE_GROQ
    ANTHROPIC_API_KEY=YOUR_API_KEY_IF_YOU_USE_ANTHROPIC
    GOOGLE_API_KEY=YOUR_API_KEY_IF_YOU_USE_GOOGLE
    COHERE_API_KEY=YOUR_API_KEY_IF_YOU_USE_COHERE
    HF_TOKEN=YOUR_API_KEY_IF_YOU_USE_HF


2. **Load Environment Variables**


You can add the following import:

.. code-block:: python

   from lightrag.utils import setup_env #noqa

Or, you can load it yourself:

.. code-block:: python

   from dotenv import load_dotenv
   load_dotenv()  # This loads the environment variables from `.env`.

This setup ensures that LightRAG can access all necessary configurations during runtime.

3. **Install Optional Packages**

LightRAG currently has built-in support for (1) OpenAI, Groq, Anthropic, Google, and Cohere, (2) FAISS and Transformers.
You can find all optional packages at :class:`utils.lazy_import.OptionalPackages`.
Make sure to install the necessary SDKs for the components you plan to use.
Here is the list of our tested versions:

.. code-block::

    openai = "^1.12.0"
    groq = "^0.5.0"
    faiss-cpu = "^1.8.0"
    sqlalchemy = "^2.0.30"
    cohere = "^5.5.8"
    pgvector = "^0.2.5"
    anthropic = "^0.26.0"
    google-generativeai = "^0.5.4"






.. Poetry Installation
.. --------------------------

.. Developers and contributors who need access to the source code or wish to contribute to the project should set up their environment as follows:

.. 1. **Clone the Repository:**

..    Start by cloning the LightRAG repository to your local machine:

..    .. code-block:: bash

..       git clone https://github.com/SylphAI-Inc/LightRAG
..       cd LightRAG

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
