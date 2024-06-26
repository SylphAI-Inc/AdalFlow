Installation
============

LightRAG can be installed either as a package using pip or set up for development by cloning from GitHub. Follow the appropriate instructions below based on your needs.

Pip Installation
--------------------------------

For general users who simply want to use LightRAG, the easiest method is to install it directly via pip:

.. code-block:: bash

   pip install lightrag

After installing the package, you need to set up your environment variables for the project to function properly:

1. **Create an Environment File:**

   Create a `.env` file in your project directory (where your scripts using LightRAG will run):

   .. code-block:: bash

      touch .env
      # Open .env and add necessary configurations such as API keys

2. **Configure Your `.env` File:**

   Add the necessary API keys and other configurations required by LightRAG. This usually includes setting up credentials for accessing various APIs that LightRAG interacts with.

3. **Load Environment Variables:**

   Make sure your application or scripts load the environment variables from the `.env` file at runtime. If you are using Python, libraries like `python-dotenv` can be used:

   .. code-block:: bash

      pip install python-dotenv

Then, in your Python script, ensure you load the variables:

.. code-block:: python

   from dotenv import load_dotenv
   load_dotenv()  # This loads the environment variables from `.env`.

This setup ensures that LightRAG can access all necessary configurations during runtime.


Poetry Installation
--------------------------

Developers and contributors who need access to the source code or wish to contribute to the project should set up their environment as follows:

1. **Clone the Repository:**

   Start by cloning the LightRAG repository to your local machine:

   .. code-block:: bash

      git clone https://github.com/SylphAI-Inc/LightRAG
      cd LightRAG

2. **Configure API Keys:**

   Copy the example environment file and add your API keys:

   .. code-block:: bash

      cp .env.example .env
      # Open .env and fill in your API keys

3. **Install Dependencies:**

   Use Poetry to install the dependencies and set up the virtual environment:

   .. code-block:: bash

      poetry install
      poetry shell

4. **Verification:**

   Now, you should be able to run any file within the repository or execute tests to confirm everything is set up correctly.