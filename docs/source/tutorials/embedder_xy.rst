.. _embedder_xy:

Embedder
============

A commonly applied approach in the NLP field to transform texts to numeric representation is Embedding. Embedding turns the texts data to the vector space.
This enables us to calculate numbers such as similarity score. With embedding, during retrieval, we can do semantic search to find similar documents.

In ``LightRAG``, text embedding is managed through three main components: ``Embedder``, ``ToEmbedderResponse``, and ``ToEmbeddings``.
These components work together to transform text into embeddings using external model APIs from providers like `OpenAI`` and `Google`.

* **Embedder:** It acts as the interface for external embedding models. It orchestrates embedding models via ``ModeClient`` and ``output_processors`` that can be used to process output from ``ModeClient``. Ensure that your model's API key is stored in the ``.env`` file. This setup prepares the Embedder to generate embeddings effectively.
.. * **ToEmbedderResponse:** It is designed to convert raw model outputs into structured ``EmbedderResponse`` formats, typically lists of float values representing embeddings. It should be used as the ``output_processors`` in the ``Embedder`` setup.
* **ToEmbeddings:** It takes documents from ``Embedder``, processes documents in batches and uses an instance of ``Embedder`` as the ``vectorizer`` to obtain embeddings. After receiving the embeddings, ``ToEmbeddings`` assigns them back to the respective Document objects. This component ensures that the input data is copied and the original data remains unmodified.

With these components, the embeddings will be created when you run ``ToEmbeddings``.

Next, we will use an example combining the document splitter and embedder to demonstrate the embedder usage.

First, import the packages and set up the environment.

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.data_components import ToEmbedderResponse, ToEmbeddings
    from lightrag.core.types import Document
    from lightrag.components.data_process import DocumentSplitter

    import dotenv
    dotenv.load_dotenv(dotenv_path=".env", override=True)

Then, set up the splitter with a simple example.

.. code-block:: python

    # combine splitter and embedding
    # configure the splitter setting
    text_splitter_settings = {
            "split_by": "word",
            "split_length": 15,
            "split_overlap": 2,
            }

    # set up the document splitter
    text_splitter = DocumentSplitter(
        split_by=text_splitter_settings["split_by"],
        split_length=text_splitter_settings["split_length"],
        split_overlap=text_splitter_settings["split_overlap"],
        )

    # create examples
    example1 = Document(
        text="Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: Positive",
    )
    documents = [example1]

    # split the documents
    splitted_docs = (text_splitter.call(documents=documents))
    print(splitted_docs)

    # Document(id=78917586-c4e5-4ce3-ac78-196ec25d39c4, meta_data=None, text=Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: , estimated_num_tokens=None), Document(id=fdbc447d-5ba2-4917-9c14-0d95cc65e6ef, meta_data=None, text=again! Sentiment: Positive, estimated_num_tokens=None)]

Now, create the embedder. ``LightRAG`` uses ``Embedder`` to configure the embedding model, process output. Developers can set up the ``batch_size`` in ``ToEmbeddings``.

.. code-block:: python

    # configure the vectorizer(embedding) setting
    vectorizer_settings = {
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
        "batch_size": 100
    }

    # set up the embedder using openai model
    vectorizer = Embedder(
            model_client=OpenAIClient,
            model_kwargs=vectorizer_settings["model_kwargs"], # set up model arguments
            output_processors=ToEmbedderResponse(), # convert the model output to EmbedderResponse
        )

    # create embedder
    # create embeddings
    to_embeddings = ToEmbeddings(
        embedder=vectorizer,
        batch_size=vectorizer_settings["batch_size"],
        )

Finally, check the results.

.. code-block:: python

    # show the embedding for each splitted doc
    embeddings = to_embeddings(splitted_docs)
    for embedding in embeddings:
        print(f"the splitted doc: {embedding.text}")
        print(f"the embedding of the doc: {embedding.vector}")

    # the splitted doc: Review: What a fantastic movie! Had a great time and would watch it again! Sentiment:
    # the embedding of the doc: [-0.06596588, 0.10054244, -0.19306852, 0.045200635, -0.015742956, 0.017384859, 0.0625372, 0.13164201, 0.020113317, 0.0060756463, 0.056549083, -0.08199859, 0.0073946016, 0.0445487, 0.041820247, 0.046842538, 0.04500747, 0.015549791, 0.11802388, 0.11155285, -0.029578406, 0.08112934, -0.11686489, -0.0076722763, -0.0027390209, -0.024447458, -0.060026057, 0.03940568, 0.06881507, -0.035276778, 0.07113305, -0.022370934, 0.04363117, -0.07881136, -0.05220287, 0.025159756, -0.022491662, -0.07407882, -0.012519513, -0.0354458, 0.059977766, 0.056500793, 0.081225924, -0.006953944, -0.04090271, 0.01965455, -0.010159277, 0.02449575, -0.02094634, 0.034190223, -0.18089913, 0.08962861, -0.028274542, 0.18447268, -0.03841571, -0.030737398, -0.013799232, 0.03575969, 0.047301304, -0.122853, -0.029819863, 0.055824716, 0.055390093, 0.028250396, 0.03648406, -0.074030526, -0.049257103, -0.025304629, 0.0935402, -0.020777322, 0.04964343, 0.03438339, 0.015791247, -0.027646756, 0.06244062, -0.016853655, 0.078569904, -0.05133363, -0.017179621, -0.07456173, -0.115995646, -0.025763396, 0.00016600126, 0.0072255824, 0.039792012, -0.08397853, -0.024821715, 0.07794212, -0.09793471, 0.08277125, 0.13125569, 0.11618881, -0.028419416, -0.013642286, -0.0092658885, 0.0708433, -0.062150873, 0.085717015, 0.014752985, 0.065724425, 0.09793471, 0.0017067948, -0.0780387, -0.00018637415, 0.0841717, 0.07654167, 0.016503545, 0.049112227, 0.052106287, -0.037787925, -0.09213976, -0.020318555, -0.08658626, 0.06978089, -0.012314276, 0.028588437, 0.08494435, -0.09199488, -0.08861449, 0.06437227, 0.00093111617, -0.07185742, -0.010340369, -0.03414193, -0.041868538, -0.0442831, -0.009410762, -0.038294982, -0.015851611, -0.05331357, -0.0009763892, 0.07736263, -0.032958794, -0.033320982, -0.11396741, -0.025811687, -0.054617435, -0.044596992, 0.0003912348, 0.05408623, -0.03288636, -0.10875195, -0.05606617, -0.031992972, 0.13985154, 0.004110795, -0.052057996, 0.06842874, -0.04732545, 0.0098876385, 0.083737075, 0.037280865, -0.14545332, -0.02072903, 0.009845384, -0.046214752, -0.078763075, 0.033200253, -0.034673136, 0.04988489, -0.028709164, 0.01721584, -0.04882248, 0.0017369769, 0.029023057, 0.045755986, -0.048170548, -0.037715487, -0.007086745, -0.04882248, 0.055341803, -0.0048683644, -0.05877048, -0.050126344, 0.007424784, -0.0249183, -0.0016585035, 0.038898624, 0.055631552, 0.0035554452, -0.024773424, -0.0059066266, -0.054762308, 0.01102852, 0.040178344, -0.013376684, -0.016201723, 0.025449503, 0.042810217, 0.100156106, 0.06335816, 0.20417552, -0.016310379, -0.027719192, -0.032282718, -0.004391488, 0.062199164, 0.14796448, -0.045128196, -0.070553556, 0.08103276, 0.057031997, 0.028274542, -0.0946509, 0.03250003, -0.001274437, -0.06615905, -0.043776043, -0.0841717, -0.06799412, 0.17761531, -0.025014881, -0.029095493, -0.11164943, 0.019038836, -0.03530092, 0.017252058, 0.06205429, -0.040274926, 0.00063306844, -0.033031233, 0.010648226, 0.104888655, -0.07296812, 0.027526028, 0.08880766, 0.009380581, -0.01047317, -0.05794953, 0.08088789, -0.048242986, -0.0024688914, 0.027888212, -0.0708916, -0.026560202, -0.0055172783, 0.044017497, 0.01541699, -0.030351067, -0.077507496, 0.09614793, 0.09503723, 0.03520434, 0.017819481, -0.11174601, -0.0015121206, 0.04744618, 0.08962861, -0.014813349, 0.059543144, 0.060750425, -0.02170693, 0.0032596611, -0.09382995, 0.011336377, -0.088952534]
    # the splitted doc: again! Sentiment: Positive
    # the embedding of the doc: [0.067159414, -0.054115806, -0.057209868, 0.083539754, 0.016137673, 0.10210415, -0.035945754, 0.02686073, 0.04844335, -0.042224888, -0.011026398, -0.06033427, -0.028711103, -0.08426777, 0.13516818, 0.077776305, -0.024570517, 0.03940383, 0.002749016, 0.042285554, -0.008243256, 0.0042391727, 0.011329738, -0.029393617, -0.035399742, 0.02642089, -0.057968218, -0.028999276, 0.1156331, -0.07298353, 0.04865569, -0.09925275, 0.09172993, -0.01283127, -0.062427312, 0.10064811, -0.03773546, -0.00045548353, -0.04735133, -0.042649563, 0.025829377, -0.10046611, 0.0144541375, -0.022795979, -0.050688066, 0.046380643, -0.051385745, -0.09433865, -0.00888027, 0.062609315, -0.090455905, 0.043802254, 0.06351934, 0.17933443, -0.017199362, -0.022022463, 0.036582768, 0.025237864, -0.010723059, -0.031425994, 0.053175453, -0.11423773, 0.08256907, -0.06691674, -0.024995193, -0.0099192085, -0.083600424, 0.121821225, 0.09634069, 0.045500956, 0.02479802, -0.02781625, 0.048018675, 0.013225611, 0.06430802, 0.020702936, 0.1029535, -0.02787692, 0.028392596, -0.103074834, -0.091183916, 0.025344033, -0.001270235, -0.106168896, -0.02047543, -0.102468155, -0.034489725, 0.0025802834, -0.03148666, 0.11532976, 0.043104574, 0.091244586, -0.034338057, -0.0118985, -0.016319677, -0.048534352, -0.03567275, 0.10519821, 0.001149847, 0.08669449, 0.01600117, 0.029454285, -0.059393916, 0.09100191, 0.015500659, 0.039555497, 0.018579558, -0.039828505, -0.019186236, -0.008804435, -0.106957585, -0.0491107, -0.02959079, 0.07595626, -0.02435818, -0.030515974, -0.042922568, 0.06527871, -0.08432844, 0.0253592, -0.026527058, -0.051749755, -0.02435818, -0.031122655, -0.08275107, -0.11496575, -0.049474705, 0.035794087, -0.052417103, -0.099495426, 0.07753363, 0.012596182, 0.05202276, 0.10762493, -0.083054416, -0.035460413, -0.0013953627, -0.042952903, -0.05369113, -0.084389105, 0.09081991, -0.102407485, -0.021855626, -0.030561475, -0.00427709, 0.011663412, 0.0027831418, 0.01821555, 0.022917315, 0.09354997, -0.012414178, 0.13395482, -0.07989968, 0.015417241, 0.039919507, -0.10289283, 0.0032514224, -0.026845565, -0.04352925, 0.0065862634, -0.014256966, 0.03254835, -0.06109262, -0.045318954, 0.090455905, 0.05584484, -0.10204348, -0.10580489, -0.122670576, -0.1124177, 0.05338779, 0.020733269, -0.060758945, -0.03828147, -0.0045538875, -0.02349366, -0.063458666, -0.002240922, -0.019337907, 0.060455605, 0.018185215, 0.090455905, 0.07158817, -0.04537962, 0.03934316, -0.022083132, -0.017942544, 0.13577485, -0.00918361, 0.042497892, 0.014401053, 0.111386344, 0.12218524, 0.065521374, 0.026026547, -0.082872406, 0.08523846, 0.092639945, -0.03718945, -0.07510691, 0.02646639, 0.024691852, 0.07261953, -0.020202424, -0.043893255, -0.044408932, -0.036249094, 0.07249819, -0.08317575, 0.038463477, 0.17654371, -0.060940947, 0.07486424, -0.012740267, -0.08651248, 0.045409955, -0.057118867, 0.069404125, 0.079110995, 0.066492066, -0.024479514, 0.022583641, 0.08851453, 0.023175154, -0.024919357, 0.06424735, 0.09209394, -0.025283365, -0.0428619, -0.16574481, 0.025237864, -0.034793064, 0.076138265, -0.09148726, -0.005429781, 0.0035945757, 0.10950564, 0.0175027, -0.04537962, -0.036188427, -0.014545139, 0.085905805, 0.06358001, 0.07061748, -0.06995014, 0.017123526, 0.06782676, 0.005190901, -0.034398723, 0.05011172, 0.035096403, 0.0023736332, -0.039161157, -0.04874669, -0.012846436, -0.011178068]

As you can see in the output, each splitted doc will have a vector representation.
