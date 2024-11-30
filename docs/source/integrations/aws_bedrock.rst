.. _integration-aws-bedrock:

AWS Bedrock API Client
=======================

.. admonition:: Author
   :class: highlight

   `Ajith Kumar <https://github.com/ajithvcoder>`_

Getting Credentials
-------------------

You need to have an AWS account and an access key and secret key to use AWS Bedrock services. Moreover, the account associated with the access key must have 
the necessary permissions to access Bedrock services. Refer to the `AWS documentation <https://docs.aws.amazon.com/singlesignon/latest/userguide/howtogetcredentials.html>`_ for more information on obtaining credentials.

Enabling Foundation Models
--------------------------

AWS Bedrock offers several foundation models from providers like "Meta," "Amazon," "Cohere," "Anthropic," and "Microsoft." To access these models, you need to enable them first. Note that each AWS region supports a specific set of models. Not all foundation models are available in every region, and pricing varies by region.

Pricing information: `AWS Bedrock Pricing <https://aws.amazon.com/bedrock/pricing/>`_

Steps for enabling model access:

1. Select the desired region in the AWS Console (e.g., `us-east-1 (N. Virginia)`).
2. Navigate to the `Bedrock services home page <https://console.aws.amazon.com/bedrock/home>`_.
3. On the left sidebar, under "Bedrock Configuration," click "Model Access."

   You will be redirected to a page where you can select the models to enable.

Note:

1. Avoid enabling high-cost models to prevent accidental high charges due to incorrect usage.
2. As of Nov 2024, a cost-effective option is the Llama-3.2 1B model, with model ID: ``meta.llama3-2-1b-instruct-v1:0`` in the ``us-east-1`` region. 
3. AWS tags certain models with `inferenceTypesSupported` = `INFERENCE_PROFILE` and in UI it might appear with a tooltip as `This model can only be used through an inference profile.` In such cases you may need to use the Model ARN: ``arn:aws:bedrock:us-east-1:306093656765:inference-profile/us.meta.llama3-2-1b-instruct-v1:0`` in the model ID field when using Adalflow.
4. Ensure (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME) or AWS_DEFAULT_PROFILE is set in the ``.env`` file. Mention exact key names in ``.env`` file for example access key id is ``AWS_ACCESS_KEY_ID`` 

.. code-block:: python

   import adalflow as adal
   import os

   # Ensure (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME) or AWS_DEFAULT_PROFILE is set in the .env file
   adal.setup_env()
   model_client = adal.BedrockAPIClient()
   model_client.list_models()

Which ever profile is tagged with ``INFERENCE_PROFILE`` you might need to provide ``Model ARN`` in ``model`` filed of ``model_kwargs``

References
----------

1. You can refer to Model IDs or Model ARNs `here <https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/models>`_. Clicking on a model card provides additional information.
2. Internally, Adalflow's AWS client uses the `Converse API <https://boto3.amazonaws.com/v1/documentation/api/1.35.8/reference/services/bedrock-runtime/client/converse.html>`_ for each conversation.
