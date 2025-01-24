# Azure AI Client Example

This example demonstrates how to use the AdalFlow Azure AI client with both API key and Azure AD authentication methods.

## Prerequisites

1. Install the required packages:
```bash
pip install adalflow azure-identity python-dotenv
```

2. Set up your Azure OpenAI service and get the necessary credentials:
   - API key authentication: Get your API key from the Azure portal
   - Azure AD authentication: Set up an Azure AD application and get the client ID, tenant ID, and client secret

3. Configure your environment variables by copying the `.env.example` file to `.env` and filling in your values:
```bash
cp .env.example .env
```

## Environment Variables

Edit the `.env` file and fill in your values:

```env
# Azure OpenAI API Configuration
AZURE_OPENAI_API_KEY="your_api_key_here"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_VERSION="2024-02-15-preview"

# Azure AD Authentication (Optional - if using AAD auth)
AZURE_CLIENT_ID="your_client_id_here"
AZURE_TENANT_ID="your_tenant_id_here"
AZURE_CLIENT_SECRET="your_client_secret_here"

# Azure Model Deployment
AZURE_MODEL_NAME="your_model_deployment_name"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
```

## Running the Example

The example script demonstrates:
1. Chat completion with API key authentication
2. Streaming chat completion
3. Text embeddings
4. Chat completion with Azure AD authentication

To run the example:

```bash
python azure_client_example.py
```

## Features Demonstrated

1. **Multiple Authentication Methods**:
   - API key-based authentication
   - Azure AD authentication using DefaultAzureCredential

2. **Chat Completions**:
   - Regular chat completion
   - Streaming chat completion
   - System and user message handling

3. **Text Embeddings**:
   - Generate embeddings for multiple texts
   - Embedding dimension output

## Example Output

You should see output similar to this:

```
Testing with API key authentication:

=== Testing Chat Completion ===
[Response from the model about Paris tourist attractions]

=== Testing Chat Completion (Streaming) ===
[Streamed response from the model]

=== Testing Embeddings ===
Generated 2 embeddings
Embedding dimension: 1536

Testing with Azure AD authentication:
[Response from the model using AAD authentication]
```
