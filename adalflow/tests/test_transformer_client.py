import unittest
import torch
from adalflow.components.model_client.transformers_client import TransformerEmbeddingModelClient, TransformerLLMModelClient, TransformerRerankerModelClient
from adalflow.core.types import ModelType
from adalflow.core import Embedder, Generator

# Set the number of threads for PyTorch, avoid segementation fault
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class TestTransformerEmbeddingModelClient(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "what is panda?"
        self.documents = [
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "The red panda (Ailurus fulgens), also called the lesser panda, the red bear-cat, and the red cat-bear, is a mammal native to the eastern Himalayas and southwestern China.",
        ]

    def test_execution(self):
        test_input = "Hello word"
        embedding_model = "thenlper/gte-base"
        model_kwargs = {"model": embedding_model}
        tokenizer_kwargs = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": 'pt'
        }
        model_client = TransformerEmbeddingModelClient(
            model_name=embedding_model,
            tokenizer_kwargs=tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=test_input, model_kwargs=model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_integration_with_embedder(self):

        test_input = "Hello word"
        embedding_model = "thenlper/gte-base"
        model_kwargs = {"model": embedding_model}
        tokenizer_kwargs = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": 'pt'
        }
        model_client = TransformerEmbeddingModelClient(
            model_name=embedding_model,
            tokenizer_kwargs=tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        embedder = Embedder(model_client=model_client,
            model_kwargs=model_kwargs
            )
        output = embedder(test_input)
        print(output)

class TestTransformerLLMModelClient(unittest.TestCase):

    def setUp(self) -> None:

        self.model_kwargs = {
            "model": "roneneldan/TinyStories-1M",
            "temperature": 0.1,
            "do_sample": True
        }
        self.tokenizer_kwargs = {
            "max_length": True,
            "truncation": True,
        }
        self.prompt_kwargs = {
            "input_str": "Where is Brian?", # test input
        }
        self.chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": False
        }
        self.chat_template = """
        {%- for message in messages %}
            {%- if message['role'] == 'user' %}
                {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
            {%- elif message['role'] == 'system' %}
                {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
            {%- elif message['role'] == 'assistant' %}
                {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
            {%- endif %}
        {%- endfor %}
        """ # Reference: https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-create-a-chat-template
    
    def test_exectution(self):
        model_client = TransformerLLMModelClient(
            tokenizer_kwargs=self.tokenizer_kwargs,
            local_files_only=False,
            init_from="autoclass",
            apply_chat_template=True,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs
            )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input="Where is brian?", model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_response(self):

        """Test the TransformerLLM model with roneneldan/TinyStories-1M for generating a response."""
        model_client = TransformerLLMModelClient(
        )

        # Define a sample input
        input_text = "Hello, what's the weather today?"

        # Test generating a response, providing the 'model' keyword
        # response = transformer_llm_model_component(input=input_text, model=transformer_llm_model)
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input_text, self.model_kwargs)
        response = model_client.call(api_kwargs)

        # Check if the response is valid
        self.assertIsInstance(response, list, "The response should be a list.")
        self.assertTrue(all([isinstance(elmt, str) for elmt in response]),  "all elements in the response list should be strings.")
        self.assertTrue(len(response) > 0, "The response should not be empty.")

        # Optionally, print the response for visual verification during testing
        print(f"Generated response: {response}")

    def test_integration_with_generator_autoclass(self):
        model_client = TransformerLLMModelClient(
            tokenizer_kwargs=self.tokenizer_kwargs,
            local_files_only=False,
            init_from="autoclass",
            apply_chat_template=True,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs
            )
        generator = Generator(
            model_client=model_client,
            model_kwargs=self.model_kwargs,
            # prompt_kwargs=prompt_kwargs,
            # output_processors=JsonParser(),
        )
        output = generator(prompt_kwargs=self.prompt_kwargs)
        print(output)

    def test_integration_with_generator_pipeline(self):
        model_client = TransformerLLMModelClient(
            tokenizer_kwargs=self.tokenizer_kwargs,
            local_files_only=False,
            init_from="pipeline",
            apply_chat_template=True,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs
            )
        generator = Generator(
            model_client=model_client,
            model_kwargs=self.model_kwargs,
            # prompt_kwargs=prompt_kwargs,
            # output_processors=JsonParser(),
        )
        output = generator(prompt_kwargs=self.prompt_kwargs)
        print(output)

class TransformerRerankerModelClient(unittest.TestCase):
    def setUp(self) -> None:

        self.query = "what is panda?"
        self.documents = [
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "The red panda (Ailurus fulgens), also called the lesser panda, the red bear-cat, and the red cat-bear, is a mammal native to the eastern Himalayas and southwestern China.",
        ]

    def test_execution(self):
        transformer_reranker_model = "BAAI/bge-reranker-base"
        transformer_reranker_model_client = TransformerRerankerModelClient(
            tokenizer_kwargs={"padding": True}
        )
        print(
            f"Testing TransformerRerankerModelClient with model {transformer_reranker_model}"
        )

        model_kwargs = {
            "model": transformer_reranker_model,
            "documents": self.documents,
            "top_k": 2,
        }

        api_kwargs = transformer_reranker_model_client.convert_inputs_to_api_kwargs(self.query, model_kwargs=model_kwargs)
        output = transformer_reranker_model_client.call(api_kwargs)
        # assert output is a list of list with length 2
        self.assertEqual(len(output), 2)
        self.assertEqual(type(output[0]), list)
        self.assertEqual(type(output[1]), list)
        # assert output[0] is a list of int of length top_k
        tok_k = model_kwargs["top_k"]
        self.assertTrue(all([isinstance(elmt, int) for elmt in output[0]]))
        self.assertEqual(len(output[0]), tok_k)
        # assert output[1] is a list of float of length top_k
        tok_k = model_kwargs["top_k"]
        self.assertTrue(all([isinstance(elmt, float) for elmt in output[1]]))
        self.assertEqual(len(output[1]), tok_k)

    def test_transformer_reranker_client(self):
        transformer_reranker_client = TransformerRerankerModelClient(
            tokenizer_kwargs={"padding": True}
        )
        print("Testing transformer reranker client")
        # run the model
        kwargs = {
            "model": "BAAI/bge-reranker-base",
            "documents": self.documents,
            "top_k": 2,
        }
        api_kwargs = transformer_reranker_client.convert_inputs_to_api_kwargs(
            input=self.query,
            model_kwargs=kwargs,

        )
        print(api_kwargs)
        self.assertEqual(api_kwargs["model"], "BAAI/bge-reranker-base")
        output = transformer_reranker_client.call(
            api_kwargs=api_kwargs
        )
        self.assertEqual(type(output), tuple)


if __name__ == "__main__":
    unittest.main()
