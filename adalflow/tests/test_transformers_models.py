"""This tests that the new transformer_client compatibility with several models hosted on HuggingFace."""
import unittest
import torch
from adalflow.components.model_client.transformers_client import TransformerEmbeddingModelClient, TransformerLLMModelClient, TransformerRerankerModelClient
from transformers import AutoModelForSequenceClassification

class TestEmbeddingModels(unittest.TestCase):
    def setUp(self) -> None:
        self.test_input = "Hello world"
        self.auto_tokenizer_kwargs = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": 'pt'
        }
    def test_thenhelper_gte_base(self):
        embedding_model = "thenlper/gte-base"
        model_kwargs = {"model": embedding_model}

        model_client = TransformerEmbeddingModelClient(
            model_name=embedding_model,
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.test_input, model_kwargs=model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_jina_embeddings_V2_small_en(self):
        embedding_model = "jinaai/jina-embeddings-v2-small-en"
        model_kwargs = {"model": embedding_model}
        model_client = TransformerEmbeddingModelClient(
            model_name=embedding_model,
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.test_input, model_kwargs=model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_t5_small_standard_bahasa_cased(self):
        embedding_model = "mesolitica/t5-small-standard-bahasa-cased"
        model_kwargs = {"model": embedding_model}

        # Subclass TransformerEmbeddingModelClient to adapt the class to Encoder-Decoder architecture
        class T5SmallStandardBahasaCased(TransformerEmbeddingModelClient):
            
            def compute_model_outputs(self, batch_dict: dict, model) -> dict:
                print(batch_dict)
                with torch.no_grad():
                    outputs = model.encoder(**batch_dict)
                return outputs


            
        model_client = T5SmallStandardBahasaCased(
            model_name=embedding_model,
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.test_input, model_kwargs=model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_sentence_transformers_all_miniLM_L6_V2(self):
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {"model": embedding_model}

        model_client = TransformerEmbeddingModelClient(
            model_name=embedding_model,
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs
        )
        print(
            f"Testing model client with model {embedding_model}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.test_input, model_kwargs=model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

class TestLLMModels(unittest.TestCase):
    """This class 'has accelerate' as a dependencie for both tests.
        You might need to run the following command in the terminal.
            `pip install accelerate`
    """
    def setUp(self) -> None:
        self.input_text = "Where is Brian?"
        self.auto_tokenizer_kwargs = {}
        self.model_kwargs = {
            "temperature": 0.1,
            "do_sample": True
        }
        self.tokenizer_decode_kwargs = {
            "max_length": True,
            "truncation": True,
        }
        self.prompt_kwargs = {
            "input_str": "Where is Brian?", # test input
        }

    def test_roneneld_tiny_stories_1M(self):
        self.model_kwargs["model"] = "roneneldan/TinyStories-1M"
        model_client = TransformerLLMModelClient(
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs,
            local_files_only=False,
            init_from="autoclass",
            )
        print(
            f"Testing model client with model {"roneneldan/TinyStories-1M"}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.input_text, model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

    def test_nickypro_tinyllama_15m(self):
        self.model_kwargs["model"] = "nickypro/tinyllama-15M"
        model_client = TransformerLLMModelClient(
            auto_tokenizer_kwargs=self.auto_tokenizer_kwargs,
            local_files_only=False,
            init_from="autoclass",
            )
        print(
            f"Testing model client with model {"nickypro/tinyllama-15M"}"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(input=self.input_text, model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs=api_kwargs)
        print(output)

class TestRerankerModel(unittest.TestCase):
    """This class has sentencepieces as a dependencie.
        You might need to run the following command in the terminal.
            `pip install transformers[sentencepiece`]`
    """
    def setUp(self) -> None:
        self.query = "Where is Brian."
        self.documents = [
            "Brian is in the Kitchen.",
            "Brian loves Adalflow.",
            "Adalflow is a python library, not some food inside the kitchen.",
        ]
        self.model_kwargs = {
            "documents": self.documents,
            "top_k": 2,
        }

    def test_jinja_reranker_V1_tiny_en(self):
        self.model_kwargs["model"] = "jinaai/jina-reranker-v1-tiny-en"
        model_client = TransformerRerankerModelClient(
           tokenizer_kwargs={"padding": True},
           auto_model_kwargs={"num_labels": 1}
            )
        print(
            f"Testing model client with model jinaai/jina-reranker-v1-tiny-en"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(self.query, model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs)

    def test_baai_bge_reranker_base(self):
        self.model_kwargs["model"] = "BAAI/bge-reranker-base"
        model_client = TransformerRerankerModelClient(
            tokenizer_kwargs={"padding": True},
            )
        print(
            f"Testing model client with model BAAI/bge-reranker-base"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(self.query, model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs)

    def test_cross_encoder_ms_marco_minilm_L_2_V2(self):
        self.model_kwargs["model"] = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        model_client = TransformerRerankerModelClient(
            tokenizer_kwargs={"padding": True},
            )
        print(
            f"Testing model client with model cross-encoder/ms-marco-MiniLM-L-2-v2"
        )
        api_kwargs = model_client.convert_inputs_to_api_kwargs(self.query, model_kwargs=self.model_kwargs)
        output = model_client.call(api_kwargs)

if __name__ == "__main__":
    unittest.main(verbosity=6)