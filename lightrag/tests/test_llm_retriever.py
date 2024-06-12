import unittest
from unittest.mock import MagicMock, Mock, patch, call  # Import 'call' here

from lightrag.components.retriever import LLMRetriever
from lightrag.core.types import GeneratorOutput
from lightrag.core.model_client import ModelClient


class TestLLMRetriever(unittest.TestCase):
    def setUp(self):
        with patch("core.model_client.ModelClient", spec=ModelClient) as MockAPI:
            mock_api_client = Mock(ModelClient)
            MockAPI.return_value = mock_api_client
            mock_api_client.call.return_value = "Generated text response"

            mock_api_client.parse_chat_completion.return_value = (
                "Generated text response"
            )
            self.mock_model_client = mock_api_client
            self.mock_generator = MagicMock()
            self.retriever = LLMRetriever(
                top_k=2,
                model_client=self.mock_model_client,
                model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.0},
                output_processors=None,
            )
            # Prepare the generator's expected behavior
            self.retriever.generator = self.mock_generator

    def test_retrieve_single_query(self):
        # Setup
        query = "What does Luna like?"
        expected_indices = [0, 1]
        self.mock_generator.return_value = GeneratorOutput(
            data=expected_indices, error=None, raw_response="[0, 1]"
        )

        # Action
        result = self.retriever.retrieve(query_or_queries=query)

        # Assert
        self.mock_generator.assert_called_once_with(
            prompt_kwargs={
                "task_desc_str": self.retriever.task_desc_prompt(
                    top_k=self.retriever.top_k
                ),
                "input_str": query,
            }
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], GeneratorOutput)
        self.assertEqual(result[0].data, expected_indices)

    def test_retrieve_multiple_queries(self):
        # Setup
        queries = ["What does Luna like?", "What does Luna do?"]
        expected_responses = [
            GeneratorOutput(data=[0], error=None, raw_response="[0]"),
            GeneratorOutput(data=[1], error=None, raw_response="[1]"),
        ]
        self.mock_generator.side_effect = expected_responses  # Mock sequential calls

        # Action
        results = self.retriever.retrieve(query_or_queries=queries)

        # Assert
        expected_calls = [
            call(
                prompt_kwargs={
                    "task_desc_str": self.retriever.task_desc_prompt(
                        top_k=self.retriever.top_k
                    ),
                    "input_str": query,
                }
            )
            for query in queries
        ]
        self.mock_generator.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(len(results), 2)
        for i, result in enumerate(results):
            self.assertEqual(result.data, expected_responses[i].data)

    def test_retrieve_with_no_results(self):
        # Setup
        query = "Non-existent query"
        self.mock_generator.return_value = GeneratorOutput(
            data=[], error=None, raw_response="[]"
        )

        # Action
        result = self.retriever.retrieve(query_or_queries=query)

        # Assert
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], GeneratorOutput)
        self.assertEqual(result[0].data, [])


if __name__ == "__main__":
    unittest.main()
