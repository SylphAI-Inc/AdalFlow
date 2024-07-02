import unittest
from unittest.mock import MagicMock, Mock, patch, call  # Import 'call' here

from lightrag.components.retriever import LLMRetriever
from lightrag.core.types import RetrieverOutput
from lightrag.core.model_client import ModelClient


class TestLLMRetriever(unittest.TestCase):
    def setUp(self):
        with patch("core.model_client.ModelClient", spec=ModelClient) as MockAPI:
            mock_api_client = Mock(ModelClient)
            MockAPI.return_value = mock_api_client
            mock_api_client.call.return_value = "Generated text response"

            mock_api_client.parse_chat_completion.return_value = "Parsed text response"
            self.mock_model_client = mock_api_client
            self.mock_generator = MagicMock()
            self.retriever = LLMRetriever(
                top_k=2,
                model_client=self.mock_model_client,
                model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.0},
            )
            # Prepare the generator's expected behavior
            self.retriever.generator = self.mock_generator


#     def test_retrieve_single_query(self):
#         # Setup
#         query = "What does Luna like?"
#         expected_indices = [0, 1]
#         self.mock_generator.return_value = RetrieverOutput(
#             doc_indices=expected_indices, doc_scores=None
#         )

#         # Action
#         result = self.retriever.retrieve(input=query)

#         # Assert
#         self.mock_generator.assert_called_once_with(
#             prompt_kwargs={
#                 "top_k": self.retriever.top_k,
#                 "input_str": query,
#             }
#         )
#         self.assertEqual(len(result), 1)
#         self.assertIsInstance(result[0], RetrieverOutput)
#         self.assertEqual(result[0].doc_indices, expected_indices)

#     def test_retrieve_multiple_queries(self):
#         # Setup
#         queries = ["What does Luna like?", "What does Luna do?"]
#         expected_responses = [
#             RetrieverOutput(
#                 doc_indices=[0],
#                 doc_scores=None,
#             ),
#             RetrieverOutput(doc_indices=[1], doc_scores=None),
#         ]
#         self.mock_generator.side_effect = expected_responses  # Mock sequential calls

#         # Action
#         results = self.retriever.retrieve(input=queries)

#         # Assert
#         expected_calls = [
#             call(
#                 prompt_kwargs={
#                     "top_k": self.retriever.top_k,
#                     "input_str": query,
#                 }
#             )
#             for query in queries
#         ]
#         self.mock_generator.assert_has_calls(expected_calls, any_order=False)
#         self.assertEqual(len(results), 2)
#         for i, result in enumerate(results):
#             self.assertIsInstance(result, RetrieverOutput)

#     def test_retrieve_with_no_results(self):
#         # Setup
#         query = "Non-existent query"
#         self.mock_generator.return_value = RetrieverOutput(
#             doc_indices=[], doc_scores=[]
#         )

#         # Action
#         result = self.retriever.retrieve(input=query)

#         # Assert
#         self.assertEqual(len(result), 1)
#         self.assertIsInstance(result[0], RetrieverOutput)
#         # self.assertEqual(result[0].doc_indices, [])


# if __name__ == "__main__":
#     unittest.main()
