
import unittest
from lightrag.core.types import Document
from lightrag.core.document_splitter_optim import TextSplitter  # Import the TextSplitter from its module

class TestTextSplitter(unittest.TestCase):

    def setUp(self):
        # Set up a TextSplitter instance before each test
        self.splitter = TextSplitter(split_by="word", chunk_size=5, chunk_overlap=2)

    def test_invalid_split_by(self):
        # Test initialization with invalid split_by value
        with self.assertRaises(ValueError):
            TextSplitter(split_by="invalid", chunk_size=5, chunk_overlap=0)

    def test_negative_chunk_size(self):
        # Test initialization with negative chunk_size
        with self.assertRaises(ValueError):
            TextSplitter(split_by="word", chunk_size=-1, chunk_overlap=0)

    def test_negative_chunk_overlap(self):
        # Test initialization with negative chunk_overlap
        with self.assertRaises(ValueError):
            TextSplitter(split_by="word", chunk_size=5, chunk_overlap=-1)

    def test_split_by_word(self):
        # Test the basic functionality of splitting by word
        text = "This is a simple test"
        expected = ["This is a simple test"]
        result = self.splitter.split_text(text)
        self.assertEqual(result, expected)

    def test_split_by_sentence(self):
        # Test splitting by sentence
        splitter = TextSplitter(split_by="sentence", chunk_size=1, chunk_overlap=0)
        text = "This is a test. It should work well."
        expected = ["This is a test.", " It should work well."]
        result = splitter.split_text(text)
        self.assertEqual(result, expected)

    def test_overlap_handling(self):
        # Test proper handling of overlap
        text = "one two three four five six seven"
        expected = ["one two three four five ", "four five six seven"]
        result = self.splitter.split_text(text)
        self.assertEqual(result, expected)

    def test_document_splitting(self):
        # Test splitting a list of documents
        docs = [Document(text="This is a simple test to check splitting.", id="1")]
        expected_texts = ["This is a simple test ", "simple test to check splitting."]
        result = self.splitter.call(docs)
        result_texts = [doc.text for doc in result]
        self.assertEqual(result_texts, expected_texts)

    def test_empty_text_handling(self):
        # Test handling of empty text
        with self.assertRaises(ValueError):
            self.splitter.call([Document(text=None, id="1")])

if __name__ == '__main__':
    unittest.main()


# import unittest
# from lightrag.core.types import Document
# from lightrag.core.document_splitter_optim import TextSplitter

# class TestTextSplitter(unittest.TestCase):

#     def setUp(self):
#         self.splitter = TextSplitter(split_by="word", chunk_size=5, chunk_overlap=2)

#     def test_empty_string(self):
#         # Testing splitting an empty string
#         text = ""
#         expected = []
#         result = self.splitter.split_text(text)
#         self.assertEqual(result, expected)

#     def test_no_delimiter_present(self):
#         # Testing text that does not contain the delimiter
#         text = "Thisisacontinuousstring"
#         expected = ["Thisisacontinuousstring"]
#         result = self.splitter.split_text(text)
#         self.assertEqual(result, expected)

#     def test_exact_chunk_length(self):
#         # Testing text that is exactly the length of one chunk
#         text = "one two three"
#         expected = ["one two three"]
#         result = self.splitter.split_text(text)
#         self.assertEqual(result, expected)

#     def test_multiple_documents(self):
#         # Testing multiple documents to ensure batch processing
#         docs = [
#             Document(text="This is the first document.", id="1"),
#             Document(text="This is the second document.", id="2")
#         ]
#         expected_texts = ["This is the first document.", "This is the second document."]
#         result = self.splitter.call(docs)
#         result_texts = [doc.text for doc in result]
#         self.assertEqual(result_texts, expected_texts)


#     def test_invalid_overlap(self):
#         # Test initialization with chunk_overlap greater than or equal to chunk_size
#         with self.assertRaises(ValueError):
#             TextSplitter(split_by="word", chunk_size=5, chunk_overlap=5)
#         with self.assertRaises(ValueError):
#             TextSplitter(split_by="word", chunk_size=5, chunk_overlap=6)


#     def test_special_characters(self):
#         # Testing texts with special characters
#         text = "Data: 123,456;789. End."
#         splitter = TextSplitter(split_by="sentence", chunk_size=1, chunk_overlap=0)
#         expected = ["Data: 123,456;789.", " End."]
#         result = splitter.split_text(text)
#         self.assertEqual(result, expected)

# if __name__ == '__main__':
#     unittest.main()

