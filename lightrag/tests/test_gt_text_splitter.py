import unittest
from lightrag.core.types import Document
from lightrag.components.data_process.text_splitter import TextSplitter  # Import your TextSplitter
from lightrag.components.data_process.document_splitter import DocumentSplitter  # Import the ground truth splitter


class TestTextSplitterComparison(unittest.TestCase):

    def setUp(self):
        self.text_splitter = TextSplitter(split_by="word", chunk_size=5, chunk_overlap=2)
        self.ground_truth_splitter = DocumentSplitter(split_by="word", split_length=5, split_overlap=2)

    def compare_splits(self, text):
        expected = self.ground_truth_splitter.split_text(text)
        result = self.text_splitter.split_text(text)
        
        print(f"expected: {expected}")
        print(f"result: {result}")
        self.assertEqual(result, expected)

    def test_exact_chunk_size(self):
        text = "one two three four five"
        self.compare_splits(text)

    def test_less_than_chunk_size(self):
        text = "one two"
        self.compare_splits(text)

    def test_single_word(self):
        text = "word"
        self.compare_splits(text)

    def test_overlap_handling(self):
        text = "one two three four five six seven"
        self.compare_splits(text)

    def test_multiple_chunks_with_overlap(self):
        text = "one two three four five six seven eight nine ten eleven"
        self.compare_splits(text)

    def test_end_index_matches_length(self):
        text = "one two three four five six"
        self.compare_splits(text)

    def test_long_text(self):
        text = " ".join(["word"] * 50)
        self.compare_splits(text)

    def test_split_by_sentence(self):
        self.text_splitter = TextSplitter(split_by="sentence", chunk_size=1, chunk_overlap=0)
        self.ground_truth_splitter = DocumentSplitter(split_by="sentence", split_length=1, split_overlap=0)
        text = "This is a test. It should work well."

        self.compare_splits(text)

    def test_split_by_page(self):
        self.text_splitter = TextSplitter(split_by="page", chunk_size=1, chunk_overlap=0)
        self.ground_truth_splitter = DocumentSplitter(split_by="page", split_length=1, split_overlap=0)
        text = "This is a test\fThis is another page"
        self.compare_splits(text)

    def test_split_by_passage(self):
        self.text_splitter = TextSplitter(split_by="passage", chunk_size=1, chunk_overlap=0)
        self.ground_truth_splitter = DocumentSplitter(split_by="passage", split_length=1, split_overlap=0)
        text = "This is a test\n\nThis is another passage"
        self.compare_splits(text)

    def test_empty_text(self):
        text = ""
        self.compare_splits(text)

    def test_special_characters(self):
        text = "one! two@ three# four$ five% six^ seven& eight* nine( ten)"
        self.compare_splits(text)

    def test_multiple_spaces(self):
        text = "one  two   three    four     five"
        self.compare_splits(text)

    def test_newlines(self):
        text = "one\ntwo\nthree\nfour\nfive\nsix"
        self.compare_splits(text)

    def test_tabs(self):
        text = "one\ttwo\tthree\tfour\tfive\tsix"
        self.compare_splits(text)

    def test_varied_delimiters(self):
        text = "one. two, three; four: five! six? seven"
        self.compare_splits(text)

    def test_text_with_punctuation(self):
        text = "Hello, world! This is a test. Let's see how it works."
        self.compare_splits(text)

    def test_long_paragraph(self):
        text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )
        self.compare_splits(text)

    def test_trailing_whitespace(self):
        text = "one two three four five   "
        self.compare_splits(text)

    def test_leading_whitespace(self):
        text = "   one two three four five"
        self.compare_splits(text)

    def test_mixed_whitespace(self):
        text = " one  two   three    four     five "
        self.compare_splits(text)

    def test_chunk_size_greater_than_overlap(self):
        self.text_splitter = TextSplitter(split_by="word", chunk_size=4, chunk_overlap=2)
        self.ground_truth_splitter = DocumentSplitter(split_by="word", split_length=4, split_overlap=2)
        text = "one two three four five six seven eight nine ten"
        self.compare_splits(text)

    def test_overlap_zero(self):
        self.text_splitter = TextSplitter(split_by="word", chunk_size=4, chunk_overlap=0)
        self.ground_truth_splitter = DocumentSplitter(split_by="word", split_length=4, split_overlap=0)
        text = "one two three four five six seven eight nine ten"
        self.compare_splits(text)
    
    def test_overlap_zero_end(self):
        self.text_splitter = TextSplitter(split_by="word", chunk_size=5, chunk_overlap=0)
        self.ground_truth_splitter = DocumentSplitter(split_by="word", split_length=5, split_overlap=0)
        text = "one two three four five six seven eight nine ten"
        self.compare_splits(text)
    

if __name__ == '__main__':
    unittest.main()
