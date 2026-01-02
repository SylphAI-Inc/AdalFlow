import unittest

from adalflow.core.types import TokenLogProb
from adalflow.components.output_parsers.constrained_parser import (
    ConstrainedSelectionParser,
)


class ConstrainedSelectionParserTests(unittest.TestCase):
    def test_logprob_selection_prefers_high_probability_option(self):
        parser = ConstrainedSelectionParser(options=["A", "B"])
        logprobs = [
            [
                TokenLogProb(token="A", logprob=-0.1, choice_index=0),
                TokenLogProb(token="B", logprob=-5.0, choice_index=0),
            ]
        ]

        result = parser.call({"response": "ignored", "logprobs": logprobs})
        self.assertEqual("A", result)

    def test_logprob_selection_falls_back_to_text_match(self):
        parser = ConstrainedSelectionParser(
            options=["Positive", "Negative"], allow_partial_match=False
        )
        result = parser.call({"response": "Negative", "logprobs": []})
        self.assertEqual("Negative", result)

    def test_no_match_returns_first_option(self):
        parser = ConstrainedSelectionParser(
            options=["Yes", "No"], allow_partial_match=False
        )
        result = parser.call("Maybe")
        self.assertEqual("Yes", result)


if __name__ == "__main__":
    unittest.main()
