import unittest
from adalflow.components.model_client.sambanova_client import SambaNovaClient
from adalflow.core import Generator
from adalflow.utils import setup_env, get_logger


class TestSambaNovaClient(unittest.TestCase):

    def setUp(self):
        self.log = get_logger(level="DEBUG")
        self.prompt_kwargs = {"input_str": "What is the meaning of life?"}
        setup_env()

    def test_sambanova_client(self):
        gen = Generator(
            model_client=SambaNovaClient(),
        )
        response = gen.generate(**self.prompt_kwargs)
        self.assertIsNotNone(response)
        self.log.debug(f"Response: {response}")


if __name__ == "__main__":
    unittest.main()
