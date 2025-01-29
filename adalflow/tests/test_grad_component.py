import unittest
import asyncio
from unittest.mock import MagicMock, patch
from adalflow.optim.grad_component import GradComponent
from adalflow.optim.parameter import Parameter, OutputParameter


class TestGradCpomponent(GradComponent):
    def __init__(self):
        super().__init__(desc="test_desc")

    def call(self, *args, **kwargs):
        return "mock_call"


class TestGradComponent(unittest.TestCase):

    def setUp(self):
        self.component = TestGradCpomponent()
        self.component.name = "test_component"
        self.component.training = True

    def test_initialization(self):
        # Test if backward_engine is set to None initially
        self.assertIsNone(self.component.backward_engine)

    @patch.object(
        TestGradCpomponent, "forward", return_value=OutputParameter(data="mock_forward")
    )
    @patch.object(TestGradCpomponent, "call", return_value="mock_call")
    def test_call_in_training(self, mock_call, mock_forward):
        # When in training mode, forward should be called
        self.component.train()
        result = self.component()
        mock_forward.assert_called_once()
        mock_call.assert_not_called()
        self.assertEqual(result.data, "mock_forward")

    @patch.object(TestGradCpomponent, "forward", return_value="mock_forward")
    @patch.object(TestGradCpomponent, "call", return_value="mock_call")
    def test_call_not_in_training(self, mock_call, mock_forward):
        # When not in training mode, call should be called
        self.component.training = False
        result = self.component()
        mock_call.assert_called_once()
        mock_forward.assert_not_called()
        self.assertEqual(result, "mock_call")

    def test_acall_not_implemented(self):
        # Test if acall raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            asyncio.run(self.component.acall())

    def test_forward(self):
        self.component.call = MagicMock(return_value="mock_data")

        # Create an actual Parameter instance
        param = Parameter(data="input_data", name="test_param")
        param.add_successor_map_fn(
            successor=self.component, map_fn=lambda x: "unwrapped_" + str(x)
        )
        args = [param]
        kwargs = {"id": 123, "other_param": param}

        # Call the forward method
        response = self.component.forward(*args, **kwargs)

        self.assertEqual(isinstance(response, Parameter), True)
        self.assertEqual(response.data, "mock_data")
        self.assertEqual(response.full_response, "mock_data")
        self.assertEqual(
            len(response.predecessors), 1
        )  #  predecessors is a set, so it should be 1


if __name__ == "__main__":
    unittest.main()
