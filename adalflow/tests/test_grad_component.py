import unittest
import asyncio
from unittest.mock import MagicMock, patch
from adalflow.optim.grad_component import GradComponent
from adalflow.optim.parameter import Parameter


class TestGradComponent(unittest.TestCase):

    def setUp(self):
        self.component = GradComponent()
        self.component.name = "test_component"
        self.component.training = True

    def test_initialization(self):
        # Test if backward_engine is set to None initially
        self.assertIsNone(self.component.backward_engine)

    @patch.object(GradComponent, "forward", return_value="mock_forward")
    @patch.object(GradComponent, "call", return_value="mock_call")
    def test_call_in_training(self, mock_call, mock_forward):
        # When in training mode, forward should be called
        self.component.training = True
        result = self.component()
        mock_forward.assert_called_once()
        mock_call.assert_not_called()
        self.assertEqual(result, "mock_forward")

    @patch.object(GradComponent, "forward", return_value="mock_forward")
    @patch.object(GradComponent, "call", return_value="mock_call")
    def test_call_not_in_training(self, mock_call, mock_forward):
        # When not in training mode, call should be called
        self.component.training = False
        result = self.component()
        mock_call.assert_called_once()
        mock_forward.assert_not_called()
        self.assertEqual(result, "mock_call")

    def test_set_backward_engine_not_implemented(self):
        # Test if set_backward_engine raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.component.set_backward_engine("mock_backward_engine")

    def test_acall_not_implemented(self):
        # Test if acall raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            asyncio.run(self.component.acall())

    def test_forward(self):
        self.component.call = MagicMock(return_value="mock_data")

        # Create an actual Parameter instance
        param = Parameter(data="input_data", name="test_param")
        param.successor_map_fn = MagicMock(side_effect=lambda x: "unwrapped_" + str(x))

        args = [param]
        kwargs = {"id": 123, "other_param": param}

        # Call the forward method
        response = self.component.forward(*args, **kwargs)

        # Assert that call was invoked with unwrapped args and kwargs
        self.component.call.assert_called_once_with(
            "unwrapped_" + str(param), other_param="unwrapped_" + str(param)
        )

        self.assertEqual(isinstance(response, Parameter), True)
        self.assertEqual(response.data, "mock_data")
        self.assertEqual(response.full_response, "mock_data")
        self.assertEqual(
            len(response.predecessors), 1
        )  #  predecessors is a set, so it should be 1


if __name__ == "__main__":
    unittest.main()
