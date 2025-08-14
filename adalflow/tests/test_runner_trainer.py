import unittest
import unittest.mock
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Callable

import adalflow as adal
from adalflow.components.agent.runner_trainer import (
    RunnerTrainer,
    load_datasets,
    default_eval_fn,
    default_loss_fn_factory,
)
from adalflow.datasets.types import GSM8KData as Example
from adalflow.core.types import RunnerResult
from adalflow.optim.parameter import Parameter
from adalflow.components.agent.react import ReActAgent


class TestRunnerTrainer(unittest.TestCase):
    """Tests for the RunnerTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock runner
        self.mock_runner = Mock()
        self.mock_runner.__class__ = adal.Component
        
        # Create test sample
        self.test_sample = Example(
            id="test_1",
            question="What is 2 + 2?",
            answer="4"
        )

    def test_init_with_defaults(self):
        """Test RunnerTrainer initialization with default parameters."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Check that defaults are set
        self.assertEqual(trainer.task, self.mock_runner)
        self.assertIsNotNone(trainer.eval_fn)
        self.assertIsNotNone(trainer.loss_fn)
        self.assertIsNone(trainer.backward_engine_model_config)
        self.assertIsNone(trainer.teacher_model_config)
        self.assertIsNone(trainer.text_optimizer_model_config)
        self.assertFalse(trainer.original_react_agent)

    def test_init_with_custom_parameters(self):
        """Test RunnerTrainer initialization with custom parameters."""
        custom_eval_fn = lambda y, y_gt: 0.8
        custom_loss_fn = Mock()
        backward_config = {"model": "test"}
        teacher_config = {"teacher": "test"}
        text_optimizer_config = {"optimizer": "test"}
        
        trainer = RunnerTrainer(
            runner=self.mock_runner,
            eval_fn=custom_eval_fn,
            loss_fn=custom_loss_fn,
            backward_engine_model_config=backward_config,
            teacher_model_config=teacher_config,
            text_optimizer_model_config=text_optimizer_config,
            original_react_agent=True
        )
        
        self.assertEqual(trainer.eval_fn, custom_eval_fn)
        self.assertEqual(trainer.loss_fn, custom_loss_fn)
        self.assertEqual(trainer.backward_engine_model_config, backward_config)
        self.assertEqual(trainer.teacher_model_config, teacher_config)
        self.assertEqual(trainer.text_optimizer_model_config, text_optimizer_config)
        self.assertTrue(trainer.original_react_agent)

    def test_prepare_task_standard_runner(self):
        """Test prepare_task method for standard Runner."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        task_fn, kwargs = trainer.prepare_task(self.test_sample)
        
        self.assertEqual(task_fn, self.mock_runner.__call__)
        self.assertIn("prompt_kwargs", kwargs)
        self.assertEqual(kwargs["prompt_kwargs"]["input_str"], "What is 2 + 2?")
        self.assertEqual(kwargs["id"], "test_1")

    def test_prepare_task_original_react_agent(self):
        """Test prepare_task method for original ReActAgent."""
        trainer = RunnerTrainer(runner=self.mock_runner, original_react_agent=True)
        
        task_fn, kwargs = trainer.prepare_task(self.test_sample)
        
        self.assertEqual(task_fn, self.mock_runner.__call__)
        self.assertIn("input", kwargs)
        self.assertEqual(kwargs["input"], "What is 2 + 2?")
        self.assertEqual(kwargs["id"], "test_1")

    def test_prepare_eval_standard_runner(self):
        """Test prepare_eval method for standard Runner."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Create mock RunnerResult
        mock_result = RunnerResult(answer="4", step_history=[])
        
        eval_fn, kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        self.assertEqual(eval_fn, trainer.eval_fn)
        self.assertEqual(kwargs["y"], "4")
        self.assertEqual(kwargs["y_gt"], "4")

    def test_prepare_eval_original_react_agent(self):
        """Test prepare_eval method for original ReActAgent."""
        trainer = RunnerTrainer(runner=self.mock_runner, original_react_agent=True)
        
        prediction = "4"
        
        eval_fn, kwargs = trainer.prepare_eval(self.test_sample, prediction)
        
        self.assertEqual(eval_fn, trainer.eval_fn)
        self.assertEqual(kwargs["y"], "4")
        self.assertEqual(kwargs["y_gt"], "4")

    def test_prepare_eval_none_result(self):
        """Test prepare_eval method with None result."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        eval_fn, kwargs = trainer.prepare_eval(self.test_sample, None)
        
        self.assertEqual(eval_fn, trainer.eval_fn)
        self.assertEqual(kwargs["y"], "")
        self.assertEqual(kwargs["y_gt"], "4")

    def test_prepare_loss_standard_runner(self):
        """Test prepare_loss method for standard Runner."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Create mock prediction parameter with RunnerResult
        mock_result = RunnerResult(answer="4", step_history=[])
        mock_pred = Parameter(
            name="pred",
            data=mock_result,
            requires_opt=True
        )
        
        loss_fn, kwargs = trainer.prepare_loss(self.test_sample, mock_pred)
        
        self.assertEqual(loss_fn, trainer.loss_fn)
        self.assertIn("kwargs", kwargs)
        self.assertIn("y", kwargs["kwargs"])
        self.assertIn("y_gt", kwargs["kwargs"])
        self.assertEqual(kwargs["id"], "test_1")
        
        # Check that pred.eval_input is set correctly
        self.assertEqual(mock_pred.eval_input, "4")
        
        # Check y_gt parameter
        y_gt = kwargs["kwargs"]["y_gt"]
        self.assertIsInstance(y_gt, Parameter)
        self.assertEqual(y_gt.data, "4")
        self.assertFalse(y_gt.requires_opt)

    def test_prepare_loss_original_react_agent(self):
        """Test prepare_loss method for original ReActAgent."""
        trainer = RunnerTrainer(runner=self.mock_runner, original_react_agent=True)
        
        # Create mock prediction parameter with string data
        mock_pred = Parameter(
            name="pred",
            data="4",
            requires_opt=True
        )
        
        loss_fn, kwargs = trainer.prepare_loss(self.test_sample, mock_pred)
        
        self.assertEqual(loss_fn, trainer.loss_fn)
        
        # Check that pred.eval_input is set directly to data
        self.assertEqual(mock_pred.eval_input, "4")

    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_prepare_loss_prints_pred(self, mock_print):
        """Test that prepare_loss prints the prediction parameter."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        mock_result = RunnerResult(answer="4", step_history=[])
        mock_pred = Parameter(
            name="pred",
            data=mock_result,
            requires_opt=True
        )
        
        trainer.prepare_loss(self.test_sample, mock_pred)
        
        # Verify print was called with the prediction
        mock_print.assert_called_once_with("pred", mock_pred)

    def test_inheritance_from_adal_component(self):
        """Test that RunnerTrainer properly inherits from AdalComponent."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        self.assertIsInstance(trainer, adal.AdalComponent)
        self.assertTrue(hasattr(trainer, 'task'))
        self.assertTrue(hasattr(trainer, 'eval_fn'))
        self.assertTrue(hasattr(trainer, 'loss_fn'))


class TestRunnerTrainerHelperFunctions(unittest.TestCase):
    """Tests for helper functions in runner_trainer module."""

    def test_default_eval_fn_exact_match(self):
        """Test default_eval_fn with exact matches."""
        result = default_eval_fn("hello", "hello")
        self.assertEqual(result, 1.0)
        
        result = default_eval_fn("  hello  ", "hello")
        self.assertEqual(result, 1.0)  # Should strip whitespace
        
        result = default_eval_fn(42, "42")
        self.assertEqual(result, 1.0)  # Should convert to string

    def test_default_eval_fn_no_match(self):
        """Test default_eval_fn with non-matches."""
        result = default_eval_fn("hello", "world")
        self.assertEqual(result, 0.0)
        
        result = default_eval_fn("42", "43")
        self.assertEqual(result, 0.0)

    def test_default_loss_fn_factory_with_default_eval(self):
        """Test default_loss_fn_factory with default eval function."""
        loss_fn = default_loss_fn_factory()
        
        self.assertIsInstance(loss_fn, adal.EvalFnToTextLoss)
        # Check that it has the expected description
        self.assertIn("Default evaluation", loss_fn.eval_fn_desc)

    def test_default_loss_fn_factory_with_custom_eval(self):
        """Test default_loss_fn_factory with custom eval function."""
        custom_eval = lambda y, y_gt: 0.5
        loss_fn = default_loss_fn_factory(custom_eval)
        
        self.assertIsInstance(loss_fn, adal.EvalFnToTextLoss)
        # The eval_fn should be wrapped in the loss function
        self.assertIn("Default evaluation", loss_fn.eval_fn_desc)

    @patch('adalflow.components.agent.runner_trainer.GSM8K')
    def test_load_datasets(self, mock_gsm8k_class):
        """Test load_datasets function."""
        # Mock the GSM8K constructor
        mock_train = Mock()
        mock_val = Mock() 
        mock_test = Mock()
        
        def mock_gsm8k_side_effect(split, size):
            if split == "train":
                return mock_train
            elif split == "val":
                return mock_val
            elif split == "test":
                return mock_test
                
        mock_gsm8k_class.side_effect = mock_gsm8k_side_effect
        
        train_data, val_data, test_data = load_datasets()
        
        # Check that GSM8K was called with correct parameters
        expected_calls = [
            unittest.mock.call(split="train", size=100),
            unittest.mock.call(split="val", size=50),
            unittest.mock.call(split="test", size=100)
        ]
        mock_gsm8k_class.assert_has_calls(expected_calls)
        
        # Check return values
        self.assertEqual(train_data, mock_train)
        self.assertEqual(val_data, mock_val)
        self.assertEqual(test_data, mock_test)


class TestRunnerTrainerIntegration(unittest.TestCase):
    """Integration tests for RunnerTrainer with mocked dependencies."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_runner = Mock(spec=adal.Component)
        self.test_sample = Example(
            id="integration_test",
            question="What is the capital of France?",
            answer="Paris"
        )

    def test_full_workflow_standard_runner(self):
        """Test complete workflow with standard Runner."""
        # Create trainer
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test prepare_task
        task_fn, task_kwargs = trainer.prepare_task(self.test_sample)
        self.assertEqual(task_fn, self.mock_runner.__call__)
        
        # Simulate runner execution
        mock_result = RunnerResult(answer="Paris", step_history=[])
        
        # Test prepare_eval
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        # Execute evaluation
        eval_score = eval_fn(**eval_kwargs)
        self.assertEqual(eval_score, 1.0)  # Should match exactly
        
        # Test prepare_loss with Parameter
        mock_pred = Parameter(
            name="prediction",
            data=mock_result,
            requires_opt=True
        )
        
        loss_fn, loss_kwargs = trainer.prepare_loss(self.test_sample, mock_pred)
        
        # Verify loss preparation
        self.assertIn("kwargs", loss_kwargs)
        self.assertEqual(loss_kwargs["id"], "integration_test")

    def test_full_workflow_original_react_agent(self):
        """Test complete workflow with original ReActAgent."""
        # Create trainer for original ReActAgent
        trainer = RunnerTrainer(runner=self.mock_runner, original_react_agent=True)
        
        # Test prepare_task
        task_fn, task_kwargs = trainer.prepare_task(self.test_sample)
        self.assertIn("input", task_kwargs)
        self.assertEqual(task_kwargs["input"], "What is the capital of France?")
        
        # Test prepare_eval with string prediction
        prediction = "Paris"
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, prediction)
        
        # Execute evaluation
        eval_score = eval_fn(**eval_kwargs)
        self.assertEqual(eval_score, 1.0)
        
        # Test prepare_loss
        mock_pred = Parameter(
            name="prediction",
            data="Paris",
            requires_opt=True
        )
        
        loss_fn, loss_kwargs = trainer.prepare_loss(self.test_sample, mock_pred)
        
        # Verify eval_input is set directly to data
        self.assertEqual(mock_pred.eval_input, "Paris")

    def test_error_handling_in_eval(self):
        """Test error handling in evaluation methods."""
        # Create trainer with faulty eval function
        def faulty_eval(y, y_gt):
            raise ValueError("Evaluation error")
            
        trainer = RunnerTrainer(runner=self.mock_runner, eval_fn=faulty_eval)
        
        mock_result = RunnerResult(answer="Test", step_history=[])
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        # Should propagate the error
        with self.assertRaises(ValueError):
            eval_fn(**eval_kwargs)

    def test_custom_configurations(self):
        """Test that custom configurations are preserved."""
        backward_config = {"model_client": "test_backward"}
        teacher_config = {"model_client": "test_teacher"}
        text_optimizer_config = {"model_client": "test_optimizer"}
        
        trainer = RunnerTrainer(
            runner=self.mock_runner,
            backward_engine_model_config=backward_config,
            teacher_model_config=teacher_config,
            text_optimizer_model_config=text_optimizer_config
        )
        
        self.assertEqual(trainer.backward_engine_model_config, backward_config)
        self.assertEqual(trainer.teacher_model_config, teacher_config)
        self.assertEqual(trainer.text_optimizer_model_config, text_optimizer_config)


class TestRunnerTrainerEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.mock_runner = Mock(spec=adal.Component)
        
    def test_empty_answer(self):
        """Test handling of empty answers."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        empty_sample = Example(
            id="empty_test",
            question="Test question?",
            answer=""
        )
        
        # Test with empty RunnerResult answer
        mock_result = RunnerResult(answer="", step_history=[])
        eval_fn, eval_kwargs = trainer.prepare_eval(empty_sample, mock_result)
        
        # Should match empty string
        result = eval_fn(**eval_kwargs)
        self.assertEqual(result, 1.0)

    def test_none_runner_result_answer(self):
        """Test handling of RunnerResult with None answer."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        test_sample = Example(
            id="none_test",
            question="Test question?",
            answer="expected"
        )
        
        # Test with RunnerResult having None answer
        mock_result = RunnerResult(answer=None, step_history=[])
        eval_fn, eval_kwargs = trainer.prepare_eval(test_sample, mock_result)
        
        # Should use empty string for None answer
        self.assertEqual(eval_kwargs["y"], "")

    def test_special_characters_in_answers(self):
        """Test handling of special characters in answers."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        special_sample = Example(
            id="special_test",
            question="What is the result?",
            answer="$100.50"
        )
        
        mock_result = RunnerResult(answer="$100.50", step_history=[])
        eval_fn, eval_kwargs = trainer.prepare_eval(special_sample, mock_result)
        
        result = eval_fn(**eval_kwargs)
        self.assertEqual(result, 1.0)

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized in evaluation."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        test_sample = Example(
            id="whitespace_test",
            question="Test?",
            answer="answer"
        )
        
        # Test with extra whitespace
        mock_result = RunnerResult(answer="  answer  ", step_history=[])
        eval_fn, eval_kwargs = trainer.prepare_eval(test_sample, mock_result)
        
        result = eval_fn(**eval_kwargs)
        self.assertEqual(result, 1.0)  # Should match after stripping


class TestRunnerTrainerNewRunnerFeatures(unittest.TestCase):
    """Tests for RunnerTrainer compatibility with new Runner features."""

    def setUp(self):
        """Set up test fixtures for new Runner features."""
        self.mock_runner = Mock(spec=adal.Component)
        self.test_sample = Example(
            id="new_feature_test",
            question="Test with new runner?",
            answer="New runner works"
        )

    def test_runner_trainer_with_training_flag(self):
        """Test RunnerTrainer works with Runner that has training parameter."""
        # Create a mock runner that simulates training behavior
        mock_runner = Mock(spec=adal.Component)
        mock_runner.training = False
        
        trainer = RunnerTrainer(runner=mock_runner)
        
        # Test that trainer initialization works with training-capable runners
        self.assertEqual(trainer.task, mock_runner)
        self.assertIsNotNone(trainer.eval_fn)
        self.assertIsNotNone(trainer.loss_fn)

    def test_runner_result_with_new_fields(self):
        """Test handling of RunnerResult with potentially new fields."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Create RunnerResult with standard fields plus potential new ones
        mock_result = RunnerResult(
            answer="test answer",
            step_history=[],
            error=None,
            id="test_id_123"
        )
        
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        # Should handle RunnerResult regardless of additional fields
        self.assertEqual(eval_fn, trainer.eval_fn)
        self.assertEqual(eval_kwargs["y"], "test answer")
        self.assertEqual(eval_kwargs["y_gt"], "New runner works")

    def test_parameter_handling_with_new_runner_output(self):
        """Test Parameter creation with new Runner output types."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test with OutputParameter (which is now potentially returned by new Runner)
        from adalflow.optim.parameter import OutputParameter
        
        # Create a mock RunnerResult
        mock_result = RunnerResult(answer="output parameter test", step_history=[])
        
        # Create an OutputParameter wrapping the result (as new Runner might return)
        mock_output_param = OutputParameter(
            name="runner_output",
            data=mock_result,
            requires_opt=True
        )
        
        # Test prepare_loss can handle OutputParameter input
        loss_fn, loss_kwargs = trainer.prepare_loss(self.test_sample, mock_output_param)
        
        self.assertEqual(loss_fn, trainer.loss_fn)
        self.assertIn("kwargs", loss_kwargs)
        self.assertEqual(loss_kwargs["id"], "new_feature_test")
        
        # Check eval_input is set correctly for OutputParameter
        self.assertEqual(mock_output_param.eval_input, "output parameter test")

    def test_runner_trainer_initialization_with_new_runner_kwargs(self):
        """Test RunnerTrainer handles any new Runner constructor parameters gracefully."""
        # Test with various configurations that might be used with new Runner
        configs = [
            {"training": True},
            {"conversation_memory": None},
            {"permission_manager": None},
            {"ctx": {"test": "context"}},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                # Mock runner should accept any additional parameters
                mock_runner = Mock(spec=adal.Component)
                for key, value in config.items():
                    setattr(mock_runner, key, value)
                
                trainer = RunnerTrainer(runner=mock_runner)
                
                # Should initialize successfully regardless of runner's additional features
                self.assertIsNotNone(trainer.task)
                self.assertIsNotNone(trainer.eval_fn)
                self.assertIsNotNone(trainer.loss_fn)

    def test_step_history_handling_with_enhanced_runner(self):
        """Test that step history handling works with potentially enhanced Runner."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Create mock step history with potentially new StepOutput format
        mock_step = Mock()
        mock_step.step = 1
        mock_step.function = Mock()
        mock_step.function.name = "test_function"
        mock_step.observation = "test observation"
        
        mock_result = RunnerResult(
            answer="final answer",
            step_history=[mock_step],
            error=None
        )
        
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        # Should extract answer correctly regardless of step history format
        self.assertEqual(eval_kwargs["y"], "final answer")

    def test_error_handling_with_new_runner_exceptions(self):
        """Test error handling with potentially new Runner exception types."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test with RunnerResult containing error information
        mock_result = RunnerResult(
            answer=None,
            step_history=[],
            error="New runner error type"
        )
        
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, mock_result)
        
        # Should handle error cases gracefully
        self.assertEqual(eval_kwargs["y"], "")  # Empty string for None answer
        self.assertEqual(eval_kwargs["y_gt"], "New runner works")

    def test_context_preservation_with_new_runner(self):
        """Test that context and metadata are preserved with new Runner features."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test prepare_task with additional context that new Runner might use
        sample_with_metadata = Example(
            id="context_test",
            question="Test with context?",
            answer="Context preserved"
        )
        
        task_fn, kwargs = trainer.prepare_task(sample_with_metadata)
        
        # Should preserve all necessary information for new Runner
        self.assertEqual(task_fn, self.mock_runner.__call__)
        self.assertIn("prompt_kwargs", kwargs)
        self.assertEqual(kwargs["prompt_kwargs"]["input_str"], "Test with context?")
        self.assertEqual(kwargs["id"], "context_test")

    def test_backward_compatibility_with_legacy_runner_features(self):
        """Test that RunnerTrainer maintains backward compatibility."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test with "old style" runner result
        old_style_result = RunnerResult(answer="legacy answer", step_history=[])
        
        eval_fn, eval_kwargs = trainer.prepare_eval(self.test_sample, old_style_result)
        
        # Should work exactly as before
        eval_result = eval_fn(**eval_kwargs)
        # This should fail since answers don't match, proving evaluation works
        self.assertEqual(eval_result, 0.0)  # "legacy answer" != "New runner works"

    def test_concurrent_runner_execution_support(self):
        """Test that RunnerTrainer supports potential concurrent execution features."""
        trainer = RunnerTrainer(runner=self.mock_runner)
        
        # Test multiple samples in sequence (simulating batch processing)
        samples = [
            Example(id=f"concurrent_{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(3)
        ]
        
        results = []
        for sample in samples:
            mock_result = RunnerResult(answer=f"Answer {sample.id.split('_')[1]}", step_history=[])
            eval_fn, eval_kwargs = trainer.prepare_eval(sample, mock_result)
            result = eval_fn(**eval_kwargs)
            results.append(result)
        
        # All should evaluate correctly
        self.assertEqual(results, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()