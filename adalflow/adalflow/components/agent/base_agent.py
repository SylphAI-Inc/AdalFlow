"""Base agent implementation with standardized interfaces."""

from typing import List, Union, Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from adalflow.core.base_data_class import DataClass
import logging

from adalflow.core.func_tool import FunctionTool, AsyncCallable
from adalflow.core.component import Component
from adalflow.core.types import (
    Function,
)
from adalflow.core.model_client import ModelClient
from adalflow.utils.logger import printc

log = logging.getLogger(__name__)


@dataclass
class Step(DataClass):
    """Standardized step structure for all agents."""

    step_number: int = field(metadata={"desc": "The step number"})
    action: Optional[Function] = field(
        default=None, metadata={"desc": "The action taken in this step"}
    )
    observation: Any = field(
        default=None, metadata={"desc": "The observation from this step"}
    )
    metadata: Dict = field(
        default_factory=dict, metadata={"desc": "Additional metadata for this step"}
    )


@dataclass
class AgentOutput(DataClass):
    """Standardized output structure for all agents."""

    id: Optional[str] = field(
        default=None, metadata={"desc": "The unique id of the output"}
    )
    step_history: List[Step] = field(
        metadata={"desc": "The history of steps."}, default_factory=list
    )
    answer: Any = field(metadata={"desc": "The final answer."}, default=None)
    metadata: Dict = field(
        default_factory=dict, metadata={"desc": "Additional metadata"}
    )

    def validate(self) -> bool:
        """Validate the output structure."""
        if not isinstance(self.step_history, list):
            return False
        if not all(isinstance(step, Step) for step in self.step_history):
            return False
        return True


class BasePlanner(Component):
    """Base interface for planning strategies."""

    def __init__(self, model_client: ModelClient, model_kwargs: Dict = {}):
        super().__init__()
        self.model_client = model_client
        self.model_kwargs = model_kwargs

    def plan(self, input: str, context: Dict) -> Function:
        """Plan the next action based on input and context."""
        raise NotImplementedError


class BaseToolManager(Component):
    """Base interface for tool management."""

    def __init__(self, tools: List[Union[Callable, AsyncCallable, FunctionTool]]):
        super().__init__()
        self.tools = tools

    def execute(self, action: Function) -> Any:
        """Execute an action using the appropriate tool."""
        raise NotImplementedError


class BaseMemory(Component):
    """Base interface for memory management."""

    def __init__(self):
        super().__init__()
        self.steps: List[Step] = []

    def store(self, step: Step) -> None:
        """Store a step in memory."""
        self.steps.append(step)

    def retrieve(self, query: str) -> List[Step]:
        """Retrieve relevant steps from memory."""
        raise NotImplementedError


class BaseAgent(Component):
    """Base agent class with standardized interfaces."""

    def __init__(
        self,
        planner: BasePlanner,
        tool_manager: BaseToolManager,
        memory: Optional[BaseMemory] = None,
        max_steps: int = 10,
        context_variables: Optional[Dict] = None,
        use_cache: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.planner = planner
        self.tool_manager = tool_manager
        self.memory = memory
        self.max_steps = max_steps
        self.context_variables = context_variables
        self.use_cache = use_cache
        self.debug = debug

    def _handle_training(self, step: Step) -> Step:
        """Handle training mode specific logic."""
        if not self.training:
            return step
        # Add training specific logic here
        return step

    def _handle_evaluation(self, step: Step) -> Step:
        """Handle evaluation mode specific logic."""
        return step

    def _format_output(self, step_history: List[Step], answer: Any) -> AgentOutput:
        """Format the final output."""
        return AgentOutput(
            step_history=step_history, answer=answer, metadata=self._get_metadata()
        )

    def _get_metadata(self) -> Dict:
        """Get metadata for the output."""
        return {
            "max_steps": self.max_steps,
            "use_cache": self.use_cache,
            "context_variables": self.context_variables,
        }

    def _run_one_step(
        self,
        step_number: int,
        input: str,
        context: Dict,
        step_history: List[Step] = [],
    ) -> Step:
        """Run one step of the agent."""
        if self.debug:
            printc(f"Running step {step_number}", color="yellow")

        # Plan the next action
        action = self.planner.plan(input, context)

        # Execute the action
        observation = self.tool_manager.execute(action)

        # Create step
        step = Step(
            step_number=step_number,
            action=action,
            observation=observation,
            metadata={"context": context},
        )

        # Handle training/evaluation mode
        if self.training:
            step = self._handle_training(step)
        else:
            step = self._handle_evaluation(step)

        # Store in memory if available
        if self.memory:
            self.memory.store(step)

        return step

    def call(self, input: str, **kwargs) -> AgentOutput:
        """Main entry point for the agent."""
        step_history: List[Step] = []
        context = {
            "input": input,
            "step_history": step_history,
            **(self.context_variables or {}),
            **kwargs,
        }

        for step_number in range(1, self.max_steps + 1):
            step = self._run_one_step(
                step_number=step_number,
                input=input,
                context=context,
                step_history=step_history,
            )
            step_history.append(step)

            # Check if we should stop
            if self._should_stop(step):
                break

        # Get final answer
        answer = self._get_answer(step_history)

        # Format and return output
        output = self._format_output(step_history, answer)
        if not output.validate():
            raise ValueError("Invalid output format")

        return output

    def _should_stop(self, step: Step) -> bool:
        """Determine if the agent should stop."""
        raise NotImplementedError

    def _get_answer(self, step_history: List[Step]) -> Any:
        """Get the final answer from step history."""
        raise NotImplementedError

    def train_step(self, input: str, target: Any) -> Dict:
        """Standard training step interface."""
        raise NotImplementedError

    def eval_step(self, input: str) -> AgentOutput:
        """Standard evaluation step interface."""
        raise NotImplementedError
