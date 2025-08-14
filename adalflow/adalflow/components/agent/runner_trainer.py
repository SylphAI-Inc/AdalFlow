from re import A
from typing import Any, Callable, Dict, Optional, Tuple, List
import adalflow as adal
from adalflow.datasets.types import GSM8KData as Example
from adalflow.datasets.gsm8k import GSM8K
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.core.types import RunnerResult
from adalflow.optim.optimizer import Optimizer
from adalflow.components.agent.react import ReActAgent
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.core.func_tool import FunctionTool


def load_datasets():
    train_data = GSM8K(split="train", size=100)
    val_data = GSM8K(split="val", size=50)
    test_data = GSM8K(split="test", size=100)
    return train_data, val_data, test_data


def default_eval_fn(y: str, y_gt: str) -> float:
    """Default dummy evaluation function that performs exact string match."""
    return 1.0 if str(y).strip() == str(y_gt).strip() else 0.0


def default_loss_fn_factory(eval_fn: Callable = None):
    """Factory function to create default loss function."""
    if eval_fn is None:
        eval_fn = default_eval_fn
    return adal.EvalFnToTextLoss(
        eval_fn=eval_fn,
        eval_fn_desc="Default evaluation: 1 if str(y) == str(y_gt) else 0",
    )


class RunnerTrainer(adal.AdalComponent):
    """Generic trainer that accepts a Runner model and allows custom eval_fn and loss_fn."""

    def __init__(
        self,
        runner: adal.Component,
        eval_fn: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
        backward_engine_model_config: Optional[Dict] = None,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None,
        original_react_agent: Optional[ReActAgent] = False, 
    ):
        # Use provided eval_fn or create default
        if eval_fn is None:
            eval_fn = default_eval_fn
        
        # Use provided loss_fn or create default
        if loss_fn is None:
            loss_fn = default_loss_fn_factory(eval_fn)
        
        super().__init__(task=runner, eval_fn=eval_fn, loss_fn=loss_fn)

        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config
        self.original_react_agent = original_react_agent

    def prepare_task(self, sample: Example) -> Tuple[Callable, Dict[str, Any]]:
        # Runner.call expects prompt_kwargs parameter, not direct keyword args
        # prepare task should not return in training mode
        # return self.task.forward, {"prompt_kwargs": {"input_str": sample.question}, "id": sample.id} 
        if self.original_react_agent:
            return self.task.__call__, {"input": sample.question, "id": sample.id}
        return self.task.__call__, {"prompt_kwargs": {"input_str": sample.question}, "id": sample.id}

    def prepare_eval(
        self, sample: Example, y_pred: RunnerResult
    ) -> Tuple[float, Dict[str, Any]]:
        y_label = ""

        if not self.original_react_agent and y_pred is not None and y_pred.answer is not None:
            y_label = y_pred.answer
        elif self.original_react_agent:
            y_label = y_pred
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(
        self, sample: Example, pred: adal.Parameter
    ) -> Tuple[Callable, Dict[str, Any]]:
        print("pred", pred)
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )

        if self.original_react_agent:
            pred.eval_input = pred.data
        else: 
            pred.eval_input = pred.data.answer

        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}


def train_original_react():
    from adalflow.utils import setup_env

    setup_env() 
    train_data, val_data, test_data = load_datasets()
    
    # Create Anthropic client configuration instead of GPT-3
    anthropic_config = {
        "model_client": adal.AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-3-5-sonnet-20241022",
            # "max_tokens": 2000,
            "temperature": 0.0,
        }
    }
    
    def llm_as_tool(input: str, id: Optional[str] = None) -> str:
        """Used as a calculator tool."""
        printc(f"llm_as_tool: {input}", color="yellow")

        return self.llm_tool(prompt_kwargs={"input_str": input}, id=id)

    react_agent = ReActAgent(
        tools=[FunctionTool(llm_as_tool)],
        max_steps=6,
        add_llm_as_fallback=True,
        **anthropic_config
    )
    
    # Create specific eval_fn and loss_fn like in GSM8K train.py
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    loss_fn = adal.EvalFnToTextLoss(
        eval_fn=eval_fn,
        eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
    )
    
    adal_component = RunnerTrainer(
        runner=react_agent,
        eval_fn=eval_fn,
        loss_fn=loss_fn,
        backward_engine_model_config=anthropic_config,
        text_optimizer_model_config=anthropic_config,
        original_react_agent=True,
    )
    trainer = adal.Trainer(
        adaltask=adal_component,
        strategy="random",
        max_steps=10,
        text_optimizers_config_kwargs={"max_past_history": 5},
    )
    trainer.fit(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        debug=False, 
    )

def train():
    from adalflow.utils import setup_env

    setup_env() 
    train_data, val_data, test_data = load_datasets()
    
    # Create Anthropic client configuration instead of GPT-3
    anthropic_config = {
        "model_client": adal.AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-3-5-sonnet-20241022",
            # "max_tokens": 2000,
            "temperature": 0.0,
        }
    }
    
    # Create Agent first, then Runner  
    agent = adal.Agent(
        name="gsm8k_agent",
        add_llm_as_fallback=True,
        max_steps = 6, 
        **anthropic_config
    )
    
    runner = adal.Runner(agent=agent, training=True)
    
    # Create specific eval_fn and loss_fn like in GSM8K train.py
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    loss_fn = adal.EvalFnToTextLoss(
        eval_fn=eval_fn,
        eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
    )
    
    adal_component = RunnerTrainer(
        runner=runner,
        eval_fn=eval_fn,
        loss_fn=loss_fn,
        backward_engine_model_config=anthropic_config,
        text_optimizer_model_config=anthropic_config,
    )
    trainer = adal.Trainer(
        adaltask=adal_component,
        strategy="random",
        max_steps=10,
        # resume_from_ckpt="./Users/jinhakim/.adalflow/ckpt/RunnerTrainer/random_max_steps_5_2bd11_run_1.json",
        text_optimizers_config_kwargs={"max_past_history": 5},
    )
    trainer.fit(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        debug=False, 
        # resume_from_ckpt="./Users/jinhakim/.adalflow/ckpt/RunnerTrainer/random_max_steps_2_cc5fb_run_1.json",
    )

    # feedback, visualizatio n


if __name__ == "__main__":
    # train()
    train_original_react()