from typing import Dict

# TODO: import all other  functions into this single file to be exposed to users


def compose_model_kwargs(default_model_kwargs: Dict, model_kwargs: Dict) -> Dict:
    r"""
    The model configuration exclude the input itself.
    Combine the default model, model_kwargs with the passed model_kwargs.
    Example:
    model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
    self.model_kwargs = {"model": "gpt-3.5"}
    combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

    """
    pass_model_kwargs = default_model_kwargs.copy()

    if model_kwargs:
        pass_model_kwargs.update(model_kwargs)
    return pass_model_kwargs
