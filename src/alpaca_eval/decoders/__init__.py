import logging
from typing import Callable, Union


def get_fn_completions(name: Union[str, Callable]) -> Callable:
    """Get a decoder by name."""
    if not isinstance(name, str):
        return name

    if name == "anthropic_completions":
        try:
            from .anthropic import anthropic_completions
        except ImportError:
            packages = ["anthropic"]
            logging.exception(f"You need {packages} to use anthropic_completions. Error:")

        return anthropic_completions

    elif name == "openai_completions":
        try:
            from .openai import openai_completions
        except ImportError:
            packages = ["openai"]
            logging.exception(f"You need {packages} to use openai_completions. Error:")

        return openai_completions

    elif name == "huggingface_api_completions":
        try:
            from .huggingface_api import huggingface_api_completions
        except ImportError:
            packages = ["huggingface_hub"]
            logging.exception(f"You need {packages} to use huggingface_api_completions. Error:")

        return huggingface_api_completions

    elif name == "huggingface_local_completions":
        try:
            from .huggingface_local import huggingface_local_completions
        except ImportError:
            packages = ["accelerate", "transformers", "bitsandbytes", "xformers", "peft", "optimum", "scipy"]
            logging.exception(f"You need {packages} to use huggingface_local_completions. Error:")

        return huggingface_local_completions

    elif name == "cohere_completions":
        try:
            from .cohere import cohere_completions
        except ImportError:
            packages = ["cohere"]
            logging.exception(f"You need {packages} to use cohere_completions. Error:")

        return cohere_completions

    else:
        raise ValueError(f"Unknown decoder: {name}")
