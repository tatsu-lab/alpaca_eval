import copy
import functools
import logging
import multiprocessing
import os
import random
from typing import Sequence
from huggingface_hub.inference_api import InferenceApi

import tqdm

__all__ = ["huggingface_completions"]


def huggingface_completions(
    prompts: Sequence[str],
    model_name: str,
    gpu: bool = False,
    do_sample: bool = False,
    **kwargs,
) -> str:
    from langchain import HuggingFaceHub
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(
            f"Using `huggingface_completions` on {n_examples} prompts using {model_name}."
        )

    llm = HuggingFaceHub(repo_id=model_name, task="text-generation", model_kwargs=kwargs)
    completions = llm.generate(prompts)
    breakpoint()

    completions = [completion[0].text for completion in completions.generations[0]]

    return completions
