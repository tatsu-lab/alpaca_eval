import functools
import logging
import multiprocessing
import os
import sys
from typing import Optional, Sequence
from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
import transformers
import accelerate
import tqdm

__all__ = ["huggingface_local_completions"]


def huggingface_local_completions(
    prompts: Sequence[str],
    model_name: str,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs={"load_in_8bit": True},
    **kwargs,
) -> str:
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(
            f"Using `huggingface_local_completions` on {n_examples} prompts using {model_name}."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))

    default_kwargs = dict(
        do_sample=do_sample, model_kwargs=model_kwargs, device_map = "auto", batch_size=batch_size
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        **default_kwargs
    )

    completions = pipeline(prompts, return_full_text=False)
    completions = [completion[0]["generated_text"] for completion in completions]

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))


    return completions
