import functools
import logging
import multiprocessing
import os
import sys
from typing import Optional, Sequence

import torch
from langchain import HuggingFacePipeline
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
        cache_dir: Optional[str] = None,
        **kwargs,
):

    llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation", cache_dir=cache_dir,
                                            model_kwargs=model_kwargs, **kwargs)

    breakpoint()
    out = llm.generate(prompts)


def huggingface_local2_completions(
        prompts: Sequence[str],
        model_name: str,
        do_sample: bool = False,
        batch_size: int = 1,
        model_kwargs={"load_in_8bit": True, "device_map": "auto"},
        cache_dir: Optional[str] = None,
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

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left", **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs).to_bettertransformer()

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        # device_map doesn't seem to work well. It's very slow and I don't see GPU usage
        device_map="auto"
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    pipeline = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, **default_kwargs)

    breakpoint()
    completions = pipeline(prompts, return_full_text=False)
    completions = [completion[0]["generated_text"] for completion in completions]

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    return completions


def huggingface_local_completions(
        prompts: Sequence[str],
        model_name: str,
        do_sample: bool = False,
        batch_size: int = 1,
        model_kwargs={"load_in_8bit": True},
        cache_dir: Optional[str] = None,
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

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to_bettertransformer()

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        # device_map doesn't seem to work well. It's very slow and I don't see GPU usage
        device_map="auto"
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **default_kwargs
    )

    breakpoint()
    completions = pipeline(prompts, return_full_text=False)
    completions = [completion[0]["generated_text"] for completion in completions]

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    return completions
