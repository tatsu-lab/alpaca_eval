import logging
from typing import Optional, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import transformers
from .. import utils

__all__ = ["huggingface_local_completions"]


def huggingface_local_completions(
        prompts: Sequence[str],
        model_name: str,
        do_sample: bool = False,
        batch_size: int = 1,
        model_kwargs={  # "load_in_8bit": True, # divides memory by 2 but is slower
            "device_map": "auto",
            "torch_dtype": torch.float16},
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

    if not torch.cuda.is_available():
        model_kwargs["load_in_8bit"] = False
        model_kwargs["torch_dtype"] = None

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left", **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=cache_dir,
                                                 **model_kwargs)
    try:
        model = model.to_bettertransformer()
    except NotImplementedError:
        pass
    
    logging.info(f"Model memory: {model.get_memory_footprint() / 1e9} GB")

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = transformers.pipeline(task="text-generation",
                                     model=model,
                                     tokenizer=tokenizer,
                                     **default_kwargs)

    ## compute and log the time for completions
    with utils.Timer() as t:
        completions = pipeline(prompts, return_full_text=False, pad_token_id=pipeline.tokenizer.eos_token_id)
    logging.info(f"Time for {n_examples} completions: {t}")
    completions = [completion[0]["generated_text"] for completion in completions]

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    return completions
