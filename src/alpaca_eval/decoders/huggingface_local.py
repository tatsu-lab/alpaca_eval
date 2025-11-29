import gc
import logging
from typing import Optional, Sequence

import numpy as np
import torch
import transformers
from peft import PeftModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import constants, utils

__all__ = ["huggingface_local_completions"]

# cache
_loaded_model = None
_loaded_tokenizer = None
_loaded_model_name = None
_loaded_adapters_name = None


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def _get_or_load_model(
    model_name: str,
    model_kwargs: dict,
    cache_dir: Optional[str],
    is_fast_tokenizer: bool,
    adapters_name: Optional[str],
    batch_size: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Caches models at the module level to avoid reloading between chunks
    which prevents OOM errors when processing large datasets in chunks.

    Parameters
    ----------
    model_name : str
        Name of the model to load.
    model_kwargs : dict
        Additional kwargs to pass to from_pretrained.
    cache_dir : str, optional
        Directory to use for caching the model.
    is_fast_tokenizer : bool
        Whether to use fast tokenizer.
    adapters_name : str, optional
        Name of adapters to merge if using PEFT.
    batch_size : int
        Batch size (affects whether to use bettertransformer).

    Returns
    -------
    model : AutoModelForCausalLM
        The loaded or cached model.
    tokenizer : AutoTokenizer
        The loaded or cached tokenizer.
    """
    global _loaded_model, _loaded_tokenizer, _loaded_model_name, _loaded_adapters_name

    need_reload = _loaded_model is None or _loaded_model_name != model_name or _loaded_adapters_name != adapters_name
    if need_reload:
        # unload previous model if switching
        if _loaded_model is not None:
            logging.info(
                f"Unloading previous model: {_loaded_model_name} "
                f"(adapters: {_loaded_adapters_name}) to load {model_name}"
            )
            del _loaded_model
            del _loaded_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side="left",
            use_fast=is_fast_tokenizer,
            **model_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs).eval()

        if adapters_name:
            logging.info(f"Merging adapter from {adapters_name}.")
            model = PeftModel.from_pretrained(model, adapters_name)
            model = model.merge_and_unload()

        if batch_size == 1:
            try:
                model = model.to_bettertransformer()
            except:
                # could be not implemented or natively supported
                pass

        # cache the loaded model
        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _loaded_model_name = model_name
        _loaded_adapters_name = adapters_name
    else:
        logging.info(f"Reusing cached model: {model_name}")
        model = _loaded_model
        tokenizer = _loaded_tokenizer

    return model, tokenizer


def huggingface_local_completions(
    prompts: Sequence[str],
    model_name: str,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs=None,
    cache_dir: Optional[str] = constants.DEFAULT_CACHE_DIR,
    remove_ending: Optional[str] = None,
    is_fast_tokenizer: bool = True,
    adapters_name: Optional[str] = None,
    **kwargs,
) -> dict[str, list]:
    """Decode locally using huggingface transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    batch_size : int, optional
        Batch size to use for decoding. This currently does not work well with to_bettertransformer.

    model_kwargs : dict, optional
        Additional kwargs to pass to from_pretrained.

    cache_dir : str, optional
        Directory to use for caching the model.

    remove_ending : str, optional
        The ending string to be removed from completions. Typically eos_token.

    kwargs :
        Additional kwargs to pass to `InferenceClient.__call__`.
    """
    model_kwargs = model_kwargs or {}
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
    if "torch_dtype" in model_kwargs and isinstance(model_kwargs["torch_dtype"], str):
        model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `huggingface_local_completions` on {n_examples} prompts using {model_name}.")

    if not torch.cuda.is_available():
        model_kwargs["load_in_8bit"] = False
        model_kwargs["torch_dtype"] = None

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = _get_or_load_model(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cache_dir=cache_dir,
        is_fast_tokenizer=is_fast_tokenizer,
        adapters_name=adapters_name,
        batch_size=batch_size,
    )

    logging.info(f"Model memory: {model.get_memory_footprint() / 1e9} GB")

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1]), reverse=True))
        prompts = list(prompts)

    if not tokenizer.pad_token_id:
        # set padding token if not set
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs={k: v for k, v in model_kwargs.items() if k != "trust_remote_code"},
        batch_size=batch_size,
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **default_kwargs,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )

    ## compute and log the time for completions
    prompts_dataset = ListDataset(prompts)
    completions = []

    with utils.Timer() as t:
        for out in tqdm(
            pipeline(
                prompts_dataset,
                return_full_text=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        ):
            generated_text = out[0]["generated_text"]
            if remove_ending is not None and generated_text.endswith(remove_ending):
                generated_text = generated_text[: -len(remove_ending)]
            completions.append(generated_text)

    logging.info(f"Time for {n_examples} completions: {t}")

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    # local => price is really your compute
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
