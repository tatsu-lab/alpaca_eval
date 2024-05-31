import logging
from typing import Sequence

import numpy as np
from vllm import LLM, SamplingParams

from .. import utils

__all__ = ["vllm_local_completions"]

llm = None
llmModelName = None


def vllm_local_completions(
    prompts: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    do_sample: bool = False,
    batch_size: int | None = None,  # default of vllm is 256
    model_kwargs=None,
    **decoding_kwargs,
) -> dict[str, list]:
    """Decode locally using vllm transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    max_new_tokens : int
        Maximum number of tokens to generate for each prompt.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    batch_size : int, optional
        Batch size to use for decoding. If None uses the default batch size of vllm.

    model_kwargs : dict, optional
        Additional kwargs to pass to `vllm.LLM` constructor.

    decoding_kwargs :
        Additional kwargs to SamplingParams
    """
    global llm, llmModelName
    model_kwargs = model_kwargs or {}

    if model_name != llmModelName:
        logging.info(f"vllm already loaded model: {llmModelName} but requested {model_name}. Let's switch...")
        llm = None

    if llm is None:
        logging.info(f"vllm: loading model: {model_name}, {model_kwargs}")
        llm = LLM(model=model_name, tokenizer=model_name, **model_kwargs)
        llmModelName = model_name

    logging.info(f"Sampling kwargs: {decoding_kwargs}")
    if batch_size is not None:
        decoding_kwargs["max_num_seqs"] = batch_size
    sampling_params = SamplingParams(max_tokens=max_new_tokens, **decoding_kwargs)
    if do_sample:
        sampling_params.use_beam_search = True
    with utils.Timer() as t:
        outputs = llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]
    price = [np.nan] * len(completions)
    avg_time = [t.duration / len(prompts)] * len(completions)
    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
