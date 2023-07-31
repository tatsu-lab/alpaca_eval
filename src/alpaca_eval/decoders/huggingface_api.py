import functools
import logging
import multiprocessing
import time
from typing import Sequence

import numpy as np
import tqdm
from huggingface_hub.inference_api import InferenceApi

from .. import constants, utils

__all__ = ["huggingface_api_completions"]


def huggingface_api_completions(
    prompts: Sequence[str],
    model_name: str,
    gpu: bool = False,
    do_sample: bool = False,
    num_procs: int = 1,
    **kwargs,
) -> dict[str, list]:
    """Decode with the API from hugging face hub.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    gpu : bool, optional
        Whether to use GPU for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    kwargs :
        Additional kwargs to pass to `InferenceApi.__call__`.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `huggingface_api_completions` on {n_examples} prompts using {model_name}.")

    inference = InferenceApi(
        model_name,
        task="text-generation",
        token=constants.HUGGINGFACEHUB_API_TOKEN,
        gpu=gpu,
    )

    default_kwargs = dict(do_sample=do_sample, options=dict(wait_for_model=True), return_full_text=False)
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")

    with utils.Timer() as t:
        partial_completion_helper = functools.partial(inference_helper, inference=inference, params=default_kwargs)
        if num_procs == 1:
            completions = [partial_completion_helper(prompt) for prompt in tqdm.tqdm(prompts, desc="prompts")]
        else:
            with multiprocessing.Pool(num_procs) as p:
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompts),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Time for {n_examples} completions: {t}")

    completions = [completion["generated_text"] for completion in completions]

    # unclear pricing
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def inference_helper(prompt: str, inference, params, n_retries=100, waiting_time=2) -> dict:
    for _ in range(n_retries):
        output = inference(inputs=prompt, params=params)
        if "error" in output and n_retries > 0:
            error = output["error"]
            if "Rate limit reached" in output["error"]:
                logging.warning(f"Rate limit reached... Trying again in {waiting_time} seconds. Full error: {error}")
                time.sleep(waiting_time)
            elif "Input validation error" in error and "max_new_tokens" in error:
                params["max_new_tokens"] = int(params["max_new_tokens"] * 0.8)
                logging.warning(
                    f"`max_new_tokens` too large. Reducing target length to {params['max_new_tokens']}, " f"Retrying..."
                )
                if params["max_new_tokens"] == 0:
                    raise ValueError(f"Error in inference. Full error: {error}")
            else:
                raise ValueError(f"Error in inference. Full error: {error}")
        else:
            return output[0]
    raise ValueError(f"Error in inference. We tried {n_retries} times and failed.")
