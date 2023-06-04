import copy
import functools
import logging
import multiprocessing
import os
import random
import time
from typing import Optional, Sequence

import numpy as np
import tqdm
import anthropic

from .. import constants, utils

__all__ = ["anthropic_completions"]


def anthropic_completions(
        prompts: Sequence[str],
        model_name="claude-v1",
        num_procs: int = constants.ANTHROPIC_MAX_CONCURRENCY,
        **decoding_kwargs,
) -> dict[str, list]:
    """Decode with Anthropic API.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model to use for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    decoding_kwargs :
        Additional kwargs to pass to `anthropic.Client.completion`.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `anthropic_completions` on {n_examples} prompts using {model_name}.")

    kwargs = dict(model=model_name, **decoding_kwargs)
    logging.info(f"Kwargs to completion: {kwargs}")
    with utils.Timer() as t:
        if num_procs == 1:
            completions = [
                _anthropic_completion_helper(prompt, **kwargs) for prompt in tqdm.tqdm(prompts, desc="prompts")
            ]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_anthropic_completion_helper, **kwargs)
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompts),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    # anthropic doesn't return total tokens but 1 token approx 4 chars
    price = [
        len(prompt) / 4 * _get_price_per_token(model_name)
        for prompt in prompts
    ]

    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def _anthropic_completion_helper(
        prompt: str,
        sleep_time: int = 2,
        anthropic_api_keys: Optional[Sequence[str]] = (constants.ANTHROPIC_API_KEY,),
        max_tokens_to_sample: Optional[int] = 1000,
        temperature: Optional[float] = 0.7,
        n_retries: Optional[int] = 3,
        **kwargs,
) -> str:
    anthropic_api_key = random.choice(anthropic_api_keys)
    client = anthropic.Client(anthropic_api_key)

    kwargs.update(dict(max_tokens_to_sample=max_tokens_to_sample, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)
    while True:
        try:
            response = client.completion(prompt=prompt, **curr_kwargs)

            if response["completion"] == "":
                response["completion"] = " "  # annoying doesn't allow empty string

            break
        except anthropic.api.ApiException as e:
            logging.warning(f"ApiException: {e}.")
            if "status code: 429" in str(e):
                if len(anthropic_api_keys) > 1:
                    anthropic_api_key = random.choice(anthropic_api_keys)
                    client = anthropic.Client(anthropic_api_key)
                    logging.info(f"Switching anthropic API key.")
                logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
            elif "exceeds max" in str(e):
                curr_kwargs["max_tokens_to_sample"] = int(curr_kwargs["max_tokens_to_sample"] * 0.8)
                if curr_kwargs["max_tokens_to_sample"] == 0:
                    raise e
                logging.warning(f"Reducing target length to {curr_kwargs['max_tokens_to_sample']}, Retrying...")
            else:
                raise ValueError(f"Unknown ApiException {e}.")
        except Exception as e:
            if n_retries > 0:
                logging.warning(f"{e}. \nRetrying...")
                n_retries = n_retries - 1
            else:
                raise e

    return response["completion"]


def _get_price_per_token(model):
    """Returns the price per token for a given model"""
    # https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf
    if "claude-v1" in model:
        return (
                11.02 / 1e6
        )  # that's not completely true because decoding is 32.68 but close enough given that most is context
    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan
