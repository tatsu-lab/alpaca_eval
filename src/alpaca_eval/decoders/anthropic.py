import copy
import functools
import logging
import multiprocessing
import random
import time
from typing import Optional, Sequence, Union

import anthropic
import numpy as np
import tqdm

from .. import constants, utils

__all__ = ["anthropic_completions"]


def anthropic_completions(
    prompts: Sequence[str],
    max_tokens_to_sample: Union[int, Sequence[int]] = 2048,
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
        Additional kwargs to pass to `anthropic.Anthropic.create`.
    """
    num_procs = num_procs or constants.ANTHROPIC_MAX_CONCURRENCY

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        to_log = f"Using `anthropic_completions` on {n_examples} prompts using {model_name} and num_procs={num_procs}."
        logging.info(to_log)

    if isinstance(max_tokens_to_sample, int):
        max_tokens_to_sample = [max_tokens_to_sample] * n_examples

    inputs = zip(prompts, max_tokens_to_sample)

    kwargs = dict(model=model_name, **decoding_kwargs)
    kwargs_to_log = {k: v for k, v in kwargs.items() if "api_key" not in k}
    logging.info(f"Kwargs to completion: {kwargs_to_log}")
    with utils.Timer() as t:
        if num_procs == 1:
            responses = [_anthropic_completion_helper(inp, **kwargs) for inp in tqdm.tqdm(inputs, desc="prompts")]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_anthropic_completion_helper, **kwargs)
                responses = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, inputs),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    completions = [response.completion for response in responses]

    # anthropic doesn't return total tokens but 1 token approx 4 chars
    price = [len(prompt) / 4 * _get_price_per_token(model_name) for prompt in prompts]

    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time, completions_all=responses)


def _anthropic_completion_helper(
    args: tuple[str, int],
    sleep_time: int = 2,
    anthropic_api_keys: Optional[Sequence[str]] = (constants.ANTHROPIC_API_KEY,),
    temperature: Optional[float] = 0.7,
    n_retries: Optional[int] = 3,
    **kwargs,
):
    prompt, max_tokens = args

    anthropic_api_keys = anthropic_api_keys or (constants.ANTHROPIC_API_KEY,)
    anthropic_api_key = random.choice(anthropic_api_keys)

    if not utils.check_pkg_atleast_version("anthropic", "0.3.0"):
        raise ValueError("Anthropic version must be at least 0.3.0. Use `pip install -U anthropic`.")

    client = anthropic.Anthropic(api_key=anthropic_api_key, max_retries=n_retries)

    kwargs.update(dict(max_tokens_to_sample=max_tokens, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)
    while True:
        try:
            response = client.completions.create(prompt=prompt, **curr_kwargs)

            if response.completion == "":
                response.completion = " "  # annoying doesn't allow empty string

            break

        except anthropic.RateLimitError as e:
            logging.warning(f"API RateLimitError: {e}.")
            if len(anthropic_api_keys) > 1:
                anthropic_api_key = random.choice(anthropic_api_keys)
                client = anthropic.Anthropic(api_key=anthropic_api_key, max_retries=n_retries)
                logging.info(f"Switching anthropic API key.")
            logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)

        except anthropic.APITimeoutError as e:
            logging.warning(f"API TimeoutError: {e}. Retrying request.")

    return response


def _get_price_per_token(model):
    """Returns the price per token for a given model"""
    # https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf
    if "claude-v1" in model or "claude-2" in model:
        return (
            11.02 / 1e6
        )  # that's not completely true because decoding is 32.68 but close enough given that most is context
    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan
