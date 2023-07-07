import copy
import functools
import logging
import math
import multiprocessing
import os
import random
from typing import Optional, Sequence

import cohere
import tqdm
from cohere import CohereError

from .. import constants, utils

__all__ = ["cohere_completions"]


def cohere_completions(
    prompts: Sequence[str],
    model_name="command",
    mode="instruct",
    num_procs: int = 5,
    **decoding_kwargs,
) -> dict[str, list]:
    """Decode with Cohere API.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model to use for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    decoding_kwargs :
        Additional kwargs to pass to `cohere.Client.generation`.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `cohere_completions` on {n_examples} prompts using {model_name}.")

    kwargs = dict(model=model_name, mode=mode, **decoding_kwargs)
    logging.info(f"Kwargs to completion: {kwargs}")

    with utils.Timer() as t:
        if num_procs == 1:
            completions = [_cohere_completion_helper(prompt, **kwargs) for prompt in tqdm.tqdm(prompts, desc="prompts")]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_cohere_completion_helper, **kwargs)
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompts),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    # cohere charges $2.5 for every 1000 call to API that is less than 1000 characters. Only counting prompts here
    price = [2.5 / 1000 * math.ceil(len(prompt) / 1000) for prompt in prompts]
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def _cohere_completion_helper(
    prompt: str,
    cohere_api_keys: Optional[Sequence[str]] = (constants.COHERE_API_KEY,),
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.7,
    max_tries=5,
    mode="instruct",
    **kwargs,
) -> str:
    cohere_api_key = random.choice(cohere_api_keys)
    client = cohere.Client(cohere_api_key)

    kwargs.update(dict(max_tokens=max_tokens, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    for trynum in range(max_tries):  # retry errors
        try:
            if mode == "instruct":
                response = client.generate(prompt=prompt, **curr_kwargs)
                text = response[0].text
            elif mode == "chat":
                response = client.chat(prompt, **curr_kwargs)
                text = response.text
            else:
                raise ValueError(f"Invalid mode {mode} for cohere_completions")

            if text == "":
                raise CohereError("Empty string response")

            return text

        except CohereError as e:
            print(f"Try #{trynum+1}/{max_tries}: Error running prompt {repr(prompt)}: {e}")

    return " "  # placeholder response for errors, doesn't allow empty string
