import logging
import multiprocessing
from functools import partial
from typing import Sequence
import requests
import json
import os
import time
from .openai import _prompt_to_chatml
from .. import utils

__all__ = ["jina_chat_completions"]


def jina_chat_completions(
    prompts: Sequence[str],
) -> dict[str, list]:
    """Get jina chat completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.
    """
    n_examples = len(prompts)
    api_key = os.environ.get('JINA_CHAT_API_KEY')

    if n_examples == 0:
        logging.info("No samples to annotate.")
        return {}
    else:
        logging.info(f"Using `jina_chat_completions` on {n_examples} prompts.")

    prompts = [_prompt_to_chatml(prompt.strip()) for prompt in prompts]
    num_processes = min(multiprocessing.cpu_count(), 4)
    with utils.Timer() as t:
        with multiprocessing.Pool(processes=num_processes) as pool:
            print(f"Number of processes: {pool._processes}")
            get_chat_completion_with_key = partial(_get_chat_completion, api_key)
            completions = pool.map(get_chat_completion_with_key, prompts)

    logging.info(f"Completed {n_examples} examples in {t}.")

    # refer to https://chat.jina.ai/billing
    price = 0
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def _get_chat_completion(api_key, prompt):
    url = 'https://api.chat.jina.ai/v1/chat/completions'
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json"
    }
    json_payload = {"messages": prompt}

    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_payload)
            response.raise_for_status()  # Will raise an HTTPError if one occurred.
            return response.json()['choices'][0]['message']['content']
        except (json.JSONDecodeError, requests.exceptions.HTTPError) as e:
            print(f"Error occurred: {e}, Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)
            if attempt + 1 == max_retries:
                print("Max retries reached. Raising exception.")
                print(f"Request data -> URL: {url}, Headers: {headers}, JSON Payload: {json_payload}")
                raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
