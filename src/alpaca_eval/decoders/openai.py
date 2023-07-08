import copy
import functools
import json
import logging
import math
import multiprocessing
import random
import time
from typing import Optional, Sequence

import numpy as np
import openai
import tiktoken
import tqdm

from .. import constants, utils

__all__ = ["openai_completions"]

DEFAULT_OPENAI_API_BASE = openai.api_base


def openai_completions(
    prompts: Sequence[str],
    model_name: str,
    tokens_to_favor: Optional[Sequence[str]] = None,
    tokens_to_avoid: Optional[Sequence[str]] = None,
    is_skip_multi_tokens_to_avoid: bool = True,
    is_strip: bool = True,
    num_procs: Optional[int] = None,
    batch_size: Optional[int] = None,
    **decoding_kwargs,
) -> dict[str, list]:
    r"""Get openai completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str
        Name of the model to use for decoding.

    tokens_to_favor : list of str, optional
        Substrings to favor in the completions. We will add a positive bias to the logits of the tokens constituting
        the substrings.

    tokens_to_avoid : list of str, optional
        Substrings to avoid in the completions. We will add a large negative bias to the logits of the tokens
        constituting the substrings.

    is_skip_multi_tokens_to_avoid : bool, optional
        Whether to skip substrings from tokens_to_avoid that are constituted by more than one token => avoid undesired
        side effects on other tokens.

    is_strip : bool, optional
        Whether to strip trailing and leading spaces from the prompts.

    decoding_kwargs :
        Additional kwargs to pass to `openai.Completion` or `openai.ChatCompletion`.

    Example
    -------
    >>> prompts = ["Respond with one digit: 1+1=", "Respond with one digit: 2+2="]
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_avoid=["2"," 2"])['completions']
    ['\n\nAnswer: \n\nTwo (or, alternatively, the number "two" or the numeral "two").', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_favor=["2"])['completions']
    ['2\n\n2', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003",
    ... tokens_to_avoid=["2 a long sentence that is not a token"])['completions']
    ['\n\n2', '\n\n4']
    >>> chat_prompt = ["<|im_start|>user\n1+1=<|im_end|>", "<|im_start|>user\nRespond with one digit: 2+2=<|im_end|>"]
    >>> openai_completions(chat_prompt, "gpt-3.5-turbo", tokens_to_avoid=["2"," 2"])['completions']
    ['As an AI language model, I can confirm that 1+1 equals  02 in octal numeral system, 10 in decimal numeral
    system, and  02 in hexadecimal numeral system.', '4']
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `openai_completions` on {n_examples} prompts using {model_name}.")

    if tokens_to_avoid or tokens_to_favor:
        tokenizer = tiktoken.encoding_for_model(model_name)

        logit_bias = decoding_kwargs.get("logit_bias", {})
        if tokens_to_avoid is not None:
            for t in tokens_to_avoid:
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 and is_skip_multi_tokens_to_avoid:
                    logging.warning(f"'{t}' has more than one token, skipping because `is_skip_multi_tokens_to_avoid`.")
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens

        if tokens_to_favor is not None:
            for t in tokens_to_favor:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = 7  # increase log prob of tokens to match

        decoding_kwargs["logit_bias"] = logit_bias

    if is_strip:
        prompts = [p.strip() for p in prompts]

    is_chat = decoding_kwargs.get("requires_chatml", _requires_chatml(model_name))
    if is_chat:
        prompts = [_prompt_to_chatml(prompt) for prompt in prompts]
        num_procs = num_procs or 4
        batch_size = batch_size or 1

        if batch_size > 1:
            logging.warning("batch_size > 1 is not supported yet for chat models. Setting to 1")
            batch_size = 1

    else:
        num_procs = num_procs or 1
        batch_size = batch_size or 10

    logging.info(f"Kwargs to completion: {decoding_kwargs}")
    n_batches = int(math.ceil(n_examples / batch_size))

    prompt_batches = [prompts[batch_id * batch_size : (batch_id + 1) * batch_size] for batch_id in range(n_batches)]

    kwargs = dict(n=1, model=model_name, is_chat=is_chat, **decoding_kwargs)
    logging.info(f"Kwargs to completion: {kwargs}")

    with utils.Timer() as t:
        if num_procs == 1:
            completions = [
                _openai_completion_helper(prompt_batch, **kwargs)
                for prompt_batch in tqdm.tqdm(prompt_batches, desc="prompt_batches")
            ]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_openai_completion_helper, **kwargs)
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompt_batches),
                        desc="prompt_batches",
                        total=len(prompt_batches),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    # flatten the list and select only the text
    completions_text = [completion.text for completion_batch in completions for completion in completion_batch]

    price = [
        completion["total_tokens"] * _get_price_per_token(model_name)
        for completion_batch in completions
        for completion in completion_batch
    ]
    avg_time = [t.duration / n_examples] * len(completions_text)

    return dict(completions=completions_text, price_per_example=price, time_per_example=avg_time)


def _openai_completion_helper(
    prompt_batch: Sequence[str],
    is_chat: bool,
    sleep_time: int = 2,
    openai_organization_ids: Optional[Sequence[str]] = constants.OPENAI_ORGANIZATION_IDS,
    openai_api_keys: Optional[Sequence[str]] = constants.OPENAI_API_KEYS,
    openai_api_base: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    top_p: Optional[float] = 1.0,
    temperature: Optional[float] = 0.7,
    **kwargs,
):
    # randomly select orgs
    if openai_organization_ids is not None:
        openai.organization = random.choice(openai_organization_ids)

    if openai_api_keys is not None:
        openai.api_key = random.choice(openai_api_keys)

    # set api base
    openai.api_base = openai_api_base if openai_api_base is not None else DEFAULT_OPENAI_API_BASE

    # copy shared_kwargs to avoid modifying it
    kwargs.update(dict(max_tokens=max_tokens, top_p=top_p, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    while True:
        try:
            if is_chat:
                completion_batch = openai.ChatCompletion.create(messages=prompt_batch[0], **curr_kwargs)

                choices = completion_batch.choices
                for choice in choices:
                    assert choice.message.role == "assistant"
                    if choice.message.content == "":
                        choice["text"] = " "  # annoying doesn't allow empty string
                    else:
                        choice["text"] = choice.message.content

                    if choice.message.get("function_call"):
                        # currently we only use function calls to get a JSON object
                        # => overwrite text with the JSON object. In the future, we could
                        # allow actual function calls
                        all_args = json.loads(choice.message.function_call.arguments)
                        assert len(all_args) == 1
                        choice["text"] = all_args[list(all_args.keys())[0]]

            else:
                completion_batch = openai.Completion.create(prompt=prompt_batch, **curr_kwargs)
                choices = completion_batch.choices

            for choice in choices:
                choice["total_tokens"] = completion_batch.usage.total_tokens / len(prompt_batch)
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                kwargs["max_tokens"] = int(kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {kwargs['max_tokens']}, Retrying...")
                if kwargs["max_tokens"] == 0:
                    logging.exception("Prompt is already longer than max context length. Error:")
                    raise e
            else:
                if "rate limit" in str(e).lower():
                    logging.warning("Hit request rate limit; retrying...")
                else:
                    logging.warning(f"Unknown error {e}. \n It's likely a rate limit so we are retrying...")
                if openai_organization_ids is not None and len(openai_organization_ids) > 1:
                    openai.organization = random.choice(
                        [o for o in openai_organization_ids if o != openai.organization]
                    )
                    logging.info(f"Switching OAI organization.")
                if openai_api_keys is not None and len(openai_api_keys) > 1:
                    openai.api_key = random.choice([o for o in openai_api_keys if o != openai.api_key])
                    logging.info(f"Switching OAI API key.")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
    return choices


def _requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    # TODO: this should ideally be an OpenAI function... Maybe it already exists?
    return "turbo" in model or "gpt-4" in model


def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> _prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


def _string_to_dict(to_convert):
    r"""Converts a string with equal signs to dictionary. E.g.
    >>> _string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def _get_price_per_token(model):
    """Returns the price per token for a given model"""
    if "gpt-4" in model:
        return (
            0.03 / 1000
        )  # that's not completely true because decoding is 0.06 but close enough given that most is context
    elif "gpt-3.5-turbo" in model:
        return 0.002 / 1000
    elif "text-davinci-003" in model:
        return 0.02 / 1000
    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan
