import logging
logging.basicConfig(level=logging.INFO)

import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

# import fire
import pandas as pd
import sys
import os
import argparse
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from alpaca_eval import analyze, annotators, constants, decoders, metrics, utils
# from .. import analyze, annotators, constants, decoders, metrics, utils

from alpaca_eval.types import AnyData, AnyLoadableDF, AnyPath

from alpaca_eval.main import evaluate, evaluate_from_model, analyze_evaluators, make_leaderboard


CUR_DIR = Path(__file__).parent

__all__ = ["evaluate", "evaluate_from_model", "analyze_evaluators", "make_leaderboard"]


def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Define arguments based on the YAML spec
    # For inputs
    parser.add_argument("--model_path", type=str, help="Path to the directory containing the model checkpoint", required=False)
    parser.add_argument("--model_name", type=str, help="Name of the model", required=False)
    parser.add_argument("--datasets_dir", type=str, help="Path to the directory containing datasets", required=False)

    # For outputs
    parser.add_argument("--output_dir", type=str, help="Output directory for the model and reference model outputs.")

    # Parse and return the arguments
    return parser.parse_args()


def stage1_main():
    args = parse_args()
    output_dir = args.output_dir

    # Example usage of parsed arguments
    print(f"Model Path: {args.model_path}")
    print(f"Model Name: {args.model_name}")
    print(f"Datasets Directory: {args.datasets_dir}")
    print(f"Output Directory for model outputs: {output_dir}")

    model_name = args.model_name if args.model_name is not None else "custom_model_name"

    ## build the model_configs dictionary
    base_model_config = {
        model_name: {
            "prompt_template": "zephyr-7b-alpha/prompt.txt",  # This is a prompt template or path to one. It contains placeholders for keys in the data dictionary, typically {instruction} and {output}.
            "fn_completions": "huggingface_local_completions",  # This is the name of the function in a library or module (presumably `alpaca_farm.decoders` or similar) that will be used for generating completions. This function needs to accept a `prompts` argument, which is a list of strings.
            "completions_kwargs": {  # These are the keyword arguments for the `fn_completions` function, specifying how completions should be generated.
                "model_name": "",  # The identifier for the model to use for decoding, which in this case, is a model hosted on Hugging Face's model hub eg. HuggingFaceH4/zephyr-7b-beta.
                "model_kwargs": {  # Additional kwargs that are likely passed to the model or the generation function, possibly for configuring the underlying machine learning framework (like PyTorch).
                    "torch_dtype": 'bfloat16'  # Specifies the data type for PyTorch tensors, in this case, using bfloat16 for potentially reduced memory usage and faster computation.
                },
                "max_new_tokens": 2048,  # The maximum number of new tokens to generate in the completions.
                "temperature": 0.7,  # The temperature setting for the generation, which controls the randomness. A value of 0.7 suggests a balance between randomness and determinism.
                "top_p": 1.0,  # The nucleus sampling parameter, controlling the cumulative probability cutoff for token selection. A value of 1.0 effectively disables nucleus sampling.
                "do_sample": True  # Indicates that sampling is enabled, allowing for stochastic generation rather than deterministic output.
            },
            "pretty_name": "Custom Model",  # A human-readable name for the model or configuration.
            "link": ""  # A URL providing more information about the model, likely pointing to its page on Hugging Face's model hub.
        }
    }

    base_model_config[model_name]["completions_kwargs"]["model_name"] = args.model_path


    evaluate_from_model(
                # model_configs="./zephyr-7b-beta",
                model_configs=base_model_config,
                annotators_config="alpaca_eval_gpt4",  # we are using alpaca eval 1, because no logprobs.
                output_path=output_dir,
            )



if __name__ == "__main__":
    stage1_main()
