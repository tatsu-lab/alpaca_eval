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
    # For inputs; here the inputs are model and reference model outputs.
    parser.add_argument("--model_outputs", type=str, help="Path to the directory containing the model outputs", required=False)

    # For outputs
    parser.add_argument("--output_dir", type=str, help="Output directory")

    # Parse and return the arguments
    return parser.parse_args()


ALL_FUNCTIONS = {
    "evaluate": evaluate,
    "evaluate_from_model": evaluate_from_model,
    "make_leaderboard": make_leaderboard,
    "analyze_evaluators": analyze_evaluators,
}


def stage2_main():
    args = parse_args()

    model_outputs_path = args.model_outputs
    output_dir = args.output_dir

    print(f"Path to model + reference model outputs: {model_outputs_path}")
    print(f"Output Directory: {output_dir}")


    # convert from json to pandas dataframe; this is what the evaluate function expects
    model_outputs_file = os.path.join(args.model_outputs, "model_outputs.json")
    reference_model_outputs_file = os.path.join(args.model_outputs, "reference_outputs.json")

    with open(model_outputs_file, 'r', encoding='utf-8') as fo:
        model_outputs = json.load(fo)

    model_outputs_df = pd.read_json(os.path.join(model_outputs_path, "model_outputs.json"), orient='records')

    with open(reference_model_outputs_file, 'r', encoding='utf-8') as fo:
        reference_model_outputs = json.load(fo)

    reference_model_outputs_df = pd.read_json(os.path.join(model_outputs_path, "reference_outputs.json"), orient='records')

    evaluate(
        model_outputs=model_outputs_df,
        reference_outputs=reference_model_outputs_df,
        annotators_config="alpaca_eval_gpt4",
        output_path=output_dir,
        max_instances=None,
    )

    # right now still consuming from model_configs folder.
    # evaluate_from_model(
    #     model_configs="./zephyr-7b-beta",
    #     annotators_config="",
    # )




if __name__ == "__main__":
    stage2_main()
