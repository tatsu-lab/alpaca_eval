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

from . import analyze, annotators, constants, decoders, metrics, utils
from .types import AnyData, AnyLoadableDF, AnyPath
from .main import evaluate, evaluate_from_model, analyze_evaluators, make_leaderboard


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif v.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Define arguments based on the YAML spec
    # For inputs; here the inputs are model and reference model outputs.
    parser.add_argument("--model_outputs", type=str, help="Path to the directory containing the model outputs", required=False)
    parser.add_argument("--use_alpaca_eval_1", type=str2bool, help="Whether to use the alpaca_eval_1 or eval_2; by default set to False.", required=False, default=True)

    # For outputs
    parser.add_argument("--output_dir", type=str, help="Output directory")

    # Parse and return the arguments
    return parser.parse_args()


def stage2_main():
    args = parse_args()

    logging.info(f"Args = \n{args}")

    model_outputs_path = args.model_outputs
    output_dir = args.output_dir

    if args.use_alpaca_eval_1:
        annotator_config_file = "alpaca_eval_gpt4"
    else:
        annotator_config_file = "weighted_alpaca_eval_gpt4_turbo"
    
    logging.info(f"Using annotator_config_file = {annotator_config_file}")


    # convert from json to pandas dataframe; this is what the evaluate function expects
    model_outputs_df = pd.read_json(os.path.join(model_outputs_path, "model_outputs.json"), orient='records')
    reference_model_outputs_df = pd.read_json(os.path.join(model_outputs_path, "reference_outputs.json"), orient='records')

    evaluate(
        model_outputs=model_outputs_df,
        reference_outputs=reference_model_outputs_df,
        annotators_config=annotator_config_file,
        output_path=output_dir,
        max_instances=None,
    )




if __name__ == "__main__":
    stage2_main()
