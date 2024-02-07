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
    parser.add_argument("--debug_len", type=int, help="Number of examples to use for debugging", required=False)
    parser.add_argument("--model_config", type=str, help="Path to the model config file", required=False, default="")

    # For outputs
    parser.add_argument("--output_dir", type=str, help="Output directory for the model and reference model outputs.")

    # Parse and return the arguments
    return parser.parse_args()


def stage1_main():
    args = parse_args()
    output_dir = args.output_dir
    model_config = str(args.model_config).strip()


    logging.info(f"Args = \n{args}")

    ## The model path is provided by the input model. 
    ## The model config is provided by the model_config parameter. 
    ## We will:
    ##      - pick up the appropriate model_config from the model_configs folder according to the model_config parameter
    ##      - we will change the model_name field to point to the model_path 
    ##      - use this model_config dict to pass to the evaluate_from_model

    ae_model_config = utils.load_configs(model_config, relative_to=constants.MODELS_CONFIG_DIR)

    # get name of model 
    model_name = list(ae_model_config.keys())[0]
    if args.model_path is not None:
        ae_model_config[model_name]["completions_kwargs"]["model_name"] = args.model_path

    logging.info(f"Model Configs = \n{ae_model_config}")
    # else go with the default AE config and model on HF.

    ## TODO: do we change the model name?
    ## For this, we will take the ae_model_config dictionary, and rename the toplevel key to the model_name
    
    # if args.model_name is not None:
    #     ae_model_config[args.model_name] = ae_model_config.pop(model_name)
    #     model_name = args.model_name

    # ae_model_config[model_name]["pretty_name"] = args.model_name


    evaluate_from_model(
                model_configs=ae_model_config,
                output_path=output_dir,
                debug_len=args.debug_len,
            )



if __name__ == "__main__":
    stage1_main()
