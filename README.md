# AlpacaEval : An Automatic Evaluator of Instruction-following Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

Evaluation of instruction-following models (GPT4, ChatGPT) typically requires human interactions. This is
time-consuming, expensive, and hard to replicate. AlpacaEval in an LLM-based automatic evaluation that is fast, cheap,
and replicable.
AlpacaEval provides the following:

- **Automatic evaluator**: an automatic evaluator that has high agreement with humans. We evaluate a model M by
  measuring the fraction of time an oracle LLM (e.g. Claude or GPT 4) prefers the desired model M over a reference.
- **Leaderboard**: a leaderboard of common models on the AlpacaFarm evaluation set.
- **Toolkit for building automatic evaluators**: a toolkit for building and analyzing automatic evaluators (quality,
  price, speed, statistical power, etc).
- **Human evaluation**: a human evaluation of the automatic evaluator.

# Quick Start

Fist, install the package: `pip install alpaca-eval`.

Then you can use it as follows:

```bash
export OPENAI_API_KEY=<your_api_key> # if using OpenAI models
alpaca_eval  --model_outputs 'example/eval_gpt_3.5-turbo-0301.json'\
             --name '**Current method**'\
             --annotators_config 'gpt4'\
```

Important parameters are the following:

- **model_outputs** : A json path to the outputs of the model to add to the leaderboard. Each dictionary should
  contain `instruction` and `output` with optional `input`.
- **annotators_config**: `gpt4`, `text-davinci-003`, `claude`... Annotator to use. `gpt4` works best. If you are
  academics, we recommend `claude` which is free for academics and nearly as good. For a comparison of
  annotators see [here](#evaluators).
- **reference_outputs**:  The outputs of the reference model. Same format as `model_outputs`. By default 003 outputs on
  AlpacaFarm evaluation set.
- **output_path**: Path for saving annotations and leaderboard.

For more details to evaluate a model see [here](#evaluating-a-model).

# Leaderboard

## Models

Our leaderboards are computed are on the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) evaluation set.
We precomputed the leaderboard for important models both using `gpt4` (best quality) and  `claude` (faster, free, nearly
as good). Details:

- [Adding your model to the leaderboard](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model)
- [Making a leaderboard for a new evaluator / dataset](https://github.com/tatsu-lab/alpaca_eval#making-a-new-leaderboard)

**GPT-4 Leaderboard**:

|                                                | Win Rate | Std Err. |
|:-----------------------------------------------|---------:|---------:|
| GPT-4                                          |     92.2 |      0.9 |
| ChatGPT                                        |     65.9 |      1.7 |
| Davinci003                                     |     50.0 |      0.0 |
| AlpacaFarm PPO sim (gpt4 greedy 20k, step 350) |     49.5 |      1.7 |
| AlpacaFarm PPO human (10k, step 40)            |     46.6 |      1.8 |
| Alpaca 7B                                      |     36.9 |      1.7 |
| Davinci001                                     |     19.8 |      1.4 |

<details>
  <summary><b>Claude Leaderboard</b></summary>

|                                                | Win Rate | Std Err. |
|:-----------------------------------------------|---------:|---------:|
| GPT-4                                          |     78.6 |      1.4 |
| ChatGPT                                        |     62.4 |      1.7 |
| AlpacaFarm PPO sim (gpt4 greedy 20k, step 350) |     58.3 |      1.7 |
| AlpacaFarm PPO human (10k, step 40)            |     56.3 |      1.7 |
| Davinci003                                     |     50.0 |      0.0 |
| Alpaca 7B                                      |     45.8 |      1.7 |
| Davinci001                                     |     24.5 |      1.5 |

</details>

## Evaluators

We evaluate different automatic annotators on the AlpacaFarm evaluation set by comparing to
2.5k [human annotation](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)
we collected. For details about the evaluation metrics see [here]().

|                           | Human agreement<br/>[%] | Price<br/>[$/1000 examples] | Time<br/>[seconds/1000 examples] |
|:--------------------------|------------------------:|----------------------------:|---------------------------------:|
| gpt4                      |                    66.6 |                        12.5 |                           1217.5 |
| alpaca_farm_greedy (gpt4) |                    66.5 |                        15.4 |                            983.8 |
| humans                    |                    65.7 |                       300.0 |                          36800.0 |
| claude                    |                    65.3 |   14.4 (free for academics) |                           1416.0 |
| text_davinci_003          |                    64.6 |                         8.8 |                             78.4 |
| lmsys (gpt4)              |                    63.8 |                        13.9 |                           6320.7 |
| alpaca_farm               |                    60.6 |                        11.9 |                            888.7 |
| guanaco_33b               |                    59.7 |                             |                            939.0 |
| chatgpt                   |                    58.5 |                         0.8 |                            311.8 |
| cohere                    |                    53.0 |                         2.5 |                            290.7 |
| oasst_pythia_12b          |                    51.2 |                             |                            230.2 |

# Use-cases

## Evaluating a model

To evaluate a model you need to:

1. Choose an evaluation set and compute outputs specified as `model_outputs`. By default, we use the AlpacaFarm
   evaluation set, which contain 805 examples. To compute outputs on
   AlpacaFarm use:

```python
import datasets

eval_set = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")["eval"]
for example in eval_set:
    # generate here is a placeholder for your models generations
    example["output"] = generate(example["instruction"])
```

2. Compute the reference outputs `reference_outputs`. By default, we use the outputs of text-davinci-003 on AlpacaFarm.
   If you
   want to use a different model or a different dataset use the same as (1).
3. Choose an evaluator specified via `annotators_config`. We recommend using `gpt4` or `claude` (if you are an
   academic). For options and comparisons see [this table](#evaluators). Depending on the evaluator you might need to
   set the appropriate API_KEY in your environment
   or [here](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/constants.py#L7).

<details>
  <summary><b>Other parameters</b></b></summary>

The easiest is to check the docstrings
of [`pairwise_winrates`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/main.py#L15). Here are some
important ones:

```
Parameters
----------
model_outputs : path or data or dict
    The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
    datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
    (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
    and `output` with optional `input`.

reference_outputs : path or data, optional
    The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
    003 outputs on AlpacaFarm evaluation set.

annotators_config : path or list of dict, optional
    The path the (or list of dict of) the annotator's config file. For details see the docstring of
    `PairwiseAnnotator`.

name : str, optional
    The name of the model to add to the leaderboard.

output_path : bool, optional
    Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
    If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

precomputed_leaderboard : path or data, optional
    The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
    column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
    in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

max_instances : int, optional
    The maximum number of instances to annotate. Useful for testing.

annotation_kwargs : dict, optional
    Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

annotator_kwargs :
    Additional arguments to pass to `PairwiseAnnotator`.
```

</details>

Running all together:

```bash
alpaca_eval  --model_outputs 'example/eval_gpt_3.5-turbo-0301.json'\
              --reference_outputs <path> \
             --name '**Current method**'\
             --annotators_config 'gpt4'\
             --output_path <example>\
             --max_instances <specify for testing>
```

## Making a new evaluator

There are 4 main ways of making new evaluators: changing the prompt, changing decoding parameters (eg temperature),
changing the model, or using multiple annotators.
In each of these cases what you need is a new `configs.yaml` configuration file, which you will then pass
as `--annotators_config <path_to_config.yaml>` to `alpaca_eval`.
In particular, you should follow the following simple steps:

- **Changing the prompt**: Write a new prompt in a text file and specific it to `prompt_templates` in the configuration
  file. Paths are relative to the configuration file.
- **Changing decoding parameters**: Specify the desired parameters in `decoder_kwargs` in the configuration file. To see
  all available parameters refer to the docstring corresponding
  function [in this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py)
  specified by `fn_completions`
  in
  the configuration file.
- **Changing the model**: Specify the desired model in `model_name` in the configuration file. You will likely have to
  change the prompt in `prompt_templates` to match that model. If the model comes from another provider you will also
  have
  to change the `fn_completions` in the configuration file which maps to the corresponding function
  in [this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py). We
  provide `fn_completions` functions to use any model on OpenAI API, Anthropic API, Cohere API, or HuggingFace hub.
- **Using multiple annotators**: Specify a list of annotators in `annotators_config` in the configuration file. For an
  example
  see [alpaca_farm configuration](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/configs/alpaca_farm/configs.yaml).

<details>
  <summary><b>Other parameters in the configuration file</b></b></summary>

The easiest is to check the docstrings
of [`SinglePairwiseAnnotator`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/annotators/pairwise_evaluator.py#L537).
Here are some important ones:

```
Parameters
----------
prompt_templates : path
    A dictionary of prompts that will be given to `fn_prompter` or path to those prompts. Path is relative to
    `configs/`

fn_completion_parser : callable or str
    Function in `completion_parsers.py` to use for parsing the completions into preferences. For each completion,
    the number of preferences should be equal to the batch_size if not we set all the preferences in that batch to
    NaN.

completion_parser_kwargs : dict
    Kwargs for fn_completion_parser.

fn_completions : callable or str
    Function in `decoders.py` to use for decoding the output.

decoder_kwargs : dict
    kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

is_randomize_output_order : bool
    Whether to randomize output_1, output_2 when formatting.

batch_size : int
    Number of examples that will be added in a single prompt.
```

</details>

Once you made the evaluator you can also analyze it and add it to the _evaluator's_ [leaderboard](#evaluators) using the
following command:

```bash
alpaca_eval analyze_evaluators --annotators_config '<path_to_config.yaml>'    
```

Note that this will evaluate 4 times (different seeds) every example in the AlpacaFarm evaluation set, i.e., ~3K
evaluation.
Be mindful of the cost of this operation depending on your model.

## Making a new leaderboard

If you want to make a new leaderboard in one go (rather than multiple `alpaca_eval` calls), for your desired evaluation
set and evaluators, you can use the following:

```bash
alpaca_eval make_leaderboard --leaderboard_path <path_to_save_leaderboard>\
                             --all_model_outputs <model_outputs_path>\
                             --reference_outputs <reference_outputs_path>\
                              --annotators_config <path_to_config.yaml>
```

where:

- `leaderboard_path`: path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists
  it will append.
- `all_model_outputs` : The json path to outputs of all models to add to the leaderboard. Each dictionary should contain
  the keys that are formatted in the prompts. E.g. by default `instruction` and `output` with optional `input`. It
  should also contain a column `generator` with the name of the current model.
- `reference_outputs` the path to the outputs of the reference model. Same format as `all_model_outputs` but without
  needing `generator`. By
  default, the reference outputs are the 003 outputs on AlpacaFarm evaluation set.
- `annotators_config`: The path to the annotator's config file. Defaults to `gpt4`.

## Developing

Install from source:

1. clone the repository
2. install as dev the package: `pip install -e .`
3. (optional) export all API_KEYs
4. test your installation (assuming you have OpenAI
   key) `alpaca_eval --model_outputs 'example/eval_gpt_3.5-turbo-0301.json' --annotators_config 'text-davinci-003' --max_instances 3 --caching_path None `

# Analysis

## Evaluator leaderboard

## Analyzing an evaluator

## Analyzing an evaluation set

to get instructions:

```python
import datasets

eval = datasets.load_dataset(
    "tatsu-lab/alpaca_farm",
    "alpaca_farm_evaluation",
)["eval"]
```

to run the eval :

```
alpaca_eval --model_outputs 'outputs/claude/<model_name>.json' --annotators_config 'claude'
```