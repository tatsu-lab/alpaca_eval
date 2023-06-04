# AlpacaEval : An Automatic Evaluator of Instruction-following Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

Evaluation of instruction-following models (GPT4, ChatGPT) typically requires human interactions. This is
time-consuming, expensive, and hard to replicate. AlpacaEval in an LLM-based automatic evaluation that is fast, cheap,
and replicable.
AlpacaEval provides the following:

- [**Automatic evaluator**](#evaluators): an automatic evaluator that has high agreement with humans. We evaluate a
  model M by
  measuring the fraction of time an oracle LLM (e.g. Claude or GPT 4) prefers the outputs from that model M over a
  reference.
- [**Leaderboard**](#models): a leaderboard of common models on the AlpacaEval evaluation set.
- [**Toolkit for building automatic evaluators**](#analysis): a toolkit for
  building and analyzing automatic evaluators (quality,
  price, speed, statistical power, etc).
- [**Human evaluation data**](#data-release): 20K human annotations of preferences between a given and reference model
  on the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/tree/main)
  evaluation set. 2.5K of which are cross-annotations (4 humans annotating the same 650 examples).
- [**AlpacaEval dataset**](#data-release): a simplification of
  the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/tree/main) evaluation set, where "instructions" and "
  inputs" are merged
  into a single field.

<details open>
  <summary><b>Table of Contents</b></b></summary>

1. [Quick Start](#quick-start)
2. [Leaderboard](#leaderboard)
    - [Models](#models)
    - [Evaluators](#evaluators)
3. [Use-cases](#use-cases)
    - [Evaluating a model](#evaluating-a-model)
    - [Making a new evaluator](#making-a-new-evaluator)
    - [Making a new leaderboard](#making-a-new-leaderboard)
4. [Analysis](#analysis)
    - [Analyzing an evaluator](#analyzing-an-evaluator)
    - [Analyzing an eval set](#analyzing-an-eval-set)
5. [Data Release](#data-release)
6. [Citation](#citation)

</details>

## Quick Start

Fist, install the package: `pip install alpaca-eval`.

Then you can use it as follows:

```bash
export OPENAI_API_KEY=<your_api_key> 
alpaca_eval  --model_outputs 'example/eval_gpt_3.5-turbo-0301.json'
```

Important parameters are the following:

- **model_outputs** : A json path to the outputs of the model to add to the leaderboard. Each dictionary should
  contain `instruction` and `output` with optional `input`.
- **annotators_config**: `gpt4`, `text-davinci-003`, `claude`... Annotator to use. `gpt4` works best. If you are
  academics, we recommend `claude` which is free for academics and nearly as good. For a comparison of
  annotators see [here](#evaluators).
- **reference_outputs**:  The outputs of the reference model. Same format as `model_outputs`. By default, 003 outputs on
  AlpacaEval dataset.
- **output_path**: Path for saving annotations and leaderboard.

If you don't have the model outputs, you can use `evaluate_from_model` and pass a local path or a name of a HuggingFace
model, or a model from a standard API (OpenAI, anthropic, cohere). For more details to evaluate a model
see [here](#evaluating-a-model). Note that by default annotations are cached on disk. Annotations are thus never
recomputed, which greatly decreases cost and time for repeated evaluations (many models have the same outputs)

## Leaderboard

### Models

Our leaderboards are computed are on the [AlpacaEval dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval).
We precomputed the leaderboard for important models both using `gpt4` (best quality) and  `claude` (free for academics,
and high quality). See below for [adding your model](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model) to the
leaderboard or making
a [new leaderboard for your evaluator/dataset](https://github.com/tatsu-lab/alpaca_eval#making-a-new-leaderboard).

**GPT-4 Leaderboard**:

|                       | Win Rate | Std Err. |
|:----------------------|---------:|---------:|
| gpt4                  |     95.3 |      0.7 |
| claude                |     88.4 |      1.1 |
| chatgpt               |     86.1 |      1.2 |
| wizardlm-13b          |     75.3 |      1.5 |
| guanaco-65b           |     71.8 |      1.6 |
| vicuna-13b            |     70.4 |      1.6 |
| oasst-rlhf-llama-33b  |     66.5 |      1.7 |
| text_davinci_003      |     50.0 |      0.0 |
| falcon-40b-instruct   |     45.7 |      1.8 |
| alpaca-farm-ppo-human |     41.2 |      1.7 |
| alpaca-7b             |     26.5 |      1.5 |
| cohere                |     17.5 |      1.3 |
| text_davinci_001      |     15.2 |      1.2 |

<details>
  <summary><b>Claude Leaderboard</b></summary>

|                       | Win Rate | Std Err. |
|:----------------------|---------:|---------:|
| gpt4                  |     77.0 |      1.5 |
| claude                |     75.8 |      1.5 |
| chatgpt               |     67.7 |      1.6 |
| wizardlm-13b          |     66.1 |      1.7 |
| vicuna-13b            |     63.2 |      1.7 |
| guanaco-65b           |     62.6 |      1.7 |
| oasst-rlhf-llama-33b  |     57.3 |      1.7 |
| text_davinci_003      |     50.0 |      0.0 |
| falcon-40b-instruct   |     46.7 |      1.8 |
| alpaca-farm-ppo-human |     46.5 |      1.8 |
| alpaca-7b             |     32.3 |      1.6 |
| cohere                |     27.5 |      1.6 |
| text_davinci_001      |     21.5 |      1.4 |

</details>

### Evaluators

We evaluate different automatic annotators on the AlpacaEval set by comparing to
2.5k [human annotation](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)
we collected. For details about the evaluation metrics see [here](#analyzing-an-evaluator).

|                         | Human agreement [%] | Price [$/1000 examples] | Time [seconds/1000 examples] | Bias | Variance | Proba. prefer longer |
|:------------------------|--------------------:|------------------------:|-----------------------------:|-----:|---------:|---------------------:|
| alpaca_eval_gpt4        |                69.2 |                    13.6 |                         1455 | 28.4 |     14.6 |                 0.68 |
| aviary_gpt4             |                69.1 |                    12.8 |                         1869 | 29.5 |     13.1 |                 0.70 |
| gpt4                    |                66.9 |                    12.5 |                         1037 | 31.5 |     14.6 |                 0.65 |
| alpaca_farm_greedy_gpt4 |                66.4 |                    15.3 |                          878 | 30.2 |     19.3 |                 0.60 |
| humans                  |                65.7 |                   300.0 |                        36800 |  0.0 |          |                 0.64 |
| claude                  |                65.5 |                    11.1 |                          173 | 31.9 |     18.0 |                 0.62 |
| text_davinci_003        |                64.1 |                     8.7 |                          121 | 33.8 |     22.7 |                 0.70 |
| lmsys_gpt4              |                63.2 |                    13.9 |                        17982 | 34.7 |     16.1 |                 0.74 |
| guanaco_33b             |                59.1 |                         |                          930 | 54.5 |     27.1 |                 0.70 |
| chatgpt                 |                57.2 |                     0.8 |                          285 | 39.4 |     34.1 |                 0.59 |

<details>
  <summary><b>Tips for choosing evaluators</b></summary>

When choosing an annotator we recommend you to (obviously) consider the **quality** / **price** / **time**, but we also
suggest considering the following:

- "Proba. prefer longer " approx. < 0.7. Indeed, we found see that the majority of preference of human annotators (which
  we use
  as gold
  standard) have strong bias for longer answers (as shown by the high quality of the "longest" evaluator that always
  prefers the longest output). This suggests that it might more of a bias with the human annotators. In order to avoid
  having leaderboards with strong biases for length, we suggest using automatic annotators with less than 0.7 "Proba.
  prefer longer".
- "Variance" approx. < 0.2. We believe that a good evaluator should have as little variance as possible so that
  different people get similar results. Note that variance can be desirable in the case where we are simulating humans
  as shown in [AlpacaFarm](https://arxiv.org/abs/2305.14387).

We filtered the rest of the annotators in the table above (besides humans / ChatGPT / 003 for reference purposes), for
all
results see [here](). In general, we found `alpaca_eval` to be a good trade-off between quality / price / time /
variance / length bias.

</details>

## Use-cases

<details>
  <summary><b>Installation from source (optional)</b></b></summary>

1. clone the repository
2. install as dev the package: `pip install -e .`
3. (optional) export
   all [API_KEYs](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/constants.py#L7)
4. test your installation (assuming you have OpenAI
   key) `alpaca_eval --model_outputs 'example/eval_gpt_3.5-turbo-0301.json' --annotators_config 'text-davinci-003' --max_instances 3 --caching_path None `

</details>

### Evaluating a model

To evaluate a model you need to:

1. Choose an evaluation set and compute outputs specified as `model_outputs`. By default, we use
   the 805 examples from [AlpacaEval](#data-release). To compute outputs on AlpacaEval use:

```python
import datasets

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
for example in eval_set:
    # generate here is a placeholder for your models generations
    example["output"] = generate(example["instruction"])
```

if your model is a HuggingFace model or from a standard API provider (OpenAI, Anthropic, Cohere). Then you can
directly use `alpaca_eval evaluate_from_model` to also take care of generating outputs on the desired data as
discussed below.

2. Compute the reference outputs `reference_outputs`. By default, we use the outputs of `text-davinci-003` on
   AlpacaEval.
   If you
   want to use a different model or a different dataset use the same as (1.).
3. Choose an evaluator specified via `annotators_config`. We recommend using `alpaca_eval_gpt4` or `claude` (if you are
   an
   academic). For options and comparisons see [this table](#evaluators). Depending on the evaluator you might need to
   set the appropriate API_KEY in your environment
   or [here](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/constants.py#L7).

<details>
  <summary><b>Other parameters</b></b></summary>

The easiest is to check the docstrings
of [`evaluate`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/main.py#L15). Here are some
important ones:

```
Parameters
----------
model_outputs : path or data or dict
    The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
    datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
    (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
    and `output` with optional `input`. If None, we just print the leaderboard.

reference_outputs : path or data, optional
    The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
    003 outputs on AlpacaEval evaluation set.

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
             --annotators_config 'alpaca_eval'\
             --max_instances <specify for testing>
```

If you don't have decoded outputs, you can use `evaluate_from_model` which takes care of decoding (model and reference)
for you. Here's an
example:

```bash
# need a GPU for pythia
export ANTHROPIC_API_KEY=<your_api_key> # let's use claude as reference
alpaca_eval evaluate_from_model --model_configs 'oasst_pythia_12b'\
              --reference_model_configs 'claude'\
             --annotators_config 'chatgpt'\
             --max_instances 3
```

Here the `model_configs` and `reference_model_configs` (optional) are paths to a directory that specifies the prompt,
the model
provider (here HuggingFace and Anthropic) and decoding parameters.
See [this directory](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs) for examples.

Note that by default annotations are cached on
disk at `caching_path`. Annotations are thus never recomputed, which greatly decreases cost and time for repeated
evaluations (many models
have
the same outputs).

### Making a new evaluator

There are 4 main ways of making new evaluators: changing the prompt, changing decoding parameters (eg temperature),
changing the model, or using multiple annotators.
In each of these cases what you need is a new `configs.yaml` configuration file, which you will then pass
as `--annotators_config <path_to_config.yaml>` to `alpaca_eval`.
In particular, you should follow the following simple steps:

- **Changing the prompt**: Write a new prompt in a text file and specify the path in `prompt_template` of the
  configuration file. Paths are relative to the configuration file.
- **Changing decoding parameters**: Specify the desired parameters in `completions_kwargs` in the configuration file. To
  see all available parameters refer to the docstring corresponding
  function [in this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py)
  specified by `fn_completions`
  in the configuration file.
- **Changing the model**: Specify the desired model in `model_name` in the configuration file. You will likely have to
  change the prompt as `prompt_template` to match that model. If the model comes from another provider you will also
  have
  to change the `fn_completions` in the configuration file which maps to the corresponding function
  in [this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py). We
  provide `fn_completions` functions to use any model on OpenAI API, Anthropic API, Cohere API, or HuggingFace hub. If
  you change provider you will need to install there API and set the appropriate API_KEY. To install all
  use `pip install alpaca_eval[all]`.
- **Using multiple annotators**: Specify a list of annotators in `annotators_config` in the configuration file. For an
  example
  see [alpaca_farm configuration](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_farm/configs.yaml).

<details>
  <summary><b>Other parameters in the configuration file</b></b></summary>

The easiest is to check the docstrings
of [`SinglePairwiseAnnotator`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/annotators/pairwise_evaluator.py#L537).
Here are some important ones:

```
Parameters
----------
prompt_template : path
    A prompt that will be given to `fn_prompter` or path to the prompts. Path is relative to
    `evaluators_configs/`

fn_completion_parser : callable or str
    Function in `completion_parsers.py` to use for parsing the completions into preferences. For each completion,
    the number of preferences should be equal to the batch_size if not we set all the preferences in that batch to
    NaN.

completion_parser_kwargs : dict
    Kwargs for fn_completion_parser.

fn_completions : callable or str
    Function in `decoders.py` to use for decoding the output.

completions_kwargs : dict
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

### Making a new leaderboard

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
  default, the reference outputs are the 003 outputs on AlpacaEval set.
- `annotators_config`: The path to the annotator's config file. Defaults to `gpt4`.

# Analysis

AlpacaEval provides a few analysis tools to help you automatic evaluation. We briefly explain them here and provide
notebooks for all analysis.

### Analyzing an evaluator

**Notebook:**
[![analyzing an evaluator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_farm/blob/main/examples/auto_annotations.ipynb)

The most important factors when selecting a an evaluator are likely the quality, price, and speed. The following plot
measures

### Analyzing an eval set

**Notebook:**
[![analyzing an evaluation set](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_farm/blob/main/examples/auto_annotations.ipynb)

# Data release

As part of AlpacaEval, we release the following data:

- **Human annotations (17K)** in order to develop and understand automatic evaluators, we release all the human pairwise
  evaluation that we collected for AlpacaFarm. This contains comparisons between 18 models with the `text-davinci-003`
  reference on the AlpacaFarm evaluation set. Annotations are from a pool of 16 crowdworkers on Amazon Mechanical Turk.
- **Human cross-annotations (3K)** in order to further analyze automatic evaluators we selected (via stratified sampling
  across models and datasets) 650 examples from the AlpacaFarm evaluation set and collected 4 human annotations per
  example. This allows us to estimate the bias and variance of automatic evaluators.
- **AlpacaEval set (800)** we made slight modifications/simplification of the AlpacaFarm evaluation set. In particular,
  we first merged
  the instruction and input fields into a single instruction field. This affects 1/4 of the examples in the AlpacaFarm
  evaluation set, all of which are from the [self-instruct evaluation set](https://arxiv.org/abs/2212.10560). Second we
  regenerated the text-davinci-003 reference outputs without limiting the length of its outputs.

## Citation

Please consider citing the repo if you use the automatic annotators, code, or results in this repo.

```
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_eval}},
}
```

If you use the human annotations, please also cite the [AlpacaFarm](https://arxiv.org/abs/2305.14387)
paper:

```
@misc{dubois2023alpacafarm,
      title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
      author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
      year={2023},
      eprint={2305.14387},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
