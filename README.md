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
export ANTHROPIC_API_KEY=<your_api_key> # if using claude 
export OPENAI_API_KEY=<your_api_key> # if using 'gpt4' or OpenAI models
alpaca_eval  --model_outputs 'example/eval_gpt_3.5-turbo-0301.json'\
             --name '**Current method**'\
             --annotators_config 'gpt4'
```

Important parameters are the following:

- **model_outputs** : The outputs of the model to add to the leaderboard. Accepts data (list of dictionary or
  datasets.Dataset) or a json path to read those. Each dictionary should contain `instruction` and `output` with
  optional `input`.
- **annotators_config**: `gpt4`, `text-davinci-003`, `claude`... Annotator to use. `gpt4` works best. If you have
  access, we recommend `claude` which is faster, free for academics and nearly as good. For a comparison of annotators
  see [here]().
- **reference_outputs**:  The outputs of the reference model. Same format as `model_outputs`. By default 003 outputs on
  AlpacaFarm evaluation set.
- **output_path**: Path for saving annotations and leaderboard.

# Leaderboard

## Models

Our leaderboards are computed are on the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) evaluation set.
We precomputed the leaderboard for important models both using `gpt4` (best quality) and  `claude` (faster, free, nearly
as good). Details:

- [Adding your model to the leaderboard]()
- [Making a leaderboard for a new evaluator / dataset]()

**Claude Leaderboard**:

|                                                | Win Rate | Std Err. |
|:-----------------------------------------------|---------:|---------:|
| GPT-4                                          |     78.6 |      1.4 |
| ChatGPT                                        |     62.4 |      1.7 |
| AlpacaFarm PPO sim (gpt4 greedy 20k, step 350) |     58.3 |      1.7 |
| AlpacaFarm PPO human (10k, step 40)            |     56.3 |      1.7 |
| Davinci003                                     |     50.0 |      0.0 |
| Alpaca 7B                                      |     45.8 |      1.7 |
| Davinci001                                     |     24.5 |      1.5 |

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

## Evaluators

We evaluate different automatic annotators on the AlpacaFarm evaluation set by comparing to
2.5k [human annotation](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)
we collected. For details about the evaluation metrics see [here]().

|                    | Human agreement [%] |   Price [$/1000 examples] | Time [seconds/1000 examples] |
|:-------------------|--------------------:|--------------------------:|-----------------------------:|
| gpt4               |                66.6 |                      12.5 |                       1217.5 |
| alpaca_farm_greedy |                66.5 |                      15.4 |                        983.8 |
| humans             |                65.7 |                     300.0 |                      36800.0 |
| claude             |                65.3 | 14.4 (free for academics) |                        177.1 |
| text_davinci_003   |                64.6 |                       8.8 |                         78.4 |
| lmsys              |                63.8 |                      13.9 |                       6320.7 |
| alpaca_farm        |                60.6 |                      11.9 |                        888.7 |
| guanaco_33b        |                59.7 |                           |                        939.0 |
| chatgpt            |                58.5 |                       0.8 |                        311.8 |
| cohere             |                53.0 |                       2.5 |                        290.7 |
| oasst_pythia_12b   |                51.2 |                           |                        230.2 |

# Usecases

## Evaluating a model

To evaluate a model you need to:

1) decide on an evaluator. E.g.

compute the outputs of the model

## Making a new evaluator

If you

## Making a new leaderboard

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