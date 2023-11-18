# Annotators configs

## Evaluator's leaderboard:

Here's the full leaderboard estimated on 4 different seeds, which allows us to also estimate the variance of the
annotators.
We compute those metrics on our suggested evaluator `alpaca_eval_gpt4`, on prior
evaluators (`aviary_gpt4`, `lmsys_gpt4`, `alpaca_farm_greedy_gpt4`), and on different base models with which we use
essentially the same prompt (`gpt4`, `text_davinci_003`, `claude`, `chatgpt`).
We also provide partial metrics (only 1 seed) for other evaluators, which include our evaluator using OpenAI's function
calls (`alpaca_eval_gpt4_fn`), prior work that we
improved (`improved_aviary_gpt4` and `improved_lmsys_gpt4`), prior work that was not meant to be used as a final
evaluator (`guanaco_33b`), and a ranking evaluator (`alpaca_farm`), and secondary models that use the same prompt as the
models above (`cohere`, `guanaco_33b`):

|                           | Human agreement [%] | Price [$/1000 examples] | Time [seconds/1000 examples] | Bias | Variance | Proba. prefer longer | Proba. prefer lists | Proba. prefer 1 | # parsed | mode     |
|:--------------------------|--------------------:|------------------------:|-----------------------------:|-----:|---------:|---------------------:|--------------------:|----------------:|---------:|:---------|
| alpaca_eval_gpt4_fn       |                71.0 |                    14.5 |                         5046 | 27.6 |     11.1 |                 0.75 |                0.63 |            0.48 |     2592 | verified |
| improved_aviary_gpt4      |                69.8 |                    12.8 |                         1831 |      |          |                 0.73 |                0.68 |            0.49 |      648 | verified |
| alpaca_eval_gpt4          |                69.2 |                    13.6 |                         1455 | 28.4 |     14.6 |                 0.68 |                0.69 |            0.50 |     2592 | minimal  |
| aviary_gpt4               |                69.1 |                    12.8 |                         1869 | 29.5 |     13.1 |                 0.70 |                0.65 |            0.53 |     2592 | minimal  |
| alpaca_eval_gpt4_turbo_fn |                68.7 |                    5.07 |                          827 |      |          |                 0.64 |                0.59 |            0.51 |      648 | minimal  |
| claude_ranking            |                67.6 |                     5.0 |                          218 |      |          |                 0.73 |                0.63 |            0.46 |      648 | verified |
| gpt4                      |                66.9 |                    12.5 |                         1037 | 31.5 |     14.6 |                 0.65 |                0.61 |            0.54 |     2592 | minimal  |
| alpaca_farm_greedy_gpt4   |                66.4 |                    15.3 |                          878 | 30.2 |     19.3 |                 0.60 |                0.59 |            0.54 |     2592 | minimal  |
| humans                    |                65.7 |                   300.0 |                        36800 |  0.0 |     34.3 |                 0.64 |                0.61 |            0.52 |     2592 | minimal  |
| claude                    |                65.5 |                    11.1 |                          173 | 31.9 |     18.0 |                 0.62 |                0.58 |            0.49 |     2592 | minimal  |
| text_davinci_003          |                64.1 |                     8.7 |                          121 | 33.8 |     22.7 |                 0.70 |                0.64 |            0.47 |     2592 | minimal  |
| lmsys_gpt4                |                63.2 |                    13.9 |                        17982 | 34.7 |     16.1 |                 0.74 |                0.64 |            0.56 |     2592 | minimal  |
| guanaco_33b               |                62.7 |                         |                          911 |      |          |                 0.70 |                0.72 |            0.43 |      451 | verified |
| improved_lmsys_gpt4       |                62.3 |                    13.9 |                         5398 |      |          |                 0.75 |                0.67 |            0.51 |      648 | verified |
| longest                   |                62.2 |                     0.0 |                            0 | 37.8 |      0.0 |                 1.00 |                0.85 |            0.42 |     2592 | verified |
| alpaca_farm               |                60.0 |                    11.5 |                          820 |      |          |                 0.60 |                0.63 |            0.52 |      648 | verified |
| chatgpt_fn                |                60.0 |                     1.0 |                          530 | 36.9 |     27.7 |                 0.62 |                0.65 |            0.49 |     2592 | minimal  |
| chatgpt                   |                57.2 |                     0.8 |                          285 | 39.4 |     34.1 |                 0.59 |                0.56 |            0.49 |     2589 | minimal  |
| cohere                    |                56.6 |                     6.5 |                          503 |      |          |                 0.63 |                0.65 |            0.46 |      643 | verified |

[//]: # (|                         | Human agreement [%] | Price [$/1000 examples] | Time [seconds/1000 examples] | Bias | Variance | Proba. prefer longer | Proba. prefer lists | Proba. prefer 1 | # parsed | mode     |)

[//]: # (|:------------------------|--------------------:|------------------------:|-----------------------------:|-----:|---------:|---------------------:|--------------------:|----------------:|---------:|:---------|)

[//]: # (| improved_aviary_gpt4    |                69.8 |                    12.8 |                         1831 |      |          |                 0.73 |                0.68 |            0.49 |      648 | verified |)

[//]: # (| alpaca_eval_gpt4        |                69.2 |                    13.6 |                         1455 | 28.4 |     14.6 |                 0.68 |                0.69 |            0.50 |     2592 | minimal  |)

[//]: # (| aviary_gpt4             |                69.1 |                    12.8 |                         1869 | 29.5 |     13.1 |                 0.70 |                0.65 |            0.53 |     2592 | minimal  |)

[//]: # (| claude_ranking          |                67.6 |                     5.0 |                          218 |      |          |                 0.73 |                0.63 |            0.46 |      648 | verified |)

[//]: # (| gpt4                    |                66.9 |                    12.5 |                         1037 | 31.5 |     14.6 |                 0.65 |                0.61 |            0.54 |     2592 | minimal  |)

[//]: # (| alpaca_farm_greedy_gpt4 |                66.4 |                    15.3 |                          878 | 30.2 |     19.3 |                 0.60 |                0.59 |            0.54 |     2592 | minimal  |)

[//]: # (| humans                  |                65.7 |                   300.0 |                        36800 |  0.0 |          |                 0.64 |                0.61 |            0.52 |     2592 | minimal  |)

[//]: # (| claude                  |                65.5 |                    11.1 |                          173 | 31.9 |     18.0 |                 0.62 |                0.58 |            0.49 |     2592 | minimal  |)

[//]: # (| text_davinci_003        |                64.1 |                     8.7 |                          121 | 33.8 |     22.7 |                 0.70 |                0.64 |            0.47 |     2592 | minimal  |)

[//]: # (| lmsys_gpt4              |                63.2 |                    13.9 |                        17982 | 34.7 |     16.1 |                 0.74 |                0.64 |            0.56 |     2592 | minimal  |)

[//]: # (| guanaco_33b             |                62.7 |                         |                          911 |      |          |                 0.70 |                0.72 |            0.43 |      451 | verified |)

[//]: # (| improved_lmsys_gpt4     |                62.3 |                    13.9 |                         5398 |      |          |                 0.75 |                0.67 |            0.51 |      648 | verified |)

[//]: # (| longest                 |                62.2 |                     0.0 |                            0 | 37.8 |      0.0 |                 1.00 |                0.85 |            0.42 |     2592 | verified |)

[//]: # (| alpaca_farm             |                60.0 |                    11.5 |                          820 |      |          |                 0.60 |                0.63 |            0.52 |      648 | verified |)

[//]: # (| chatgpt                 |                57.2 |                     0.8 |                          285 | 39.4 |     34.1 |                 0.59 |                0.56 |            0.49 |     2589 | minimal  |)

[//]: # (| cohere                  |                53.4 |                     3.5 |                          217 |      |          |                 0.50 |                0.51 |            0.47 |      648 | verified |)

Note that `improved_*` are evaluators of other groups that we improved. In particular, we added randomization of the
examples in the prompts and decreased temperature.

## Directory structure

Each evaluator has its own directory. Inside the directory we have:

- add a `configs.yaml` file that configures the evaluator (API provider, model, parameters, parsing function,
  prompts...)
- typically the prompts used for evaluation (besides if we reuse prompts from other models)

When using the evaluator we will by default cache all the annotations in `annotations_seed{seed}_configs.json` which
ensures that we do not rerun annotations (faster, cheaper, more reproducible).  
