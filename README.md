# AlpacaEval

1. install `pip install -e . `
2. set `ANTHROPIC_API_KEY`
3. testing
   all: `alpaca_eval --model_outputs 'example/eval_gpt_3.5-turbo-0301.json' --annotators_config 'claude' --max_instances 3 --saving_path None `

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