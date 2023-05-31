# AlpacaEval

to get instructions

```python
import datasets

eval = datasets.load_dataset(
    "tatsu-lab/alpaca_farm",
    "alpaca_farm_evaluation",
)["eval"]
```

to run the eval

```
alpaca_eval --model_outputs 'example/eval_gpt_3.5-turbo-0301.json' --annotators_config 'claude'  --max_instances 3 --saving_path None 
```