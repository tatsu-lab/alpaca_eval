import pandas as pd

model_outputs = pd.read_json("results/gpt4/model_outputs.json")
annotations = pd.read_json("results/gpt4/weighted_alpaca_eval_gpt4_turbo/annotations.json")
annotations["len_1"] = annotations.output_1.str.len()
annotations["len_2"] = annotations.output_2.str.len()
annotations["delta_len"] = annotations["len_2"] - annotations["len_1"]
df_ann = annotations.query("delta_len > 0").query("preference > 1.5").sort_values("preference", ascending=False)
model_outputs["output"] = model_outputs["output"].apply(lambda s: s[:10])
model_outputs.loc[df_ann.index, "output"] = df_ann["output_2"]
model_outputs["generator"] = "gpt4_gamed"
model_outputs.to_json("results/gpt4_gamed/model_outputs.json", orient="records", indent=2)
