from pathlib import Path
from alpaca_eval.constants import PRECOMPUTED_LEADERBOARDS, MODELS_CONFIG_DIR, MINIMAL_MODELS, VERIFIED_MODELS
from alpaca_eval.utils import load_or_convert_to_dataframe, load_configs

for leaderboard_file in PRECOMPUTED_LEADERBOARDS.values():
    df = load_or_convert_to_dataframe(leaderboard_file)
    df = df[['win_rate']]
    df = df.reset_index(names='name')
    df['link'] = ''
    df['filter'] = ''
    for idx in range(len(df)):
        informal_name = df.loc[idx, 'name']
        model_config = load_configs(df.loc[idx, 'name'], relative_to=MODELS_CONFIG_DIR)[informal_name]
        if 'pretty_name' in model_config:
            df.loc[idx, 'name'] = model_config['pretty_name']
        if 'link' in model_config:
            df.loc[idx, 'link'] = model_config['link']
        if informal_name in MINIMAL_MODELS:
            df.loc[idx, 'filter'] = 'minimal'
        elif informal_name in VERIFIED_MODELS:
            df.loc[idx, 'filter'] = 'verified'
        else:
            df.loc[idx, 'filter'] = 'community'
    df = df.sort_values(by=['win_rate'], ascending=False)
    df.to_csv(Path('docs') / leaderboard_file.name, index=False)