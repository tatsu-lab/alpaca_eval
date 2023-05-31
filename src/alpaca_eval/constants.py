import getpass
import os
from pathlib import Path

import datasets

DEFAULT_CACHE_DIR = None
DATASETS_TOKEN = os.environ.get("DATASETS_TOKEN", None)

OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")

MODELS_TO_BENCHMARK = (
    "GPT-4",
    "ChatGPT",
    "AlpacaFarm PPO sim (gpt4 greedy 20k, step 350)",
    "AlpacaFarm PPO human (10k, step 40)",
    "Alpaca",
    "Davinci003",
    "Davinci001",
)

API_EVALUATORS_TO_ANALYZE = (
    "gpt4",
    "claude",
    "text_davinci_003",
    "gpt4",
    "chatgpt",
    "guanaco_33b",
    "lmsys",
    "cohere",
    "alpaca_farm",
    "alpaca_farm_greedy_gpt4",
)
LOCAL_EVALUATORS_TO_ANALYZE = (
    "oasst_pythia_12b",
    "stablelm_alpha_7b",
)

EVALUATORS_TO_ANALYZE = tuple(
    list(LOCAL_EVALUATORS_TO_ANALYZE)
    + list(API_EVALUATORS_TO_ANALYZE)
    + ["humans", "length"]
)

HUMAN_ANNOTATED_MODELS_TO_KEEP = (
    "GPT-4",
    "gpt-4-0314",
    "PPO 40 steps",
    "ChatGPT",
    "Best-of-16",
    "PPO 30 steps (Greedy GPT4)",
    "text-davinci-003",
    "gpt-3.5-turbo-0301",
    "ExpIter-128",
    "SFT 10K",
    "PPO 40 steps (multi)",
    "SFT 52K",
    "FeedMe",
    "Davinci001",
    "LLaMA 7B",
)

CUR_DIR = Path(__file__).parent


def ALPACAFARM_REFERENCE_OUTPUTS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_evaluation",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        download_mode="force_redownload",
    )["eval"]


def ALPACAFARM_ALL_OUTPUTS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_evaluation_all_outputs",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        download_mode="force_redownload",
    )["eval"]


def ALPACAFARM_GOLD_CROSSANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        download_mode="force_redownload",
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


def ALPACAFARM_GOLD_ANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_annotations",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        download_mode="force_redownload",
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


PRECOMPUTED_LEADERBOARDS = {
    (str(ALPACAFARM_REFERENCE_OUTPUTS), "alpaca_farm"): CUR_DIR
    / "leaderboards/AlpacaFarm/alpaca_farm_leaderboard.csv",
    (str(ALPACAFARM_REFERENCE_OUTPUTS), "claude"): CUR_DIR
    / "leaderboards/AlpacaFarm/claude_leaderboard.csv",
}

CURRENT_USER = getpass.getuser()
if CURRENT_USER in ["yanndubs"]:
    DEFAULT_CACHE_DIR = "/juice5/scr5/nlp/crfm/human-feedback/cache"
