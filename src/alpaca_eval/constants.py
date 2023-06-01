import getpass
import os
from pathlib import Path

import datasets

### API specific ###
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
ANTHROPIC_MAX_CONCURRENCY = int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", 1))

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)

DATASETS_TOKEN = os.environ.get("DATASETS_TOKEN", None)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
########################

DEFAULT_CACHE_DIR = None
MODEL_LEADERBOARD = (
    "GPT-4",
    "ChatGPT",
    "AlpacaFarm PPO sim (gpt4 greedy 20k, step 350)",
    "AlpacaFarm PPO human (10k, step 40)",
    "Alpaca 7B",
    "Davinci003",
    "Davinci001",
)
EVALUATORS_TO_BENCHMARK = (
    "gpt4",
    "claude",
    "text_davinci_003",
    "chatgpt",
    "guanaco_33b",
    "gpt4_b5",
    "lmsys",
    "cohere",
    "oasst_pythia_12b",
    "humans"
)

API_EVALUATORS_TO_ANALYZE = (
    "gpt4",
    "claude",
    "text_davinci_003",
    "gpt4_b5",
    "aviary",
    "chatgpt",
    "guanaco_33b",
    "lmsys",
    "cohere",
    "alpaca_farm",
    "alpaca_farm_greedy",
)
LOCAL_EVALUATORS_TO_ANALYZE = (
    "oasst_pythia_12b",
    "stablelm_alpha_7b",
)

EVALUATORS_TO_ANALYZE = tuple(
    list(LOCAL_EVALUATORS_TO_ANALYZE)
    + list(API_EVALUATORS_TO_ANALYZE)
    + ["humans", "longest"]
)

HUMAN_ANNOTATED_MODELS_TO_KEEP = (
    "GPT-4 300 characters",
    "GPT-4",
    "AlpacaFarm PPO sim (step 40)",
    "ChatGPT",
    "ChatGPT 300 characters",
    "AlpacaFarm best-of-16 human",
    "AlpacaFarm PPO sim (gpt4 greedy, step 30)",
    "Davinci003",
    "AlpacaFarm ExpIter human (n=128)",
    "AlpacaFarm SFT 10K",
    "AlpacaFarm PPO human (10k, step 40)",
    "Alpaca 7B",
    "AlpacaFarm FeedMe human",
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
        # download_mode="force_redownload",
    )["eval"]


def ALPACAFARM_ALL_OUTPUTS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_evaluation_all_outputs",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        # download_mode="force_redownload",
    )["eval"]


def ALPACAFARM_GOLD_CROSSANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=DEFAULT_CACHE_DIR,
        use_auth_token=DATASETS_TOKEN,
        # download_mode="force_redownload",
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
        # download_mode="force_redownload",
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
    (str(ALPACAFARM_REFERENCE_OUTPUTS), "gpt4"): CUR_DIR
                                                 / "leaderboards/AlpacaFarm/gpt4_leaderboard.csv",
}

CURRENT_USER = getpass.getuser()
if CURRENT_USER in ["yanndubs"]:
    DEFAULT_CACHE_DIR = "/juice5/scr5/nlp/crfm/human-feedback/cache"
