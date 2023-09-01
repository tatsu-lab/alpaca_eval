import getpass
import os
from pathlib import Path

import datasets

### API specific ###
OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", None))
if isinstance(OPENAI_API_KEYS, str):
    OPENAI_API_KEYS = OPENAI_API_KEYS.split(",")
OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")
OPENAI_MAX_CONCURRENCY = int(os.environ.get("OPENAI_MAX_CONCURRENCY", 5))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
ANTHROPIC_MAX_CONCURRENCY = int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", 1))

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)

DATASETS_TOKEN = os.environ.get("DATASETS_TOKEN", None)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
DATASETS_FORCE_DOWNLOAD = os.environ.get("DATASETS_FORCE_DOWNLOAD", False)
########################

DEFAULT_CACHE_DIR = None
CURRENT_DIR = Path(__file__).parent
EVALUATORS_CONFIG_DIR = CURRENT_DIR / "evaluators_configs"
MODELS_CONFIG_DIR = CURRENT_DIR / "models_configs"
BASE_DIR = Path(__file__).parents[2]

MINIMAL_EVALUATORS = (
    "alpaca_eval_gpt4",
    "aviary_gpt4",
    "gpt4",
    "claude",
    "text_davinci_003",
    "chatgpt",
    "lmsys_gpt4",
    "humans",
    "alpaca_farm_greedy_gpt4",
)

VERIFIED_EVALUATORS = tuple(
    list(MINIMAL_EVALUATORS)
    + [
        "claude_ranking",
        "improved_aviary_gpt4",
        "improved_lmsys_gpt4",
        "lmsys_gpt4",
        "cohere",
        "alpaca_farm",
        "alpaca_farm_greedy_gpt4",
        "guanaco_33b",
        "longest",
    ]
)

# order matters i => i+1 when filtering
ORDERED_LEADERBOARD_MODES = ["minimal", "verified", "community"]


def ALPACAEVAL_REFERENCE_OUTPUTS():
    dataset = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
    )["eval"]
    return dataset


def ALPACAFARM_ALL_OUTPUTS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval_all_outputs",
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
    )["eval"]


def ALPACAFARM_GOLD_CROSSANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
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
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


ALPACAEVAL_LEADERBOARD_PATHS = CURRENT_DIR / "leaderboards/data_AlpacaEval"
PRECOMPUTED_LEADERBOARDS = {
    (str(ALPACAEVAL_REFERENCE_OUTPUTS), "claude"): ALPACAEVAL_LEADERBOARD_PATHS / "claude_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS), "alpaca_eval_gpt4"): ALPACAEVAL_LEADERBOARD_PATHS
    / "alpaca_eval_gpt4_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS), "chatgpt_fn"): ALPACAEVAL_LEADERBOARD_PATHS / "chatgpt_fn_leaderboard.csv",
}

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

EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE = [
    "Human agreement [%]",
    "Price [$/1000 examples]",
    "Time [seconds/1000 examples]",
    "Bias",
    "Variance",
    "Proba. prefer longer",
    "Proba. prefer lists",
    "Proba. prefer 1",
]
EVALUATORS_LEADERBOARD_COLS_TO_PRINT = EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE[:6]

CURRENT_USER = getpass.getuser()
if CURRENT_USER in ["yanndubs"]:
    DEFAULT_CACHE_DIR = "/juice5/scr5/nlp/crfm/human-feedback/cache"
