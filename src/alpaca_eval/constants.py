import getpass

DEFAULT_CACHE_DIR = None
DATASETS_TOKEN = None
OPENAI_ORGANIZATION_IDS = None

# MODELS_TO_BENCHMARK

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

EVALUATORS_TO_ANALYZE = tuple(list(LOCAL_EVALUATORS_TO_ANALYZE) + list(API_EVALUATORS_TO_ANALYZE) + ["humans",
                                                                                                     "length"])

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

CURRENT_USER = getpass.getuser()
if CURRENT_USER in ["yanndubs"]:
    DEFAULT_CACHE_DIR = "/juice5/scr5/nlp/crfm/human-feedback/cache"
