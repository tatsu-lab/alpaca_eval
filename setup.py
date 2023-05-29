import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "alpaca_eval", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find `__version__`.")

PACKAGES_DEV = ["pre-commit>=3.2.0", "black>=23.1.0", "isort"]
PACKAGES_ANALYSIS = [
    "seaborn",
    "matplotlib",
    "jupyterlab"
]
PACKAGES_LOCAL = ["accelerate", "transformers", "bitsandbytes", "xformers", "optimum", "scipy"]
PACKAGES_ALL_API = ["anthropic", "huggingface_hub", "cohere"]
PACKAGES_ALL = PACKAGES_LOCAL + PACKAGES_ALL_API + PACKAGES_ANALYSIS + PACKAGES_DEV

setuptools.setup(
    name="alpaca_eval",
    version="0.1.0",
    description="Automatic evaluation of instruction-following models by LLMs.",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    author="Alpaca Team",
    install_requires=[
        "datasets",
        "openai",
        "pandas",
        "tiktoken>=0.3.2",
        "fire",
    ],
    extras_require={
        "analysis": PACKAGES_ANALYSIS,
        "dev": PACKAGES_DEV,
        "all_api": PACKAGES_ALL_API,
        "all": PACKAGES_ALL,
    },
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "alpaca_eval=alpaca_eval:main",
        ],
    },
)
