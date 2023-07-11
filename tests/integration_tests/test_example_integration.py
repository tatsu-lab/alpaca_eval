import subprocess

import pytest

from alpaca_eval import main


@pytest.mark.slow
def test_cli_evaluate_example():
    result = subprocess.run(
        [
            "alpaca_eval",
            "--model_outputs",
            "example/outputs.json",
            "--max_instances",
            "3",
            "--annotators_config",
            "claude",
        ],
        capture_output=True,
        text=True,
    )
    normalized_output = " ".join(result.stdout.split())
    expected_output = " ".join("example 33.33 33.33 3".split())

    assert expected_output in normalized_output
